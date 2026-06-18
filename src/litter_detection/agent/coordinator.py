"""Mission coordinator: runs ExploreAgent and LitterDetector concurrently.

- ExploreAgent runs in a background thread (synchronous path-following loop).
- LitterDetector runs as an asyncio task (camera frame processing + LLM verifier).
- When the detector confirms litter it calls ExploreAgent.handle_request(BLOCK)
  directly, then block_movement() for the robot hardware (reverse + Zenoh flag),
  then ExploreAgent.handle_request(UNBLOCK) once the hold duration has elapsed.
- The mission ends when the entire coverage path has been walked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import zenoh
from pydantic_ai import BinaryContent

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from litter_detection.config import Settings
from litter_detection.agent.actions import block_movement, publish_zenoh_alert
from litter_detection.agent.detector import LitterDetector, crop_to_detection
from litter_detection.agent.exploreAgent import ExploreAgent
from litter_detection.agent.models import DetectionResult, VerifiedDetection, VerifierDeps
from litter_detection.agent.verifier import verifier_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("coordinator")

settings = Settings()


@dataclass
class _VerifyJob:
    """A litter detection awaiting LLM verification, decoupled from inference."""

    result: DetectionResult
    verify_image: bytes


class MissionCoordinator:
    def __init__(
        self,
        session: zenoh.Session,
        field_size: dict,
        lane_spacing_m: float,
    ) -> None:
        self.session = session
        self.field_size = field_size
        self.lane_spacing_m = lane_spacing_m
        self.explore_agent = ExploreAgent(session=session)
        self._explore_done = threading.Event()
        # Verifier runs off the inference loop: a single worker drains this
        # bounded queue, so inference keeps going while one verify is in flight.
        self._verify_queue: asyncio.Queue[_VerifyJob] | None = None
        self._verify_busy: bool = False
        self._last_alert_time: float = 0.0

    # ------------------------------------------------------------------
    # Dashboard command handler
    # ------------------------------------------------------------------

    def _on_dashboard_command(self, sample: zenoh.Sample) -> None:
        """Handle control commands published by the dashboard."""
        try:
            payload = json.loads(bytes(sample.payload).decode())
        except Exception:
            logger.warning("Coordinator: could not parse dashboard command payload")
            return
        action = payload.get("action", "")
        logger.info("Coordinator: dashboard command received — action=%s", action)
        if action == "stop":
            self.explore_agent.stop_exploration()
            self.session.publish(settings.topic_robodog_command, json.dumps({"command": "stop"}).encode())
            self._explore_done.set()
        elif action == "manual_override":
            self.explore_agent.handle_request({"type": "BLOCK", "reason": "manual_override"})
        elif action == "start_autonomous":
            self.explore_agent.handle_request({"type": "UNBLOCK"})

    # ------------------------------------------------------------------
    # Exploration thread
    # ------------------------------------------------------------------

    def _run_exploration(self) -> None:
        request = {
            "type": "START_EXPLORATION",
            "from": "coordinator",
            "field_size": self.field_size,
            "lane_spacing_m": self.lane_spacing_m,
        }
        result = self.explore_agent.handle_request(request)
        logger.info(f"Exploration finished: {result}")
        self._explore_done.set()

    # ------------------------------------------------------------------
    # Detector async task
    # ------------------------------------------------------------------

    async def _run_detector(self) -> None:
        loop = asyncio.get_running_loop()
        logger.info("Coordinator: loading detector model …")
        try:
            detector = await loop.run_in_executor(None, LitterDetector, self.session)
        except Exception:
            logger.exception("Coordinator: failed to initialise LitterDetector — detector disabled")
            return
        detector.start()
        logger.info("Coordinator: detector ready — listening for frames on %s", settings.topic_frame)
        last_inference_time: float = 0.0

        try:
            while True:
                frame = await loop.run_in_executor(None, detector.next_frame)
                if frame is None:
                    continue

                if time.time() - last_inference_time < settings.INFERENCE_MIN_INTERVAL_S:
                    continue
                last_inference_time = time.time()

                frame_bytes, frame_h, frame_w = frame
                result, annotated_frame = await loop.run_in_executor(
                    None, detector.infer, frame_bytes, frame_h, frame_w
                )

                if not result.litter_detected:
                    continue

                # Decouple verification from inference: skip while a verify is in
                # flight or the alert cooldown is active, otherwise the inference
                # loop would block waiting for the (slow) vision model.
                if self._verify_busy:
                    logger.debug("Detector: verifier busy — skipping frame")
                    continue

                if time.time() - self._last_alert_time < settings.ALERT_COOLDOWN_S:
                    logger.debug("Detector: cooldown active — skipping frame")
                    continue

                if annotated_frame is not None and result.bbox is not None:
                    verify_image = crop_to_detection(
                        annotated_frame, result.bbox, result.frame_height, result.frame_width
                    )
                else:
                    verify_image = annotated_frame if annotated_frame is not None else frame_bytes

                self._enqueue_verification(_VerifyJob(result=result, verify_image=verify_image))

        finally:
            detector.stop()

    def _enqueue_verification(self, job: _VerifyJob) -> None:
        """Hand a detection to the verify worker without blocking inference."""
        if self._verify_queue is None:
            return
        try:
            self._verify_queue.put_nowait(job)
            self._verify_busy = True
        except asyncio.QueueFull:
            logger.debug("Detector: verification queue full — dropping frame")

    async def _run_verify_worker(self) -> None:
        """Single consumer: process queued detections one at a time."""
        assert self._verify_queue is not None
        loop = asyncio.get_running_loop()
        while True:
            job = await self._verify_queue.get()
            try:
                await self._process_verification(loop, job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Coordinator: verification worker error")
            finally:
                self._verify_busy = False
                self._verify_queue.task_done()

    async def _process_verification(self, loop: asyncio.AbstractEventLoop, job: _VerifyJob) -> None:
        result = job.result
        if settings.USE_VERIFIER:
            logger.info(
                f"Detector: ML flagged litter "
                f"(coverage={result.pixel_coverage:.2%}) — verifying"
            )
            try:
                verifier_result = await asyncio.wait_for(
                    verifier_agent.run(
                        [
                            BinaryContent(data=job.verify_image, media_type="image/jpeg"),
                            (
                                f"The ML segmentation model flagged a region covering "
                                f"{result.pixel_coverage:.2%} of the original camera frame. "
                                "This image is a cropped view of that region — red pixels mark "
                                "exactly what the model detected. "
                                "Does the red-highlighted area contain actual litter? "
                                "Reject if it looks like natural ground, shadows, or image noise."
                            ),
                        ],
                        deps=VerifierDeps(detection=result),
                    ),
                    timeout=settings.VERIFIER_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Detector: verifier timed out after %.0fs — is Ollama running? (ollama serve)",
                    settings.VERIFIER_TIMEOUT_S,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Detector: verifier failed (%s: %s) — skipping frame",
                    type(exc).__name__, exc,
                )
                return

            verified = verifier_result.data
            if not verified.litter_confirmed:
                logger.info(
                    f"Detector: verifier rejected detection "
                    f"(confidence={verified.confidence}): {verified.description}"
                )
                return

            logger.info(
                f"Detector: litter confirmed "
                f"(confidence={verified.confidence}): {verified.description}"
            )
            await publish_zenoh_alert(self.session, result, verified)
        else:
            verified = VerifiedDetection(
                litter_confirmed=True,
                confidence="low",
                description="Verifier disabled — ML model decision only",
            )
            logger.info(
                f"Detector: litter detected "
                f"(coverage={result.pixel_coverage:.2%}) — verifier disabled"
            )
            await publish_zenoh_alert(self.session, result, verified)

        self._last_alert_time = time.time()

        if result.pixel_coverage > 0.05:
            # Tell ExploreAgent to pause directly — no Zenoh round-trip needed
            # since coordinator has direct access to both agents.
            await loop.run_in_executor(
                None,
                self.explore_agent.handle_request,
                {"type": "BLOCK", "reason": "litter_detected"},
            )
            # block_movement publishes to Zenoh for the robot hardware
            # (reverse command + blocked flag) and waits for the hold duration.
            await block_movement(self.session, result)
            await loop.run_in_executor(
                None,
                self.explore_agent.handle_request,
                {"type": "UNBLOCK"},
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        dashboard_sub = self.session.declare_subscriber(
            settings.topic_robodog_command, self._on_dashboard_command
        )
        logger.info("Coordinator: subscribed to dashboard commands on %s", settings.topic_robodog_command)

        explore_thread = threading.Thread(
            target=self._run_exploration,
            daemon=True,
            name="explore-thread",
        )
        explore_thread.start()
        logger.info(
            f"Coordinator: exploration started "
            f"(field={self.field_size}, lane_spacing={self.lane_spacing_m}m)"
        )

        self._verify_queue = asyncio.Queue(maxsize=1)
        verify_task = asyncio.create_task(self._run_verify_worker())
        detector_task = asyncio.create_task(self._run_detector())
        logger.info("Coordinator: detector + verify worker started")

        try:
            while not self._explore_done.is_set():
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            logger.info("Coordinator: interrupted — stopping exploration")
            self.explore_agent.stop_exploration()
            raise
        finally:
            dashboard_sub.undeclare()
            logger.info("Coordinator: coverage complete — shutting down detector")
            for task in (detector_task, verify_task):
                task.cancel()
            for task in (detector_task, verify_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Coordinator: detector/verify task raised an unexpected exception")
            explore_thread.join(timeout=5.0)
            logger.info("Coordinator: mission complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Litter Detection Mission Coordinator")
    parser.add_argument("--width", type=float, default=5.0, help="Field width in metres")
    parser.add_argument("--height", type=float, default=5.0, help="Field height in metres")
    parser.add_argument(
        "--lane-spacing", type=float, default=0.5, help="Lane spacing in metres"
    )
    args = parser.parse_args()

    logger.info("Coordinator: opening Zenoh session → %s", settings.ZENOH_ROUTER)
    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", json.dumps([settings.ZENOH_ROUTER]))
    session = zenoh.open(conf)
    logger.info("Coordinator: Zenoh session open")

    logger.info("Coordinator: initialising MissionCoordinator (PathPlanner + ExploreAgent) …")
    coordinator = MissionCoordinator(
        session=session,
        field_size={"width_m": args.width, "height_m": args.height},
        lane_spacing_m=args.lane_spacing,
    )
    logger.info("Coordinator: MissionCoordinator ready")

    try:
        asyncio.run(coordinator.run())
    except KeyboardInterrupt:
        logger.info("Coordinator: mission aborted by user")
    finally:
        session.close()


if __name__ == "__main__":
    main()
