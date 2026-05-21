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
from pathlib import Path

import zenoh
from pydantic_ai import BinaryContent

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from litter_detection.config import Settings
from litter_detection.agent.actions import block_movement, publish_zenoh_alert
from litter_detection.agent.detector import LitterDetector
from litter_detection.agent.exploreAgent import ExploreAgent
from litter_detection.agent.models import VerifiedDetection, VerifierDeps
from litter_detection.agent.verifier import verifier_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("coordinator")

settings = Settings()


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
        self.explore_agent = ExploreAgent()
        self._explore_done = threading.Event()

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
        last_alert_time: float = 0.0

        try:
            while True:
                frame = await loop.run_in_executor(None, detector.next_frame)
                if frame is None:
                    continue

                frame_bytes, frame_h, frame_w = frame
                result, annotated_frame = await loop.run_in_executor(
                    None, detector.infer, frame_bytes, frame_h, frame_w
                )

                if not result.litter_detected:
                    continue

                if time.time() - last_alert_time < settings.ALERT_COOLDOWN_S:
                    logger.debug("Detector: cooldown active — skipping frame")
                    continue

                if settings.USE_VERIFIER:
                    verify_image = (
                        annotated_frame if annotated_frame is not None else frame_bytes
                    )
                    logger.info(
                        f"Detector: ML flagged litter "
                        f"(coverage={result.pixel_coverage:.2%}) — verifying"
                    )
                    try:
                        verifier_result = await verifier_agent.run(
                            [
                                BinaryContent(data=verify_image, media_type="image/jpeg"),
                                (
                                    f"The segmentation model detected potential litter covering "
                                    f"{result.pixel_coverage:.2%} of this frame "
                                    f"({result.frame_width}x{result.frame_height} px). "
                                    "Red-highlighted pixels show what the model flagged. "
                                    "Please confirm whether the highlighted area actually contains litter."
                                ),
                            ],
                            deps=VerifierDeps(detection=result),
                        )
                    except Exception:
                        logger.exception("Detector: verifier failed — skipping frame")
                        continue

                    verified = verifier_result.data
                    if not verified.litter_confirmed:
                        logger.info(
                            f"Detector: verifier rejected detection "
                            f"(confidence={verified.confidence}): {verified.description}"
                        )
                        continue

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

                last_alert_time = time.time()

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

        finally:
            detector.stop()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
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

        detector_task = asyncio.create_task(self._run_detector())
        logger.info("Coordinator: detector started")

        try:
            while not self._explore_done.is_set():
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            logger.info("Coordinator: interrupted — stopping exploration")
            self.explore_agent.stop_exploration()
            raise
        finally:
            logger.info("Coordinator: coverage complete — shutting down detector")
            detector_task.cancel()
            try:
                await detector_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Coordinator: detector task raised an unexpected exception")
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
