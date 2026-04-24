import sys
import asyncio
import threading
import time
import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import zenoh
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTO_RESEARCH_DIR = PROJECT_ROOT / "auto-research"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(AUTO_RESEARCH_DIR))

from config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("litter-detector-agent")

settings = Settings()

# ── Pydantic Models ────────────────────────────────────────────────────────────
class DetectionResult(BaseModel):
    timestamp: float
    litter_detected: bool
    pixel_coverage: float       # fraction of pixels classified as litter (0..1)
    mask_shape: tuple[int, int] # (height, width) of the inference mask
    frame_height: int
    frame_width: int


@dataclass
class ReactorDeps:
    zenoh_session: zenoh.Session
    detection: DetectionResult


# ── Reactor Agent ──────────────────────────────────────────────────────────────
reactor_agent = Agent(
    "claude-sonnet-4-6",
    deps_type=ReactorDeps,
    system_prompt=(
        "You are a litter-detection reactor. The robodog patrols an area autonomously. "
        "Because detection has latency, the robodog may have already walked past the litter "
        "by the time this alert fires. "
        "You must always: "
        "1) publish the Zenoh alert, "
        "2) block the movement agent and send a reverse command so the robodog backtracks "
        "to the litter position — this happens automatically inside block_movement_agent. "
        "Only call block_movement_agent when pixel_coverage exceeds 5%%; for lower coverage "
        "call publish_zenoh_alert only."
    ),
)


@reactor_agent.tool
async def publish_zenoh_alert(ctx: RunContext[ReactorDeps]) -> str:
    """Publish a structured litter alert to the 'litter/alert' Zenoh topic."""
    d = ctx.deps.detection
    payload = {
        "timestamp": d.timestamp,
        "litter_detected": d.litter_detected,
        "pixel_coverage": round(d.pixel_coverage, 4),
        "mask_shape": list(d.mask_shape),
        "frame_height": d.frame_height,
        "frame_width": d.frame_width,
    }
    ctx.deps.zenoh_session.put(settings.topic_alert, json.dumps(payload).encode())
    logger.info(f"[ReactorAgent] Alert → {settings.topic_alert}  (coverage={d.pixel_coverage:.2%})")
    return f"Alert published (coverage={d.pixel_coverage:.2%})"


@reactor_agent.tool
async def send_robodog_command(ctx: RunContext[ReactorDeps]) -> str:
    """Send a reverse command to the robodog to backtrack to the litter position.

    Because detection latency may cause the robodog to walk past the litter before
    the alert fires, this command instructs it to reverse for a short duration so it
    ends up at the litter site.
    """
    d = ctx.deps.detection
    payload = {
        "action": "reverse",
        "duration_s": 2.0,
        "reason": "litter_detected_late",
        "pixel_coverage": round(d.pixel_coverage, 4),
        "timestamp": d.timestamp,
    }
    ctx.deps.zenoh_session.put(settings.topic_robodog_command, json.dumps(payload).encode())
    logger.info(
        f"[ReactorAgent] Robodog reverse command → {settings.topic_robodog_command}  "
        f"(coverage={d.pixel_coverage:.2%})"
    )
    return "Robodog reverse command sent (duration=2.0 s)"


@reactor_agent.tool
async def block_movement_agent(ctx: RunContext[ReactorDeps]) -> str:
    """Block the movement agent for 5 seconds, then release it.

    Publishes a blocked signal on the Zenoh movement topic, waits 5 seconds,
    then publishes an unblock signal on the same topic.
    The explore agent subscribes to this topic and pauses until the signal is cleared.
    Use this when litter coverage is significant enough to alert to the litter.
    """
    d = ctx.deps.detection
    session = ctx.deps.zenoh_session

    block_payload = {
        "blocked": True,
        "reason": "litter_detected",
        "pixel_coverage": round(d.pixel_coverage, 4),
        "timestamp": d.timestamp,
    }
    session.put(settings.topic_movement_blocked, json.dumps(block_payload).encode())
    logger.info(
        f"[ReactorAgent] Movement blocked → {settings.topic_movement_blocked}  "
        f"(coverage={d.pixel_coverage:.2%})"
    )

    await send_robodog_command(ctx)

    await asyncio.sleep(5)

    unblock_payload = {
        "blocked": False,
        "reason": "timed_release",
        "pixel_coverage": round(d.pixel_coverage, 4),
        "timestamp": time.time(),
    }
    session.put(settings.topic_movement_blocked, json.dumps(unblock_payload).encode())
    logger.info(f"[ReactorAgent] Movement unblocked → {settings.topic_movement_blocked}")

    return f"Movement agent blocked for 5 s, then released (coverage={d.pixel_coverage:.2%})"


# ── Detector Sub-Agent ─────────────────────────────────────────────────────────
class DetectorSubAgent:
    """
    Subscribes to Zenoh 'litter/frame', runs  inference on
    each frame, and returns DetectionResult objects via infer().
    """

    FRAME_MAX_AGE_S = 0.5
    INPUT_SIZE = 512

    def __init__(self, zenoh_session: zenoh.Session) -> None:
        self._session = zenoh_session
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model()
        self._queue: deque[tuple[bytes, float, int, int]] = deque(maxlen=20)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._subscriber: zenoh.Subscriber | None = None

    def _load_model(self) -> torch.nn.Module:
        checkpoint = PROJECT_ROOT / settings.MODEL_NAME
        if not checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
        model = settings.MODEL_CLASS(dropout=settings.DROPOUT).to(self._device)
        model.load_state_dict(torch.load(checkpoint, map_location=self._device))
        model.eval()
        n = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded {checkpoint.name} on {self._device} ({n:,} params)")
        return model

    # ── Zenoh callback ─────────────────────────────────────────────────────────
    def _on_frame(self, sample: zenoh.Sample) -> None:
        try:
            frame_bytes = bytes(sample.payload)
            arr = np.frombuffer(frame_bytes, np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return
            h, w = frame_bgr.shape[:2]
            with self._lock:
                self._queue.append((frame_bytes, time.perf_counter(), h, w))
                self._event.set()
        except Exception:
            logger.exception("DetectorSubAgent: failed to enqueue frame")

    def start(self) -> None:
        self._subscriber = self._session.declare_subscriber(settings.topic_frame, self._on_frame)
        logger.info(f"DetectorSubAgent: subscribed to {settings.topic_frame}")

    def stop(self) -> None:
        if self._subscriber is not None:
            self._subscriber.undeclare()
            logger.info("DetectorSubAgent: subscriber closed")

    # ── Frame access ───────────────────────────────────────────────────────────
    def next_frame(self, timeout: float = 1.0) -> tuple[bytes, int, int] | None:
        """Return the freshest frame (bytes, height, width), or None if none available."""
        if not self._event.wait(timeout):
            return None

        frame_bytes = ts = h = w = None
        with self._lock:
            while self._queue:
                frame_bytes, ts, h, w = self._queue.popleft()
                age = time.perf_counter() - ts
                if age <= self.FRAME_MAX_AGE_S:
                    break
                logger.debug(f"DetectorSubAgent: skipping stale frame (age={age:.2f}s)")
                frame_bytes = None
            if not self._queue:
                self._event.clear()

        if frame_bytes is None:
            return None
        return frame_bytes, h, w

    # ── Preprocessing ──────────────────────────────────────────────────────────
    def _preprocess(self, frame_bytes: bytes) -> torch.Tensor:
        img_rgb = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        if img_rgb.shape[0] > self.INPUT_SIZE or img_rgb.shape[1] > self.INPUT_SIZE:
            scale = min(self.INPUT_SIZE / img_rgb.shape[0], self.INPUT_SIZE / img_rgb.shape[1])
            new_w = int(img_rgb.shape[1] * scale)
            new_h = int(img_rgb.shape[0] * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div(255.0)
            .to(self._device)
        )

    # ── Inference ──────────────────────────────────────────────────────────────
    def infer(self, frame_bytes: bytes, frame_h: int, frame_w: int) -> DetectionResult:
        img_tensor = self._preprocess(frame_bytes)
        with torch.no_grad():
            output = self._model(img_tensor)

        if not isinstance(output, torch.Tensor):
            logger.warning("Model returned unexpected output type — skipping frame")
            return DetectionResult(
                timestamp=time.time(),
                litter_detected=False,
                pixel_coverage=0.0,
                mask_shape=(0, 0),
                frame_height=frame_h,
                frame_width=frame_w,
            )

        mask_probs = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
        mask_binary = (mask_probs > settings.THRESHOLD).astype(np.uint8)
        coverage = float(mask_binary.mean())

        return DetectionResult(
            timestamp=time.time(),
            litter_detected=coverage > settings.LITTER_COVERAGE_THRESHOLD,
            pixel_coverage=coverage,
            mask_shape=(int(mask_binary.shape[0]), int(mask_binary.shape[1])),
            frame_height=frame_h,
            frame_width=frame_w,
        )


# ── Main loop ──────────────────────────────────────────────────────────────────
async def run(zenoh_session: zenoh.Session) -> None:
    detector = DetectorSubAgent(zenoh_session)
    detector.start()

    try:
        while True:
            frame = await asyncio.get_event_loop().run_in_executor(
                None, detector.next_frame
            )
            if frame is None:
                continue

            frame_bytes, frame_h, frame_w = frame
            result = await asyncio.get_event_loop().run_in_executor(
                None, detector.infer, frame_bytes, frame_h, frame_w
            )

            logger.info(
                f"Frame processed: litter={result.litter_detected} "
                f"coverage={result.pixel_coverage:.2%}"
            )

            if result.litter_detected:
                logger.info("Litter confirmed — handing off to ReactorAgent")
                await reactor_agent.run(
                    f"Litter detected with {result.pixel_coverage:.2%} pixel coverage. "
                    f"Frame size: {result.frame_width}x{result.frame_height}.",
                    deps=ReactorDeps(zenoh_session=zenoh_session, detection=result),
                )
    finally:
        detector.stop()


def main() -> None:
    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", json.dumps([settings.ZENOH_ROUTER]))
    session = zenoh.open(conf)
    try:
        asyncio.run(run(session))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        session.close()


if __name__ == "__main__":
    main()
