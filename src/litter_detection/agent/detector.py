import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import zenoh

from litter_detection.config import Settings
from litter_detection.agent.models import DetectionResult

logger = logging.getLogger("litter-detector-agent")
settings = Settings()

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class LitterDetector:
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
        # pretrained=False: the checkpoint below supplies all weights, so we
        # avoid the ImageNet backbone download (fails on an offline Jetson).
        model = settings.MODEL_CLASS(dropout=settings.DROPOUT, pretrained=False).to(self._device)
        model.load_state_dict(torch.load(checkpoint, map_location=self._device))
        model.eval()
        n = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded {checkpoint.name} on {self._device} ({n:,} params)")
        return model

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
            logger.exception("LitterDetector: failed to enqueue frame")

    def start(self) -> None:
        self._subscriber = self._session.declare_subscriber(settings.topic_frame, self._on_frame)
        logger.info(f"LitterDetector: subscribed to {settings.topic_frame}")

    def stop(self) -> None:
        if self._subscriber is not None:
            self._subscriber.undeclare()
            logger.info("LitterDetector: subscriber closed")

    def next_frame(self, timeout: float = 1.0) -> tuple[bytes, int, int] | None:
        if not self._event.wait(timeout):
            return None

        frame_bytes = ts = h = w = None
        with self._lock:
            while self._queue:
                frame_bytes, ts, h, w = self._queue.popleft()
                age = time.perf_counter() - ts
                if age <= self.FRAME_MAX_AGE_S:
                    break
                logger.debug(f"LitterDetector: skipping stale frame (age={age:.2f}s)")
                frame_bytes = None
            if not self._queue:
                self._event.clear()

        if frame_bytes is None:
            return None
        return frame_bytes, h, w

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

    def _overlay_mask(
        self, frame_bytes: bytes, mask_binary: np.ndarray, frame_h: int, frame_w: int
    ) -> bytes:
        frame_bgr = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        mask_resized = cv2.resize(
            mask_binary, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
        )
        overlay = frame_bgr.copy()
        overlay[mask_resized == 1] = [0, 0, 255]  # red highlight on detected pixels
        annotated = cv2.addWeighted(frame_bgr, 0.6, overlay, 0.4, 0)
        _, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return encoded.tobytes()

    def infer(
        self, frame_bytes: bytes, frame_h: int, frame_w: int
    ) -> tuple[DetectionResult, bytes | None]:
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
            ), None

        mask_probs = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
        mask_binary = (mask_probs > settings.THRESHOLD).astype(np.uint8)
        coverage = float(mask_binary.mean())

        # Compute tight bounding box in original frame coordinates
        mask_frame = cv2.resize(mask_binary, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        rows = np.any(mask_frame, axis=1)
        cols = np.any(mask_frame, axis=0)
        if rows.any() and cols.any():
            y_idx = np.where(rows)[0]
            x_idx = np.where(cols)[0]
            bbox: tuple[int, int, int, int] | None = (
                int(x_idx[0]), int(y_idx[0]), int(x_idx[-1]), int(y_idx[-1])
            )
        else:
            bbox = None

        result = DetectionResult(
            timestamp=time.time(),
            litter_detected=coverage > settings.LITTER_COVERAGE_THRESHOLD,
            pixel_coverage=coverage,
            mask_shape=(int(mask_binary.shape[0]), int(mask_binary.shape[1])),
            frame_height=frame_h,
            frame_width=frame_w,
            bbox=bbox,
        )
        annotated_frame = self._overlay_mask(frame_bytes, mask_binary, frame_h, frame_w)
        return result, annotated_frame


def crop_to_detection(
    frame_bytes: bytes,
    bbox: tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
    padding_px: int = 40,
    min_frac: float = 0.5,
) -> bytes:
    """Crop frame to the detected litter region with padding, re-encoded as JPEG.

    Litter typically covers only ~3% of the frame, so the raw bbox is tiny and
    gives the verifier almost no surrounding context. On top of the fixed
    ``padding_px`` we therefore guarantee the crop spans at least ``min_frac`` of
    each frame dimension, expanded symmetrically around the bbox centre (then
    clamped to the frame). This keeps the litter centred while showing enough
    background to tell real litter from natural ground.
    """
    x1, y1, x2, y2 = bbox
    x1 -= padding_px
    y1 -= padding_px
    x2 += padding_px
    y2 += padding_px

    # Enforce a minimum crop size (fraction of the frame) around the bbox centre.
    min_w = int(frame_w * min_frac)
    min_h = int(frame_h * min_frac)
    if (x2 - x1) < min_w:
        cx = (x1 + x2) / 2
        x1 = int(cx - min_w / 2)
        x2 = int(cx + min_w / 2)
    if (y2 - y1) < min_h:
        cy = (y1 + y2) / 2
        y1 = int(cy - min_h / 2)
        y2 = int(cy + min_h / 2)

    # Clamp to frame bounds (shift-free: just clip, the centre stays close enough).
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w - 1, x2)
    y2 = min(frame_h - 1, y2)

    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    crop = img[y1 : y2 + 1, x1 : x2 + 1]
    _, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes()
