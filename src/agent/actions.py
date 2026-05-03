import asyncio
import json
import logging
import time

import zenoh

from config import Settings
from agent.models import DetectionResult, VerifiedDetection

logger = logging.getLogger("litter-detector-agent")
settings = Settings()


async def publish_zenoh_alert(
    session: zenoh.Session,
    detection: DetectionResult,
    verified: VerifiedDetection,
) -> None:
    payload = {
        "timestamp": detection.timestamp,
        "litter_detected": True,
        "pixel_coverage": round(detection.pixel_coverage, 4),
        "mask_shape": list(detection.mask_shape),
        "frame_height": detection.frame_height,
        "frame_width": detection.frame_width,
        "verified_by_llm": True,
        "confidence": verified.confidence,
        "description": verified.description,
    }
    session.put(settings.topic_alert, json.dumps(payload).encode())
    logger.info(
        f"[Alert] → {settings.topic_alert}  "
        f"coverage={detection.pixel_coverage:.2%}  confidence={verified.confidence}"
    )


async def send_robodog_command(session: zenoh.Session, detection: DetectionResult) -> None:
    payload = {
        "action": "reverse",
        "duration_s": 1.0,
        "reason": "litter_detected_late",
        "pixel_coverage": round(detection.pixel_coverage, 4),
        "timestamp": detection.timestamp,
    }
    session.put(settings.topic_robodog_command, json.dumps(payload).encode())
    logger.info(f"[Command] Reverse → {settings.topic_robodog_command}")


async def block_movement(session: zenoh.Session, detection: DetectionResult) -> None:
    block_payload = {
        "blocked": True,
        "reason": "litter_detected",
        "pixel_coverage": round(detection.pixel_coverage, 4),
        "timestamp": detection.timestamp,
    }
    session.put(settings.topic_movement_blocked, json.dumps(block_payload).encode())
    logger.info(f"[Block] Movement blocked → {settings.topic_movement_blocked}")

    await send_robodog_command(session, detection)
    await asyncio.sleep(5)

    unblock_payload = {
        "blocked": False,
        "reason": "timed_release",
        "pixel_coverage": round(detection.pixel_coverage, 4),
        "timestamp": time.time(),
    }
    session.put(settings.topic_movement_blocked, json.dumps(unblock_payload).encode())
    logger.info(f"[Block] Movement unblocked → {settings.topic_movement_blocked}")
