import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import zenoh
from pydantic_ai import BinaryContent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTO_RESEARCH_DIR = PROJECT_ROOT / "auto-research"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(AUTO_RESEARCH_DIR))

from config import Settings
from agent.actions import block_movement, publish_zenoh_alert
from agent.detector import LitterDetector
from agent.models import VerifiedDetection
from agent.verifier import verifier_agent, VerifierDeps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("litter-detector-agent")

settings = Settings()


async def run(zenoh_session: zenoh.Session) -> None:
    detector = LitterDetector(zenoh_session)
    detector.start()
    last_alert_time: float = 0.0

    try:
        while True:
            frame = await asyncio.get_event_loop().run_in_executor(
                None, detector.next_frame
            )
            if frame is None:
                continue

            frame_bytes, frame_h, frame_w = frame
            result, annotated_frame = await asyncio.get_event_loop().run_in_executor(
                None, detector.infer, frame_bytes, frame_h, frame_w
            )

            if not result.litter_detected:
                continue

            if time.time() - last_alert_time < settings.ALERT_COOLDOWN_S:
                logger.debug("Cooldown active — skipping frame")
                continue

            if settings.USE_VERIFIER:
                logger.info(
                    f"ML model flagged litter (coverage={result.pixel_coverage:.2%}) "
                    f"— sending to verifier"
                )
                verify_image = annotated_frame if annotated_frame is not None else frame_bytes
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
                    logger.exception("Verifier agent failed — skipping frame")
                    continue

                verified = verifier_result.data

                if not verified.litter_confirmed:
                    logger.info(
                        f"Verifier rejected ML detection "
                        f"(confidence={verified.confidence}): {verified.description}"
                    )
                    continue

                logger.info(
                    f"Litter confirmed by verifier "
                    f"(confidence={verified.confidence}): {verified.description}"
                )
                await publish_zenoh_alert(zenoh_session, result, verified)
            else:
                logger.info(
                    f"Litter detected (coverage={result.pixel_coverage:.2%}) "
                    f"— verifier disabled, reacting directly"
                )
                await publish_zenoh_alert(
                    zenoh_session,
                    result,
                    VerifiedDetection(
                        litter_confirmed=True,
                        confidence="low",
                        description="Verifier disabled — ML model decision only",
                    ),
                )

            last_alert_time = time.time()

            if result.pixel_coverage > 0.05:
                await block_movement(zenoh_session, result)

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
