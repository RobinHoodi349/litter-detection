"""Zenoh-Gateway für Roboterbewegungsbefehle."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from litter_detection.agent.models import MovementCommand
from litter_detection.config import Settings

logger = logging.getLogger("move-agent")

AUTONOMOUS_MAX_X_MPS = 0.3
AUTONOMOUS_MAX_Y_MPS = 0.3
AUTONOMOUS_MAX_Z_DEG_PER_S = 30.0

_settings = Settings()
DEFAULT_MOVEMENT_TOPIC = _settings.topic_movement_command
DEFAULT_ZENOH_ROUTER = _settings.ZENOH_ROUTER


class RobotMotionGateway:
    """Kapselt die Zenoh-Ausgabe und erlaubt trockene Testläufe ohne Roboter."""

    def __init__(
        self,
        router: str = DEFAULT_ZENOH_ROUTER,
        movement_topic: str = DEFAULT_MOVEMENT_TOPIC,
        dry_run: bool = False,
    ) -> None:
        self.router = router
        self.movement_topic = movement_topic
        self.dry_run = dry_run
        self.published_commands: list[MovementCommand] = []
        self._session: Any | None = None

    @classmethod
    def from_env(cls) -> "RobotMotionGateway":
        """Erstellt ein Gateway aus Settings; dry_run via MOVE_AGENT_DRY_RUN."""

        dry_run = _env_bool("MOVE_AGENT_DRY_RUN", default=False)
        return cls(
            router=_settings.ZENOH_ROUTER,
            movement_topic=_settings.topic_movement_command,
            dry_run=dry_run,
        )

    def publish_movement(self, command: MovementCommand) -> None:
        """Publiziert einen Bewegungsbefehl oder speichert ihn im Dry Run."""

        self.published_commands.append(command)
        if self.dry_run:
            logger.info("Dry Run: %s -> %s", self.movement_topic, command.model_dump())
            return

        session = self._ensure_session()
        session.put(self.movement_topic, command.model_dump_json().encode())
        logger.info("Published movement command to %s", self.movement_topic)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session

        try:
            import zenoh
        except ImportError as exc:
            raise RuntimeError(
                "Zenoh ist nicht installiert. Bitte `uv sync` ausführen oder "
                "MOVE_AGENT_DRY_RUN=1 setzen."
            ) from exc

        conf = zenoh.Config()
        if self.router:
            conf.insert_json5("connect/endpoints", json.dumps([self.router]))
        self._session = zenoh.open(conf)
        return self._session


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
