"""Gemeinsame Bewegungsmodelle und Sicherheitslogik für den Move Agent."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("move-agent")

AUTONOMOUS_MAX_X_MPS = 0.3
AUTONOMOUS_MAX_Y_MPS = 0.3
AUTONOMOUS_MAX_Z_DEG_PER_S = 30.0
COMMAND_MAX_AGE_SECONDS = 1.0
DEFAULT_MOVEMENT_TOPIC = "robot/cmd/movement"
DEFAULT_ZENOH_ROUTER = "tcp/localhost:7447"


class MovementSource(str, Enum):
    """Quelle eines Bewegungsbefehls."""

    controller = "controller"
    autonomous = "autonomous"
    explore = "explore"


class MovementCommand(BaseModel):
    """Kompatibles Bewegungsmodell für die Roboterschnittstelle."""

    x: float = Field(default = 0.0, description = "Vorwärts-/Rückwärtsgeschwindigkeit in m/s.")
    y: float = Field(default = 0.0, description = "Seitwärtsgeschwindigkeit in m/s.")
    z_deg: float = Field(default = 0.0, description = "Drehgeschwindigkeit in Grad/s.")
    source: MovementSource = Field(default = MovementSource.explore)
    timestamp: float = Field(default_factory = time.time, description = "Unix-Zeitstempel in Sekunden.")

    def is_zero(self, tolerance: float = 1e-6) -> bool:
        """Gibt zurück, ob der Befehl keine Bewegung auslöst."""

        return (
            abs(self.x) <= tolerance
            and abs(self.y) <= tolerance
            and abs(self.z_deg) <= tolerance
        )

    def age_seconds(self, now: float | None = None) -> float:
        """Alter des Befehls in Sekunden."""

        return (time.time() if now is None else now) - self.timestamp

    def capped(self) -> "MovementCommand":
        """Begrenzt Agenten-Befehle auf die erlaubten Geschwindigkeiten."""

        if self.source == MovementSource.controller:
            return self

        return self.model_copy(
            update = {
                "x": _clamp(self.x, -AUTONOMOUS_MAX_X_MPS, AUTONOMOUS_MAX_X_MPS),
                "y": _clamp(self.y, -AUTONOMOUS_MAX_Y_MPS, AUTONOMOUS_MAX_Y_MPS),
                "z_deg": _clamp(
                    self.z_deg,
                    -AUTONOMOUS_MAX_Z_DEG_PER_S,
                    AUTONOMOUS_MAX_Z_DEG_PER_S,
                ),
            }
        )


class RobotPose(BaseModel):
    """Lokalisierte Roboterposition im Suchfeld."""

    x_m: float = 0.0
    y_m: float = 0.0
    heading_deg: float = 0.0

    @model_validator(mode = "after")
    def normalize_heading(self) -> "RobotPose":
        self.heading_deg = self.heading_deg % 360.0
        return self

    def moved(self, distance_m: float) -> "RobotPose":
        """Schätzt eine neue Pose nach Geradeauslauf entlang des Headings."""

        heading_rad = math.radians(self.heading_deg)
        return RobotPose(
            x_m = self.x_m + math.cos(heading_rad) * distance_m,
            y_m = self.y_m + math.sin(heading_rad) * distance_m,
            heading_deg = self.heading_deg,
        )

    def turned(self, angle_deg: float) -> "RobotPose":
        """Schätzt eine neue Pose nach einer Drehung auf der Stelle."""

        return RobotPose(
            x_m = self.x_m,
            y_m = self.y_m,
            heading_deg = self.heading_deg + angle_deg,
        )


class MovementToolResult(BaseModel):
    """Standardisierte Antwort eines Move-Agent-Tools."""

    ok: bool
    message: str
    command: MovementCommand | None = None
    stop_command: MovementCommand | None = None
    pose: RobotPose | None = None
    topic: str = DEFAULT_MOVEMENT_TOPIC
    dry_run: bool = True


class RobotMotionGateway:
    """Kapselt die Zenoh-Ausgabe und erlaubt trockene Testläufe ohne Roboter."""

    def __init__(
        self,
        router: str = DEFAULT_ZENOH_ROUTER,
        movement_topic: str = DEFAULT_MOVEMENT_TOPIC,
        dry_run: bool = True,
    ) -> None:
        self.router = router
        self.movement_topic = movement_topic
        self.dry_run = dry_run
        self.published_commands: list[MovementCommand] = []
        self._session: Any | None = None

    @classmethod
    def from_env(cls) -> "RobotMotionGateway":
        """Erstellt ein Gateway aus Umgebungsvariablen."""

        dry_run = _env_bool("MOVE_AGENT_DRY_RUN", default = True)
        return cls(
            router = os.getenv("ZENOH_ROUTER", DEFAULT_ZENOH_ROUTER),
            movement_topic = os.getenv("MOVE_AGENT_MOVEMENT_TOPIC", DEFAULT_MOVEMENT_TOPIC),
            dry_run = dry_run,
        )

    def publish_movement(self, command: MovementCommand) -> None:
        """Publiziert einen Bewegungsbefehl oder speichert ihn im Dry Run."""

        self.published_commands.append(command)
        if self.dry_run:
            logger.info("Dry Run: %s -> %s", self.movement_topic, command.model_dump())
            return

        session = self._ensure_session()
        session.put(self.movement_topic, command.model_dump_json())
        logger.info("Published movement command to %s", self.movement_topic)

    def close(self) -> None:
        """Schliesst die Zenoh-Session, falls eine geöffnet wurde."""

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
                "MOVE_AGENT_DRY_RUN = 1 setzen."
            ) from exc

        conf = zenoh.Config()
        if self.router:
            conf.insert_json5("connect/endpoints", json.dumps([self.router]))
        self._session = zenoh.open(conf)
        return self._session


@dataclass
class MotionState:
    """Laufzeitstatus des Move Agents."""

    current_pose: RobotPose | None = dataclass_field(default_factory=RobotPose)
    last_movement_command: MovementCommand = dataclass_field(default_factory=MovementCommand)
    emergency_stop_active: bool = False
    controller_override_active: bool = False
    controller_last_seen: float | None = None
    controller_idle_timeout_seconds: float = 1.0

    def register_controller_command(self, command: MovementCommand) -> None:
        """Merkt Controller-Aktivität für die Quellen-Hierarchie."""

        if command.source != MovementSource.controller:
            return

        self.controller_last_seen = command.timestamp
        self.controller_override_active = not command.is_zero()
        self.last_movement_command = command

    def controller_is_active(self, now: float | None = None) -> bool:
        """Prüft, ob der menschliche Controller noch Vorrang hat."""

        if not self.controller_override_active:
            return False

        current_time = time.time() if now is None else now
        if (
            self.controller_last_seen is not None
            and current_time - self.controller_last_seen > self.controller_idle_timeout_seconds
        ):
            self.controller_override_active = False
            return False

        return True


@dataclass
class MoveAgentDeps:
    """Abhängigkeiten, die Pydantic AI an die Move-Tools übergibt."""

    gateway: RobotMotionGateway = dataclass_field(default_factory=RobotMotionGateway.from_env)
    state: MotionState = dataclass_field(default_factory=MotionState)
    source: MovementSource = MovementSource.explore
    execute_real_time: bool = False
    auto_stop_after_tool: bool = True
    max_tool_duration_s: float = 10.0

    @classmethod
    def from_env(cls) -> "MoveAgentDeps":
        gateway = RobotMotionGateway.from_env()
        return cls(
            gateway=gateway,
            execute_real_time = _env_bool("MOVE_AGENT_REAL_TIME", default = not gateway.dry_run),
            auto_stop_after_tool = _env_bool("MOVE_AGENT_AUTO_STOP", default = True),
            max_tool_duration_s = float(os.getenv("MOVE_AGENT_MAX_TOOL_DURATION_S", "10.0")),
        )


def execute_timed_movement(
    deps: MoveAgentDeps,
    *,
    command: MovementCommand,
    duration_s: float,
    estimated_distance_m: float = 0.0,
    estimated_turn_deg: float = 0.0,
    source_timestamp: float | None = None,
) -> MovementToolResult:
    """Validiert, publiziert und optional stoppt einen Bewegungsbefehl."""

    now = time.time()
    stale_reason = _stale_reason(source_timestamp, now)
    if stale_reason is not None:
        return _rejected(stale_reason, deps)

    if command.age_seconds(now) > COMMAND_MAX_AGE_SECONDS:
        return _rejected("Bewegungsbefehl ist veraltet und wurde verworfen.", deps)

    if deps.state.emergency_stop_active:
        return _rejected("E-Stop ist aktiv. Es werden keine Bewegungen ausgegeben.", deps)

    if command.source != MovementSource.controller and deps.state.controller_is_active(now):
        return _rejected("Controller hat Vorrang. Agenten-Befehl pausiert.", deps)

    if duration_s <= 0:
        return _rejected("duration_s muss größer als 0 sein.", deps)

    if duration_s > deps.max_tool_duration_s:
        return _rejected(
            f"duration_s={duration_s:.2f}s überschreitet das Limit "
            f"von {deps.max_tool_duration_s:.2f}s.",
            deps,
        )

    safe_command = command.capped()
    next_pose = _estimate_next_pose(
        deps.state.current_pose,
        estimated_distance_m = estimated_distance_m,
        estimated_turn_deg = estimated_turn_deg,
    )

    deps.gateway.publish_movement(safe_command)
    deps.state.last_movement_command = safe_command

    stop_command: MovementCommand | None = None
    if deps.auto_stop_after_tool:
        if deps.execute_real_time:
            time.sleep(duration_s)

        stop_command = MovementCommand(source=safe_command.source)
        deps.gateway.publish_movement(stop_command)
        deps.state.last_movement_command = stop_command

    if next_pose is not None:
        deps.state.current_pose = next_pose

    cap_note = ""
    if safe_command != command:
        cap_note = " Geschwindigkeiten wurden auf die erlaubten Grenzwerte begrenzt."

    return MovementToolResult(
        ok = True,
        message = f"Bewegungsbefehl ausgeführt.{cap_note}".strip(),
        command = safe_command,
        stop_command = stop_command,
        pose = deps.state.current_pose,
        topic = deps.gateway.movement_topic,
        dry_run = deps.gateway.dry_run,
    )


def stop_robot(deps: MoveAgentDeps, reason: str = "requested") -> MovementToolResult:
    """Sendet einen Nullbefehl und aktualisiert den Bewegungsstatus."""

    command = MovementCommand(source=deps.source)
    deps.gateway.publish_movement(command)
    deps.state.last_movement_command = command
    return MovementToolResult(
        ok = True,
        message = f"Roboter gestoppt: {reason}",
        command = command,
        pose = deps.state.current_pose,
        topic = deps.gateway.movement_topic,
        dry_run = deps.gateway.dry_run,
    )


def _estimate_next_pose(
    pose: RobotPose | None,
    *,
    estimated_distance_m: float,
    estimated_turn_deg: float,
) -> RobotPose | None:
    if pose is None:
        return None

    next_pose = pose
    if estimated_distance_m:
        next_pose = next_pose.moved(estimated_distance_m)
    if estimated_turn_deg:
        next_pose = next_pose.turned(estimated_turn_deg)
    return next_pose


def _stale_reason(source_timestamp: float | None, now: float) -> str | None:
    if source_timestamp is None:
        return None

    if now - source_timestamp > COMMAND_MAX_AGE_SECONDS:
        return "Explore-Zeitstempel ist älter als 1 Sekunde."

    return None


def _rejected(message: str, deps: MoveAgentDeps) -> MovementToolResult:
    logger.warning("Movement rejected: %s", message)
    return MovementToolResult(
        ok = False,
        message = message,
        pose = deps.state.current_pose,
        topic = deps.gateway.movement_topic,
        dry_run = deps.gateway.dry_run,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
