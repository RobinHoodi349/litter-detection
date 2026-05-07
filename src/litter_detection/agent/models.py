from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Detection models ---

class DetectionResult(BaseModel):
    timestamp: float
    litter_detected: bool
    pixel_coverage: float
    mask_shape: tuple[int, int]
    frame_height: int
    frame_width: int


class VerifiedDetection(BaseModel):
    litter_confirmed: bool
    confidence: Literal["high", "medium", "low"]
    description: str


@dataclass
class VerifierDeps:
    detection: DetectionResult


# --- Movement message models (Zenoh interface) ---

class MovementSource(StrEnum):
    controller = "controller"
    autonomous = "autonomous"
    planner = "planner"


class MovementCommand(BaseModel):
    """Bewegungsbefehl für den Robodog (Zenoh-Topic: topic_movement_command)."""

    x: float = Field(default=0.0, description="Vorwärts-/Rückwärtsgeschwindigkeit in m/s.")
    y: float = Field(default=0.0, description="Seitwärtsgeschwindigkeit in m/s.")
    z_deg: float = Field(default=0.0, description="Drehgeschwindigkeit in Grad/s.")
    source: MovementSource = MovementSource.autonomous
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Odometry message model (Zenoh-Topic: topic_odometry) ---

class OdometryState(BaseModel):
    """Roboterpose aus der Odometriequelle (rt/utlidar/robot_pose)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    quaternion: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0],
        description="Orientation as [qx, qy, qz, qw]",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_raw(cls, message: dict[str, Any]) -> OdometryState | None:
        """Parst eine rohe Zenoh-Odometrienachricht.

        Unterstützt zwei Formate:
          Flat (rt/utlidar/robot_pose):
            {"x":…, "y":…, "z":…, "quaternion":[qx,qy,qz,qw], "timestamp":"…"}
          ROS-nested:
            {"data":{"header":{"stamp":…},"pose":{"position":…,"orientation":…}}}
        """
        try:
            if "quaternion" in message:
                ts_raw = message.get("timestamp")
                ts = (
                    datetime.fromisoformat(ts_raw)
                    if isinstance(ts_raw, str)
                    else datetime.now(timezone.utc)
                )
                return cls(
                    x=float(message["x"]),
                    y=float(message["y"]),
                    z=float(message["z"]),
                    quaternion=[float(v) for v in message["quaternion"]],
                    timestamp=ts,
                )
            # ROS nested fallback
            data = message["data"]
            stamp = data["header"]["stamp"]
            pos = data["pose"]["position"]
            ori = data["pose"]["orientation"]
            return cls(
                x=pos["x"],
                y=pos["y"],
                z=pos["z"],
                quaternion=[ori["x"], ori["y"], ori["z"], ori["w"]],
                timestamp=datetime.fromtimestamp(
                    stamp["sec"] + stamp["nanosec"] / 1e9, tz=timezone.utc
                ),
            )
        except Exception:
            return None
