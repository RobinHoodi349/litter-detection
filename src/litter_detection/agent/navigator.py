"""Einfacher koordinatenbasierter Navigator für den Robodog.

Steuerungslogik:
  1. Odometrie per Zenoh empfangen (Quaternion → Yaw)
  2. Winkel- und Distanzfehler berechnen
  3. Erst drehen bis zur Ausrichtung, dann geradeaus laufen
  4. Stoppen bei Zielankunft

Konfiguration via litter_detection.config.Settings:
  ZENOH_ROUTER           Zenoh-Router-Adresse
  topic_odometry         Odometrie-Topic (rt/utlidar/robot_pose)
  topic_movement_command Bewegungsbefehl-Topic
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from litter_detection.config import Settings
from litter_detection.agent.models import MovementCommand, MovementSource, OdometryState
from litter_detection.agent.tools.motion_types import (
    AUTONOMOUS_MAX_X_MPS,
    AUTONOMOUS_MAX_Z_DEG_PER_S,
    RobotMotionGateway,
)

_settings = Settings()
logger = logging.getLogger("navigator")

POSITION_TOLERANCE_M = 0.15
HEADING_TOLERANCE_DEG = 5.0
WALK_SPEED_MPS = 0.2
TURN_SPEED_DEG_S = 20.0
PUBLISH_INTERVAL_S = 0.5


@dataclass
class _Pose:
    x: float
    y: float
    yaw_deg: float  # [0, 360)


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extrahiert den Yaw-Winkel (Z-Rotation) aus einem Quaternion, in Grad [0, 360)."""
    yaw_rad = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return math.degrees(yaw_rad) % 360.0


def _shortest_angle(current_deg: float, target_deg: float) -> float:
    """Kürzester vorzeichenbehafteter Winkelunterschied in (-180, 180]."""
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


class PointNavigator:
    """Navigiert den Robodog sequenziell zu einer x/y-Zielkoordinate.

    Ablauf pro Steuerloop-Iteration:
    - Ist der Roboter nahe genug am Ziel? → Stoppen.
    - Zeigt der Roboter in die richtige Richtung? → Nein: Drehen (z_deg ≠ 0, x = 0).
    - Ausgerichtet? → Geradeaus laufen (x ≠ 0, z_deg = 0).

    Drehen und Laufen werden NIE gleichzeitig gesendet.
    """

    def __init__(
        self,
        target_x: float,
        target_y: float,
        gateway: RobotMotionGateway,
        odom_topic: str = _settings.topic_odometry,
        position_tolerance_m: float = POSITION_TOLERANCE_M,
        heading_tolerance_deg: float = HEADING_TOLERANCE_DEG,
        walk_speed_mps: float = WALK_SPEED_MPS,
        turn_speed_deg_s: float = TURN_SPEED_DEG_S,
        publish_interval_s: float = PUBLISH_INTERVAL_S,
    ) -> None:
        self.target_x = target_x
        self.target_y = target_y
        self.gateway = gateway
        self.odom_topic = odom_topic
        self.position_tolerance_m = position_tolerance_m
        self.heading_tolerance_deg = heading_tolerance_deg
        self.walk_speed_mps = min(walk_speed_mps, AUTONOMOUS_MAX_X_MPS)
        self.turn_speed_deg_s = min(turn_speed_deg_s, AUTONOMOUS_MAX_Z_DEG_PER_S)
        self.publish_interval_s = publish_interval_s
        self._pose: _Pose | None = None

    def run(self, stop_event: threading.Event | None = None) -> bool:
        """Blockiert bis zur Zielankunft oder Abbruch.

        Args:
            stop_event: Wenn gesetzt, bricht die Navigation sofort ab (z.B. durch BLOCK).

        Returns:
            True wenn Ziel erreicht, False bei Abbruch.
        """
        import zenoh

        conf = zenoh.Config()
        if self.gateway.router:
            conf.insert_json5("connect/endpoints", json.dumps([self.gateway.router]))

        logger.info("Navigator gestartet → Ziel (%.2f, %.2f)", self.target_x, self.target_y)

        with zenoh.open(conf) as session:
            sub = session.declare_subscriber(self.odom_topic, self._on_odometry)
            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        logger.info("Navigation unterbrochen.")
                        self._stop()
                        return False
                    cmd = self._compute_command()
                    if cmd is None:
                        self._stop()
                        logger.info("Ziel (%.2f, %.2f) erreicht.", self.target_x, self.target_y)
                        return True
                    self.gateway.publish_movement(cmd)
                    time.sleep(self.publish_interval_s)
            except KeyboardInterrupt:
                logger.info("Navigation abgebrochen.")
                self._stop()
                return False
            finally:
                sub.undeclare()

    def _on_odometry(self, sample: Any) -> None:
        """Zenoh-Subscriber-Callback: aktualisiert die interne Pose."""
        try:
            raw = json.loads(bytes(sample.payload))
            state = OdometryState.from_raw(raw)
            if state is None:
                return
            qx, qy, qz, qw = state.quaternion
            self._pose = _Pose(
                x=state.x,
                y=state.y,
                yaw_deg=_yaw_from_quaternion(qx, qy, qz, qw),
            )
        except Exception:
            logger.debug("Odometrie-Verarbeitung fehlgeschlagen", exc_info=True)

    def _compute_command(self) -> MovementCommand | None:
        """Berechnet den nächsten Bewegungsbefehl; None bedeutet Ziel erreicht."""
        pose = self._pose
        if pose is None:
            logger.debug("Warte auf erste Odometrienachricht …")
            return MovementCommand(source=MovementSource.autonomous)

        dx = self.target_x - pose.x
        dy = self.target_y - pose.y
        distance = math.hypot(dx, dy)

        if distance <= self.position_tolerance_m:
            return None

        target_heading_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        heading_error = _shortest_angle(pose.yaw_deg, target_heading_deg)

        if abs(heading_error) > self.heading_tolerance_deg:
            z = math.copysign(self.turn_speed_deg_s, heading_error)
            logger.debug(
                "Drehen: Fehler=%.1f° → z_deg=%.1f", heading_error, z
            )
            return MovementCommand(x=0.0, y=0.0, z_deg=z, source=MovementSource.autonomous)

        logger.debug("Laufen: Distanz=%.2f m", distance)
        return MovementCommand(
            x=self.walk_speed_mps, y=0.0, z_deg=0.0, source=MovementSource.autonomous
        )

    def _stop(self) -> None:
        self.gateway.publish_movement(MovementCommand(source=MovementSource.autonomous))


def navigate_to(
    target_x: float,
    target_y: float,
    *,
    walk_speed_mps: float = WALK_SPEED_MPS,
    turn_speed_deg_s: float = TURN_SPEED_DEG_S,
    position_tolerance_m: float = POSITION_TOLERANCE_M,
    heading_tolerance_deg: float = HEADING_TOLERANCE_DEG,
    publish_interval_s: float = PUBLISH_INTERVAL_S,
    odom_topic: str = _settings.topic_odometry,
) -> bool:
    """Convenience-Funktion: navigiert zum Ziel mit Einstellungen aus Settings."""
    gateway = RobotMotionGateway(
        router=_settings.ZENOH_ROUTER,
        movement_topic=_settings.topic_movement_command,
        dry_run=False,
    )
    nav = PointNavigator(
        target_x=target_x,
        target_y=target_y,
        gateway=gateway,
        odom_topic=odom_topic,
        position_tolerance_m=position_tolerance_m,
        heading_tolerance_deg=heading_tolerance_deg,
        walk_speed_mps=walk_speed_mps,
        turn_speed_deg_s=turn_speed_deg_s,
        publish_interval_s=publish_interval_s,
    )
    return nav.run()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Navigiert den Robodog zu einer Zielkoordinate.")
    parser.add_argument("x", type=float, help="Ziel X in Metern")
    parser.add_argument("y", type=float, help="Ziel Y in Metern")
    parser.add_argument("--walk-speed", type=float, default=WALK_SPEED_MPS)
    parser.add_argument("--turn-speed", type=float, default=TURN_SPEED_DEG_S)
    parser.add_argument("--tolerance", type=float, default=POSITION_TOLERANCE_M)
    parser.add_argument("--heading-tolerance", type=float, default=HEADING_TOLERANCE_DEG)
    parser.add_argument("--interval", type=float, default=PUBLISH_INTERVAL_S)
    parser.add_argument("--odom-topic", type=str, default=_settings.topic_odometry)
    args = parser.parse_args()

    success = navigate_to(
        args.x,
        args.y,
        walk_speed_mps=args.walk_speed,
        turn_speed_deg_s=args.turn_speed,
        position_tolerance_m=args.tolerance,
        heading_tolerance_deg=args.heading_tolerance,
        publish_interval_s=args.interval,
        odom_topic=args.odom_topic,
    )
    raise SystemExit(0 if success else 1)
