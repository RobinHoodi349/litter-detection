"""Koordinaten-Tool für den Move Agent."""

from __future__ import annotations

import math
import time
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.agent.tools.motion_types import (
    AUTONOMOUS_MAX_X_MPS,
    AUTONOMOUS_MAX_Z_DEG_PER_S,
    COMMAND_MAX_AGE_SECONDS,
    MoveAgentDeps,
    MovementCommand,
    MovementSource,
    MovementToolResult,
    RobotPose,
    execute_timed_movement,
    stop_robot,
)

CoordinateMeters = Annotated[float, Field(description = "Zielkoordinate in Metern.")]
SpeedMetersPerSecond = Annotated[float, Field(gt = 0.0, le = 1.0)]
AngularSpeedDegreesPerSecond = Annotated[float, Field(gt = 0.0, le = 90.0)]
ToleranceMeters = Annotated[float, Field(gt = 0.0, le = 1.0)]


class CoordinateMoveResult(BaseModel):
    """Antwort für eine Bewegung zu einem Zielpunkt."""

    ok: bool
    message: str
    target_pose: RobotPose
    start_pose: RobotPose | None = None
    final_pose: RobotPose | None = None
    turn_result: MovementToolResult | None = None
    walk_results: list[MovementToolResult] = Field(default_factory = list)
    dry_run: bool = True


def move_to_coordinate(
    ctx: RunContext[MoveAgentDeps],
    target_x_m: CoordinateMeters,
    target_y_m: CoordinateMeters,
    speed_mps: SpeedMetersPerSecond = 0.2,
    turn_speed_deg_s: AngularSpeedDegreesPerSecond = 20.0,
    position_tolerance_m: ToleranceMeters = 0.05,
    explore_timestamp: float | None = None,
) -> CoordinateMoveResult:
    """Läuft zu einer bereits geplanten x/y-Zielkoordinate.

    Dieses Tool plant keinen Pfad. Es setzt nur den nächsten Zielpunkt um,
    den der Explore Agent geliefert hat.
    """

    deps = ctx.deps
    start_pose = deps.state.current_pose
    target_pose = RobotPose(
        x_m=target_x_m,
        y_m=target_y_m,
        heading_deg=start_pose.heading_deg if start_pose is not None else 0.0,
    )

    if start_pose is None:
        return _coordinate_rejected(
            "Keine aktuelle Roboterpose vorhanden.",
            target_pose = target_pose,
            deps = deps,
        )

    if explore_timestamp is not None and time.time() - explore_timestamp > COMMAND_MAX_AGE_SECONDS:
        return _coordinate_rejected(
            "Explore-Zeitstempel ist älter als 1 Sekunde.",
            target_pose = target_pose,
            deps = deps,
            start_pose = start_pose,
        )

    dx = target_x_m - start_pose.x_m
    dy = target_y_m - start_pose.y_m
    distance_m = math.hypot(dx, dy)

    if distance_m <= position_tolerance_m:
        stop_result = stop_robot(deps, reason = "Zielkoordinate bereits erreicht")
        return CoordinateMoveResult(
            ok=True,
            message = "Zielkoordinate liegt innerhalb der Toleranz. Roboter bleibt stehen.",
            target_pose = target_pose,
            start_pose = start_pose,
            final_pose = deps.state.current_pose,
            walk_results = [stop_result],
            dry_run = deps.gateway.dry_run,
        )

    target_heading_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    target_pose = RobotPose(
        x_m = target_x_m,
        y_m = target_y_m,
        heading_deg = target_heading_deg,
    )
    turn_angle_deg = _shortest_signed_angle(start_pose.heading_deg, target_heading_deg)

    turn_result: MovementToolResult | None = None
    if abs(turn_angle_deg) > 1.0:
        turn_speed = _signed_speed(
            turn_angle_deg,
            _effective_turn_speed(deps.source, abs(turn_speed_deg_s)),
        )
        turn_duration_s = abs(turn_angle_deg) / abs(turn_speed)
        turn_command = MovementCommand(
            x = 0.0,
            y = 0.0,
            z_deg = turn_speed,
            source = deps.source,
        )

        turn_result = execute_timed_movement(
            deps,
            command = turn_command,
            duration_s = turn_duration_s,
            estimated_turn_deg = turn_angle_deg,
            source_timestamp = explore_timestamp,
        )
        if not turn_result.ok:
            return CoordinateMoveResult(
                ok = False,
                message = f"Drehung zur Zielkoordinate fehlgeschlagen: {turn_result.message}",
                target_pose = target_pose,
                start_pose = start_pose,
                final_pose = deps.state.current_pose,
                turn_result = turn_result,
                dry_run = deps.gateway.dry_run,
            )

    walk_results: list[MovementToolResult] = []
    effective_speed = _effective_walk_speed(deps.source, abs(speed_mps))
    remaining_m = distance_m

    while remaining_m > position_tolerance_m:
        step_distance_m = min(remaining_m, effective_speed * deps.max_tool_duration_s)
        duration_s = step_distance_m / effective_speed
        walk_command = MovementCommand(
            x = abs(speed_mps),
            y = 0.0,
            z_deg = 0.0,
            source = deps.source,
        )

        # Längere Strecken werden in kurze, sichere Laufbefehle aufgeteilt.
        walk_result = execute_timed_movement(
            deps,
            command = walk_command,
            duration_s = duration_s,
            estimated_distance_m = step_distance_m,
            source_timestamp = explore_timestamp,
        )
        walk_results.append(walk_result)

        if not walk_result.ok:
            return CoordinateMoveResult(
                ok = False,
                message = f"Weg zur Zielkoordinate fehlgeschlagen: {walk_result.message}",
                target_pose = target_pose,
                start_pose = start_pose,
                final_pose = deps.state.current_pose,
                turn_result = turn_result,
                walk_results = walk_results,
                dry_run = deps.gateway.dry_run,
            )

        remaining_m -= step_distance_m

    if deps.state.current_pose is not None:
        deps.state.current_pose = RobotPose(
            x_m = target_x_m,
            y_m = target_y_m,
            heading_deg = deps.state.current_pose.heading_deg,
        )

    return CoordinateMoveResult(
        ok = True,
        message = "Zielkoordinate erreicht.",
        target_pose = target_pose,
        start_pose = start_pose,
        final_pose = deps.state.current_pose,
        turn_result = turn_result,
        walk_results = walk_results,
        dry_run = deps.gateway.dry_run,
    )


def _coordinate_rejected(
    message: str,
    *,
    target_pose: RobotPose,
    deps: MoveAgentDeps,
    start_pose: RobotPose | None = None,
) -> CoordinateMoveResult:
    return CoordinateMoveResult(
        ok = False,
        message = message,
        target_pose = target_pose,
        start_pose = start_pose,
        final_pose = deps.state.current_pose,
        dry_run = deps.gateway.dry_run,
    )


def _shortest_signed_angle(current_deg: float, target_deg: float) -> float:
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def _signed_speed(angle_deg: float, speed_abs: float) -> float:
    if angle_deg >= 0.0:
        return speed_abs
    return -speed_abs


def _effective_walk_speed(source: MovementSource, speed_abs: float) -> float:
    if source == MovementSource.controller:
        return speed_abs
    return min(speed_abs, AUTONOMOUS_MAX_X_MPS)


def _effective_turn_speed(source: MovementSource, speed_abs: float) -> float:
    if source == MovementSource.controller:
        return speed_abs
    return min(speed_abs, AUTONOMOUS_MAX_Z_DEG_PER_S)
