"""Turn Tool fuer den Move Agent."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai import RunContext

from litter_detection.agent.tools.motion_types import (
    AUTONOMOUS_MAX_Z_DEG_PER_S,
    MoveAgentDeps,
    MovementCommand,
    MovementSource,
    MovementToolResult,
    execute_timed_movement,
)

TurnDirection = Literal["left", "right"]
TurnAngleDegrees = Annotated[float, Field(gt = 0.0, le = 360.0)]
TurnSpeedDegreesPerSecond = Annotated[float, Field(gt = 0.0, le = 90.0)]


def turn(
    ctx: RunContext[MoveAgentDeps],
    direction: TurnDirection,
    angle_deg: TurnAngleDegrees,
    angular_speed_deg_s: TurnSpeedDegreesPerSecond = 20.0,
    explore_timestamp: float | None = None,
) -> MovementToolResult:
    """Dreht den Roboterhund auf der Stelle nach links oder rechts."""

    signed_angle = abs(angle_deg) if direction == "left" else -abs(angle_deg)
    signed_speed = (
        abs(angular_speed_deg_s)
        if direction == "left"
        else -abs(angular_speed_deg_s)
    )
    effective_speed = _effective_turn_speed(ctx.deps.source, signed_speed)
    duration_s = abs(angle_deg) / abs(effective_speed)

    command = MovementCommand(
        x = 0.0,
        y = 0.0,
        z_deg = signed_speed,
        source = ctx.deps.source,
    )

    return execute_timed_movement(
        ctx.deps,
        command = command,
        duration_s = duration_s,
        estimated_turn_deg = signed_angle,
        source_timestamp = explore_timestamp,
    )


def _effective_turn_speed(source: MovementSource, signed_speed: float) -> float:
    if source == MovementSource.controller:
        return signed_speed

    if signed_speed >= 0:
        return min(signed_speed, AUTONOMOUS_MAX_Z_DEG_PER_S)

    return max(signed_speed, -AUTONOMOUS_MAX_Z_DEG_PER_S)
