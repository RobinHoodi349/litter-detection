"""Walk Tool für den Move Agent."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai import RunContext

from src.agent.tools.motion_types import (
    AUTONOMOUS_MAX_X_MPS,
    MoveAgentDeps,
    MovementCommand,
    MovementSource,
    MovementToolResult,
    execute_timed_movement,
    stop_robot,
)

WalkDirection = Literal["forward", "backward"]
DurationSeconds = Annotated[float, Field(gt = 0.0, le = 30.0)]
WalkSpeedMetersPerSecond = Annotated[float, Field(gt = 0.0, le = 1.0)]


def walk(
    ctx: RunContext[MoveAgentDeps],
    direction: WalkDirection,
    duration_s: DurationSeconds,
    speed_mps: WalkSpeedMetersPerSecond = 0.2,
    explore_timestamp: float | None = None,
) -> MovementToolResult:
    """Lässt den Roboterhund geradeaus vorwärts oder rückwärts laufen."""

    signed_speed = abs(speed_mps) if direction == "forward" else -abs(speed_mps)
    effective_speed = _effective_walk_speed(ctx.deps.source, signed_speed)
    command = MovementCommand(
        x = signed_speed,
        y = 0.0,
        z_deg = 0.0,
        source = ctx.deps.source,
    )

    # Die Posenfortschreibung ist eine Schaetzung; echte Lokalisierung überschreibt sie später.
    return execute_timed_movement(
        ctx.deps,
        command = command,
        duration_s = duration_s,
        estimated_distance_m = effective_speed * duration_s,
        source_timestamp = explore_timestamp,
    )


def stop_movement(
    ctx: RunContext[MoveAgentDeps],
    reason: str = "requested",
) -> MovementToolResult:
    """Stoppt den Roboterhund mit einem Null-Bewegungsbefehl."""

    return stop_robot(ctx.deps, reason=reason)


def _effective_walk_speed(source: MovementSource, signed_speed: float) -> float:
    if source == MovementSource.controller:
        return signed_speed

    if signed_speed >= 0:
        return min(signed_speed, AUTONOMOUS_MAX_X_MPS)

    return max(signed_speed, -AUTONOMOUS_MAX_X_MPS)
