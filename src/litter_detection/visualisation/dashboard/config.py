"""Configuration for the Gradio Robodog dashboard.

Values are intentionally environment-driven so the visualization can be reused
from local development, Docker, and robot runtime processes without hardcoded
paths or network assumptions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DashboardConfig:
    """Runtime configuration for the Gradio dashboard."""

    host: str = os.getenv("LITTER_DASHBOARD_HOST", "127.0.0.1")
    port: int = int(os.getenv("LITTER_DASHBOARD_PORT", "7860"))
    share: bool = _env_bool("LITTER_DASHBOARD_SHARE", False)
    provider: str = os.getenv("LITTER_DASHBOARD_PROVIDER", "zenoh")
    refresh_interval_s: float = float(os.getenv("LITTER_DASHBOARD_REFRESH_S", "1.0"))
    max_log_entries: int = int(os.getenv("LITTER_DASHBOARD_MAX_LOGS", "250"))
    max_detections: int = int(os.getenv("LITTER_DASHBOARD_MAX_DETECTIONS", "80"))
    max_obstacles: int = int(os.getenv("LITTER_DASHBOARD_MAX_OBSTACLES", "400"))
    obstacle_max_age_s: float = float(os.getenv("LITTER_DASHBOARD_OBSTACLE_MAX_AGE_S", "60"))
    lidar_obstacle_cell_size_m: float = float(os.getenv("LITTER_DASHBOARD_LIDAR_CELL_SIZE_M", "0.25"))
    camera_fov_h_deg: float = float(os.getenv("LITTER_DASHBOARD_CAMERA_FOV_H_DEG", "90"))
    initial_mode: str = os.getenv("LITTER_DASHBOARD_INITIAL_MODE", "IDLE")
    control_buttons: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            item.strip()
            for item in os.getenv(
                "LITTER_DASHBOARD_CONTROLS",
                "Stop,Start,Zurück zum Start,Manuelle Übernahme,Karte speichern",
            ).split(",")
            if item.strip()
        )
    )
