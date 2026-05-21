"""Data-provider interfaces and mock queues for the Gradio dashboard."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Queue
from typing import Protocol

import numpy as np

from litter_detection.visualisation.dashboard.config import DashboardConfig


@dataclass(frozen=True)
class CameraFrame:
    """Camera frame with metadata for the dashboard."""

    image: np.ndarray
    timestamp: str
    fps: float


@dataclass(frozen=True)
class MapFrame:
    """Map visualization with robot pose metadata."""

    image: np.ndarray
    x_m: float
    y_m: float
    yaw_deg: float


@dataclass(frozen=True)
class TrashDetection:
    """Single detected litter object shown in the trash gallery."""

    image: np.ndarray
    label: str
    confidence: float
    timestamp: str
    position: str


@dataclass(frozen=True)
class LogEntry:
    """Structured log entry for display and filtering."""

    timestamp: str
    level: str
    source: str
    message: str


@dataclass(frozen=True)
class RobotStatus:
    """Operational robot status shown in the control panel."""

    mode: str
    battery_percent: int
    connected: bool


class DashboardDataProvider(Protocol):
    """Interface between robot logic and dashboard panels."""

    def get_camera_frame(self) -> CameraFrame:
        """Return the latest camera frame."""

    def get_map_frame(self) -> MapFrame:
        """Return the latest map frame."""

    def get_trash_detections(self) -> list[TrashDetection]:
        """Return recent trash detections."""

    def get_logs(self, level_filter: str = "ALL") -> list[LogEntry]:
        """Return recent log entries, optionally filtered by level."""

    def get_status(self) -> RobotStatus:
        """Return the current robot status."""

    def handle_control(self, action: str) -> str:
        """Execute or enqueue a robot control action."""


class QueueDashboardDataProvider:
    """Queue-backed provider with mock fallback data.

    Robot-side code can push real objects into the public queues. If no real
    data is available yet, deterministic mock data keeps the UI usable.
    """

    def __init__(self, config: DashboardConfig | None = None) -> None:
        self.config = config or DashboardConfig()
        self.camera_queue: Queue[CameraFrame] = Queue(maxsize=5)
        self.map_queue: Queue[MapFrame] = Queue(maxsize=5)
        self.detection_queue: Queue[TrashDetection] = Queue(maxsize=100)
        self.log_queue: Queue[LogEntry] = Queue(maxsize=500)
        self.command_queue: Queue[str] = Queue(maxsize=50)
        self._detections: deque[TrashDetection] = deque(maxlen=self.config.max_detections)
        self._logs: deque[LogEntry] = deque(maxlen=self.config.max_log_entries)
        self._status = RobotStatus(self.config.initial_mode, 87, True)
        self._last_camera: CameraFrame | None = None
        self._last_map: MapFrame | None = None
        self._has_real_camera = False
        self._has_real_map = False
        self._started_at = time.monotonic()

    def get_camera_frame(self) -> CameraFrame:
        self._drain_realtime_queues()
        if not self._has_real_camera:
            self._last_camera = self._mock_camera_frame()
        return self._last_camera

    def get_map_frame(self) -> MapFrame:
        self._drain_realtime_queues()
        if not self._has_real_map:
            self._last_map = self._mock_map_frame()
        return self._last_map

    def get_trash_detections(self) -> list[TrashDetection]:
        self._drain_realtime_queues()
        if not self._detections:
            self._detections.append(self._mock_detection("plastic_bottle", 0.91, 0))
            self._detections.append(self._mock_detection("paper_cup", 0.84, 1))
        return list(reversed(self._detections))

    def get_logs(self, level_filter: str = "ALL") -> list[LogEntry]:
        self._drain_realtime_queues()
        if not self._logs:
            self._logs.extend(
                [
                    self._mock_log("INFO", "coordinator", "Dashboard initialized"),
                    self._mock_log("INFO", "navigator", "Awaiting mission start"),
                    self._mock_log("WARN", "mapping", "Mock map data active"),
                ]
            )
        if level_filter == "ALL":
            return list(reversed(self._logs))
        return [entry for entry in reversed(self._logs) if entry.level == level_filter]

    def get_status(self) -> RobotStatus:
        return self._status

    def handle_control(self, action: str) -> str:
        # TODO: connect to real robot command publisher.
        self.command_queue.put(action)
        mode_by_action = {
            "Stop": "STOPPED",
            "Start": "AUTONOMOUS",
            "Zurück zum Start": "RETURN_HOME",
            "Zurueck zum Start": "RETURN_HOME",
            "Manuelle Übernahme": "MANUAL",
            "Manuelle Uebernahme": "MANUAL",
            "Karte speichern": self._status.mode,
        }
        self._status = RobotStatus(mode_by_action.get(action, self._status.mode), self._status.battery_percent, True)
        self._logs.append(self._mock_log("INFO", "control", f"Command queued: {action}"))
        return f"{self._now()} | command queued: {action}"

    def _drain_realtime_queues(self) -> None:
        # TODO: connect to real data source.
        self._last_camera, camera_updated = self._drain_latest(self.camera_queue, self._last_camera)
        self._last_map, map_updated = self._drain_latest(self.map_queue, self._last_map)
        self._has_real_camera = self._has_real_camera or camera_updated
        self._has_real_map = self._has_real_map or map_updated
        self._drain_all(self.detection_queue, self._detections)
        self._drain_all(self.log_queue, self._logs)

    @staticmethod
    def _drain_latest(queue: Queue, fallback):
        latest = fallback
        updated = False
        while True:
            try:
                latest = queue.get_nowait()
                updated = True
            except Empty:
                return latest, updated

    @staticmethod
    def _drain_all(queue: Queue, target: deque) -> None:
        while True:
            try:
                target.append(queue.get_nowait())
            except Empty:
                return

    def _mock_camera_frame(self) -> CameraFrame:
        elapsed = time.monotonic() - self._started_at
        h, w = 360, 640
        y = np.linspace(0, 1, h)[:, None]
        x = np.linspace(0, 1, w)[None, :]
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[..., 0] = (40 + 80 * x).astype(np.uint8)
        image[..., 1] = (60 + 90 * y).astype(np.uint8)
        image[..., 2] = 105
        cx = int((0.5 + 0.25 * math.sin(elapsed)) * w)
        cy = int((0.5 + 0.20 * math.cos(elapsed / 1.7)) * h)
        image[max(cy - 20, 0) : cy + 20, max(cx - 35, 0) : cx + 35] = [155, 120, 230]
        return CameraFrame(image=image, timestamp=self._now(), fps=24.0 + 2.0 * math.sin(elapsed))

    def _mock_map_frame(self) -> MapFrame:
        elapsed = time.monotonic() - self._started_at
        size = 520
        image = np.full((size, size, 3), 245, dtype=np.uint8)
        image[::40, :, :] = 226
        image[:, ::40, :] = 226
        center = size // 2
        path_points: list[tuple[int, int]] = []
        for idx in range(90):
            angle = idx / 10.0
            radius = 6 + idx * 2
            path_points.append((center + int(math.cos(angle) * radius), center + int(math.sin(angle) * radius)))
        for x_px, y_px in path_points:
            image[max(y_px - 2, 0) : y_px + 3, max(x_px - 2, 0) : x_px + 3] = [247, 184, 25]
        robot_x = center + int(math.cos(elapsed / 2.0) * 130)
        robot_y = center + int(math.sin(elapsed / 2.0) * 95)
        image[max(robot_y - 10, 0) : robot_y + 11, max(robot_x - 10, 0) : robot_x + 11] = [120, 76, 35]
        return MapFrame(image=image, x_m=(robot_x - center) / 40, y_m=(center - robot_y) / 40, yaw_deg=(elapsed * 20) % 360)

    def _mock_detection(self, label: str, confidence: float, index: int) -> TrashDetection:
        image = np.full((160, 220, 3), [26, 45, 90], dtype=np.uint8)
        image[35 + index * 10 : 105 + index * 10, 70:150] = [236, 205, 70]
        return TrashDetection(image=image, label=label, confidence=confidence, timestamp=self._now(), position=f"x={1.2 + index:.1f}m, y={0.8 + index * 0.4:.1f}m")

    def _mock_log(self, level: str, source: str, message: str) -> LogEntry:
        return LogEntry(timestamp=self._now(), level=level, source=source, message=message)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
