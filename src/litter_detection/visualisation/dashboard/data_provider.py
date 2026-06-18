"""Data-provider interfaces and mock queues for the Gradio dashboard."""

from __future__ import annotations

import math
import json
import logging
import os
import socket
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import RLock, Thread
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

from litter_detection.visualisation.dashboard.config import DashboardConfig


logger = logging.getLogger(__name__)

OBSTACLE_Z_MIN_M = 0.05
OBSTACLE_Z_MAX_M = 0.80


def pick_detection_image(overlay, camera, fallback):
    """Choose the image stored for a detection.

    Prefers the model's annotated overlay (the recognised mask) so the
    validation view shows *what the model detected*, then the raw camera
    frame, then a placeholder. Inputs are typically ``np.ndarray`` or ``None``.
    """

    if overlay is not None:
        return overlay
    if camera is not None:
        return camera
    return fallback


@dataclass(frozen=True)
class ZenohDashboardSettings:
    """Lightweight Zenoh settings for the dashboard runtime."""

    router: str = os.getenv("ZENOH_ROUTER", "tcp/localhost:7447")
    topic_frame: str = os.getenv("LITTER_TOPIC_FRAME", "robodog/sensors/go2_camera")
    topic_visualization: str = os.getenv("LITTER_TOPIC_VISUALIZATION", "litter/visualization")
    topic_alert: str = os.getenv("LITTER_TOPIC_ALERT", "litter/alert")
    topic_odometry: str = os.getenv("LITTER_TOPIC_ODOMETRY", "robodog/system_state/odometry")
    topic_lidar: str = os.getenv("LITTER_TOPIC_LIDAR", "robodog/sensors/go2_lidar")
    topic_obstacle: str = os.getenv("LITTER_TOPIC_OBSTACLE", "litter/obstacles")
    topic_occupancy_grid: str = os.getenv("LITTER_TOPIC_OCCUPANCY_GRID", "litter/occupancy_grid")
    topic_robodog_command: str = os.getenv("LITTER_TOPIC_ROBODOG_COMMAND", "litter/robodog/command")
    topic_movement_command: str = os.getenv("LITTER_TOPIC_MOVEMENT_COMMAND", "robodog/command/motion/move")


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
    camera_obstacles: int = 0
    lidar_obstacles: int = 0
    explored_cells: int = 0
    explored_area_m2: float = 0.0


@dataclass(frozen=True)
class ObstacleDetection:
    """Obstacle projected into map coordinates."""

    x_m: float
    y_m: float
    source: str
    label: str = "Hindernis"
    confidence: float = 0.0
    radius_m: float = 0.18
    detected_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class LitterMapMarker:
    """Map position where litter was detected."""

    x_m: float
    y_m: float
    detected_at: float = field(default_factory=time.monotonic)


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
            

    def handle_manual_control(self, direction: str, speed: float) -> str:
        """Execute or enqueue a manual controller movement."""

    def save_current_map(self) -> str:
        """Save the current map image and return the created file path."""


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
        self.obstacle_queue: Queue[ObstacleDetection] = Queue(maxsize=200)
        self.log_queue: Queue[LogEntry] = Queue(maxsize=500)
        self.command_queue: Queue[str] = Queue(maxsize=50)
        self._detections: deque[TrashDetection] = deque(maxlen=self.config.max_detections)
        self._camera_obstacles: deque[ObstacleDetection] = deque(maxlen=self.config.max_obstacles)
        self._lidar_obstacles: dict[tuple[int, int], ObstacleDetection] = {}
        self._litter_markers: deque[LitterMapMarker] = deque(maxlen=self.config.max_detections)
        self._logs: deque[LogEntry] = deque(maxlen=self.config.max_log_entries)
        self._status = RobotStatus(self.config.initial_mode, 87, True)
        self._last_camera: CameraFrame | None = None
        self._last_map: MapFrame | None = None
        # Latest annotated model overlay (recognised mask), used as the image
        # shown for a detection so the validation view reflects the model output.
        self._last_overlay: np.ndarray | None = None
        self._has_real_camera = False
        self._has_real_map = False
        self._started_at = time.monotonic()
        self._home_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._mock_pose: tuple[float, float, float] = self._home_pose
        self._mock_path: deque[tuple[float, float]] = deque(maxlen=300)
        self._mock_path.append((self._home_pose[0], self._home_pose[1]))
        self._mock_phase = 0.0
        self._manual_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_motion_update = time.monotonic()
        self._saved_maps_dir = Path(tempfile.gettempdir()) / "litter_detection_maps"
        if self.config.provider.strip().lower() == "mock":
            self._seed_mock_map_features()

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
        normalized = self._normalize_action(action)
        self._apply_control_motion_state(normalized)
        mode = self._mode_for_action(normalized, self._status.mode)
        self._status = RobotStatus(mode, self._status.battery_percent, True)
        self._logs.append(self._mock_log("INFO", "control", f"Command queued: {action}"))
        return f"{self._now()} | command queued: {action}"

    def handle_manual_control(self, direction: str, speed: float) -> str:
        """Update the mock controller state for manual movement."""

        speed = self._clamp_speed(speed)
        self._manual_velocity = self._manual_velocity_for(direction, speed)
        self._last_motion_update = time.monotonic()
        self._status = RobotStatus("MANUAL", self._status.battery_percent, True)
        message = f"manual control: {direction} ({speed:.2f})"
        self.command_queue.put(message)
        self._logs.append(self._mock_log("INFO", "control", message))
        return f"{self._now()} | {message}"

    def save_current_map(self) -> str:
        """Write the latest map image as PNG and return its path."""

        frame = self.get_map_frame()
        self._saved_maps_dir.mkdir(parents=True, exist_ok=True)
        path = self._saved_maps_dir / f"robodog_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        Image.fromarray(frame.image).save(path)
        self._logs.append(self._mock_log("INFO", "mapping", f"Map saved: {path}"))
        return str(path)

    def _drain_realtime_queues(self) -> None:
        # TODO: connect to real data source.
        self._last_camera, camera_updated = self._drain_latest(self.camera_queue, self._last_camera)
        self._last_map, map_updated = self._drain_latest(self.map_queue, self._last_map)
        self._has_real_camera = self._has_real_camera or camera_updated
        self._has_real_map = self._has_real_map or map_updated
        self._drain_all(self.detection_queue, self._detections)
        self._drain_obstacle_queue()
        self._drain_all(self.log_queue, self._logs)
        self._prune_obstacles()

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

    def _drain_obstacle_queue(self) -> None:
        while True:
            try:
                self._store_obstacle(self.obstacle_queue.get_nowait())
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
        now = time.monotonic()
        dt = min(max(now - self._last_motion_update, 0.0), 1.0)
        self._last_motion_update = now
        self._advance_mock_motion(dt)

        x_m, y_m, yaw_deg = self._mock_pose
        size = 520
        image = np.full((size, size, 3), 245, dtype=np.uint8)
        image[::40, :, :] = 226
        image[:, ::40, :] = 226
        center = size // 2
        scale = 40

        home_x, home_y, _ = self._home_pose
        home_px = center + int(home_x * scale)
        home_py = center - int(home_y * scale)
        image[max(home_py - 8, 0) : home_py + 9, max(home_px - 8, 0) : home_px + 9] = [34, 197, 94]

        for path_x, path_y in self._mock_path:
            px = center + int(path_x * scale)
            py = center - int(path_y * scale)
            if 0 <= px < size and 0 <= py < size:
                image[max(py - 2, 0) : py + 3, max(px - 2, 0) : px + 3] = [247, 184, 25]

        obstacles = self._current_obstacles()
        self._draw_obstacles(image, center, scale, obstacles)

        robot_px = center + int(x_m * scale)
        robot_py = center - int(y_m * scale)
        if 0 <= robot_px < size and 0 <= robot_py < size:
            image[max(robot_py - 10, 0) : robot_py + 11, max(robot_px - 10, 0) : robot_px + 11] = [247, 184, 25]
            heading = math.radians(yaw_deg)
            tip_x = robot_px + int(math.cos(heading) * 22)
            tip_y = robot_py - int(math.sin(heading) * 22)
            self._draw_line(image, robot_px, robot_py, tip_x, tip_y, [146, 95, 8])
        self._draw_litter_markers(image, center, scale, list(self._litter_markers))

        camera_count, lidar_count = self._obstacle_source_counts(obstacles)
        return MapFrame(
            image=image,
            x_m=x_m,
            y_m=y_m,
            yaw_deg=yaw_deg % 360.0,
            camera_obstacles=camera_count,
            lidar_obstacles=lidar_count,
        )

    def _mock_detection(self, label: str, confidence: float, index: int) -> TrashDetection:
        image = self._mock_overlay_image(index)
        return TrashDetection(image=image, label=label, confidence=confidence, timestamp=self._now(), position=f"x={1.2 + index:.1f}m, y={0.8 + index * 0.4:.1f}m")

    @staticmethod
    def _mock_overlay_image(index: int) -> np.ndarray:
        """Mock 'model overlay': a camera-like scene with a blended red mask.

        Mirrors the real detector overlay so the validation view shows what the
        model would have recognised even when running on mock data.
        """

        h, w = 200, 280
        image = np.empty((h, w, 3), dtype=np.uint8)
        image[..., 0] = 58
        image[..., 1] = 82
        image[..., 2] = 70
        # The detected litter object.
        y0 = 55 + index * 14
        x0 = 92
        image[y0 : y0 + 78, x0 : x0 + 96] = [158, 146, 120]
        # Model mask overlay (red), blended over the recognised region.
        mask_region = image[y0 + 8 : y0 + 70, x0 + 8 : x0 + 88].astype(np.float32)
        red = np.array([220, 40, 40], dtype=np.float32)
        image[y0 + 8 : y0 + 70, x0 + 8 : x0 + 88] = (0.55 * mask_region + 0.45 * red).astype(np.uint8)
        return image

    def _mock_log(self, level: str, source: str, message: str) -> LogEntry:
        return LogEntry(timestamp=self._now(), level=level, source=source, message=message)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _seed_mock_map_features(self) -> None:
        now = time.monotonic()
        self._camera_obstacles.extend(
            [
                ObstacleDetection(1.15, -0.85, "camera", "Kamera-Hindernis", 0.82, 0.22, now),
                ObstacleDetection(-1.35, 1.45, "camera", "Kamera-Hindernis", 0.74, 0.18, now),
            ]
        )
        for x_m, y_m in [(2.1, -0.7), (2.35, -0.72), (2.6, -0.65), (-0.8, -1.7), (-0.55, -1.95)]:
            obstacle = ObstacleDetection(x_m, y_m, "lidar", "LiDAR-Hindernis", 1.0, 0.14, now)
            self._lidar_obstacles[self._obstacle_cell(x_m, y_m)] = obstacle
        self._litter_markers.extend(
            [
                LitterMapMarker(1.2, 0.8, now),
                LitterMapMarker(2.2, 1.2, now),
            ]
        )

    def _current_obstacles(self) -> list[ObstacleDetection]:
        self._prune_obstacles()
        obstacles = list(self._camera_obstacles) + list(self._lidar_obstacles.values())
        obstacles.sort(key=lambda item: item.detected_at, reverse=True)
        return obstacles[: self.config.max_obstacles]

    def _prune_obstacles(self) -> None:
        cutoff = time.monotonic() - self.config.obstacle_max_age_s
        self._camera_obstacles = deque(
            (obstacle for obstacle in self._camera_obstacles if obstacle.detected_at >= cutoff),
            maxlen=self.config.max_obstacles,
        )
        stale_lidar = [cell for cell, obstacle in self._lidar_obstacles.items() if obstacle.detected_at < cutoff]
        for cell in stale_lidar:
            del self._lidar_obstacles[cell]

    def _store_obstacle(self, obstacle: ObstacleDetection) -> None:
        if obstacle.source == "lidar":
            self._lidar_obstacles[self._obstacle_cell(obstacle.x_m, obstacle.y_m)] = obstacle
            self._trim_lidar_obstacles()
            return
        self._camera_obstacles.append(obstacle)

    def _trim_lidar_obstacles(self) -> None:
        overflow = len(self._lidar_obstacles) - self.config.max_obstacles
        if overflow <= 0:
            return
        oldest = sorted(self._lidar_obstacles.items(), key=lambda item: item[1].detected_at)
        for cell, _ in oldest[:overflow]:
            del self._lidar_obstacles[cell]

    def _obstacle_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        cell_size = self.config.lidar_obstacle_cell_size_m
        return int(math.floor(x_m / cell_size)), int(math.floor(y_m / cell_size))

    @staticmethod
    def _obstacle_source_counts(obstacles: list[ObstacleDetection]) -> tuple[int, int]:
        camera_count = sum(1 for obstacle in obstacles if obstacle.source == "camera")
        lidar_count = sum(1 for obstacle in obstacles if obstacle.source == "lidar")
        return camera_count, lidar_count

    def _draw_obstacles(
        self,
        image: np.ndarray,
        center: int,
        scale: int,
        obstacles: list[ObstacleDetection],
    ) -> None:
        for obstacle in reversed(obstacles):
            px = center + int(obstacle.x_m * scale)
            py = center - int(obstacle.y_m * scale)
            if not (0 <= px < image.shape[1] and 0 <= py < image.shape[0]):
                continue
            half_size = max(5, int(obstacle.radius_m * scale))
            image[
                max(py - half_size, 0) : min(py + half_size + 1, image.shape[0]),
                max(px - half_size, 0) : min(px + half_size + 1, image.shape[1]),
            ] = [0, 0, 0]

    def _draw_litter_markers(
        self,
        image: np.ndarray,
        center: int,
        scale: int,
        markers: list[LitterMapMarker],
    ) -> None:
        for marker in markers:
            px = center + int(marker.x_m * scale)
            py = center - int(marker.y_m * scale)
            if not (0 <= px < image.shape[1] and 0 <= py < image.shape[0]):
                continue
            arm = 9
            self._draw_line(image, px - arm, py - arm, px + arm, py + arm, [220, 38, 38])
            self._draw_line(image, px - arm, py + arm, px + arm, py - arm, [220, 38, 38])

    def _store_litter_marker_from_payload(self, payload: dict[str, Any]) -> None:
        x_m, y_m = self._map_xy_from_payload(payload)
        if x_m is None or y_m is None:
            x_m, y_m, _ = self._robot_pose_for_obstacles()
        self._litter_markers.append(LitterMapMarker(x_m=x_m, y_m=y_m))

    def _ingest_obstacle_payload(self, payload: dict[str, Any], default_source: str = "camera") -> int:
        raw_items = payload.get("obstacles") or payload.get("objects") or payload.get("detections")
        if raw_items is None:
            raw_items = [payload]
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            return 0

        stored = 0
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            obstacle = self._obstacle_from_payload(raw_item, default_source)
            if obstacle is None:
                continue
            self._store_obstacle(obstacle)
            stored += 1
        return stored

    def _obstacle_from_payload(self, payload: dict[str, Any], default_source: str) -> ObstacleDetection | None:
        source = str(payload.get("source") or default_source).strip().lower()
        if source not in {"camera", "lidar"}:
            source = default_source
        label = str(
            payload.get("label")
            or payload.get("class")
            or payload.get("name")
            or payload.get("type")
            or "Hindernis"
        )
        confidence = self._confidence_value(payload.get("confidence"))
        radius_m = self._float_value(payload.get("radius_m") or payload.get("size_m"), 0.18)

        x_m, y_m = self._map_xy_from_payload(payload)
        if x_m is None or y_m is None:
            projection = self._project_camera_obstacle(payload)
            if projection is None:
                return None
            x_m, y_m = projection

        return ObstacleDetection(
            x_m=x_m,
            y_m=y_m,
            source=source,
            label=label,
            confidence=confidence,
            radius_m=radius_m,
            detected_at=time.monotonic(),
        )

    def _map_xy_from_payload(self, payload: dict[str, Any]) -> tuple[float | None, float | None]:
        if "x" in payload and "y" in payload:
            return self._float_value(payload["x"]), self._float_value(payload["y"])
        position = payload.get("position") or payload.get("map_position")
        if isinstance(position, dict) and "x" in position and "y" in position:
            return self._float_value(position["x"]), self._float_value(position["y"])
        return None, None

    def _project_camera_obstacle(self, payload: dict[str, Any]) -> tuple[float, float] | None:
        distance_m = self._float_value(
            payload.get("distance_m")
            or payload.get("depth_m")
            or payload.get("range_m")
            or payload.get("distance")
        )
        if distance_m is None:
            return None

        bearing_deg = self._float_value(payload.get("bearing_deg") or payload.get("angle_deg"), 0.0)
        if "bearing_deg" not in payload and "angle_deg" not in payload:
            bearing_deg = self._bearing_from_bbox(payload) or 0.0

        robot_x, robot_y, yaw_deg = self._robot_pose_for_obstacles()
        heading = math.radians(yaw_deg + bearing_deg)
        return robot_x + math.cos(heading) * distance_m, robot_y + math.sin(heading) * distance_m

    def _bearing_from_bbox(self, payload: dict[str, Any]) -> float | None:
        bbox = payload.get("bbox") or payload.get("bounding_box")
        frame_width = self._float_value(payload.get("frame_width") or payload.get("image_width"))
        if not isinstance(bbox, list) or len(bbox) != 4 or not frame_width:
            return None

        x0, _, x2_or_w, _ = [self._float_value(value, 0.0) for value in bbox]
        center_x = (x0 + x2_or_w) / 2.0 if x2_or_w > x0 else x0 + x2_or_w / 2.0
        normalized = max(0.0, min(center_x / frame_width, 1.0))
        return (normalized - 0.5) * self.config.camera_fov_h_deg

    def _robot_pose_for_obstacles(self) -> tuple[float, float, float]:
        return self._mock_pose

    @staticmethod
    def _confidence_value(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return {"high": 0.9, "medium": 0.6, "low": 0.3}.get(str(value).lower(), 0.0)

    @staticmethod
    def _float_value(value: Any, default: float | None = None) -> float | None:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _payload_is_obstacle(payload: dict[str, Any]) -> bool:
        if any(key in payload for key in ("obstacles", "objects", "detections", "obstacle_detected")):
            return True
        raw_kind = payload.get("type") or payload.get("label") or payload.get("class") or payload.get("category")
        kind = str(raw_kind or "").lower()
        return "obstacle" in kind or "hindernis" in kind

    def _extract_lidar_points(self, payload: dict[str, Any]) -> np.ndarray | None:
        points = payload.get("points")
        if points is None:
            points = payload.get("data", {}).get("data", {}).get("points", [])
        if not points:
            return None

        positions = np.array(points, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] < 3:
            return None
        positions = positions[:, :3]
        if self._points_are_robot_relative(payload):
            positions = self._transform_robot_points_to_world(positions)
        return positions

    @staticmethod
    def _points_are_robot_relative(payload: dict[str, Any]) -> bool:
        frame = str(
            payload.get("frame")
            or payload.get("frame_id")
            or payload.get("coordinate_frame")
            or payload.get("data", {}).get("header", {}).get("frame_id", "")
        ).lower()
        return frame in {"robot", "base", "base_link", "body", "go2", "camera", "lidar"}

    def _transform_robot_points_to_world(self, points: np.ndarray) -> np.ndarray:
        robot_x, robot_y, yaw_deg = self._robot_pose_for_obstacles()
        yaw = math.radians(yaw_deg)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world = points.copy()
        world[:, 0] = robot_x + points[:, 0] * cos_yaw - points[:, 1] * sin_yaw
        world[:, 1] = robot_y + points[:, 0] * sin_yaw + points[:, 1] * cos_yaw
        return world

    def _update_lidar_obstacles_from_points(self, points: np.ndarray) -> int:
        mask = (points[:, 2] >= OBSTACLE_Z_MIN_M) & (points[:, 2] <= OBSTACLE_Z_MAX_M)
        obstacle_points = points[mask]
        if obstacle_points.size == 0:
            return 0

        now = time.monotonic()
        cells: dict[tuple[int, int], tuple[float, float]] = {}
        for x_m, y_m, _ in obstacle_points:
            cell = self._obstacle_cell(float(x_m), float(y_m))
            if cell not in cells:
                cells[cell] = (float(x_m), float(y_m))

        for x_m, y_m in cells.values():
            self._store_obstacle(ObstacleDetection(x_m, y_m, "lidar", "LiDAR-Hindernis", 1.0, 0.14, now))
        return len(cells)

    def _apply_control_motion_state(self, normalized_action: str) -> None:
        if normalized_action == "stop":
            self._manual_velocity = (0.0, 0.0, 0.0)
            self._last_motion_update = time.monotonic()
        elif normalized_action == "start_autonomous":
            self._reset_mock_mission()
        elif normalized_action in {"return_home", "manual_override"}:
            self._manual_velocity = (0.0, 0.0, 0.0)
            self._last_motion_update = time.monotonic()

    def _reset_mock_mission(self) -> None:
        self._mock_pose = self._home_pose
        self._mock_path.clear()
        self._mock_path.append((self._home_pose[0], self._home_pose[1]))
        self._mock_phase = 0.0
        self._manual_velocity = (0.0, 0.0, 0.0)
        self._last_motion_update = time.monotonic()

    def _advance_mock_motion(self, dt: float) -> None:
        if dt <= 0.0:
            return

        mode = self._status.mode
        if mode == "AUTONOMOUS":
            self._advance_autonomous_mock(dt)
        elif mode == "RETURN_HOME":
            self._advance_return_home_mock(dt)
        elif mode == "MANUAL":
            self._advance_manual_mock(dt)

    def _advance_autonomous_mock(self, dt: float) -> None:
        self._mock_phase += dt
        home_x, home_y, _ = self._home_pose
        angle = self._mock_phase * 0.8
        radius = min(3.0, 0.12 + self._mock_phase * 0.08)
        x_m = home_x + math.cos(angle) * radius
        y_m = home_y + math.sin(angle) * radius * 0.75
        yaw_deg = math.degrees(angle + math.pi / 2.0) % 360.0
        self._set_mock_pose(x_m, y_m, yaw_deg)

    def _advance_return_home_mock(self, dt: float) -> None:
        x_m, y_m, yaw_deg = self._mock_pose
        home_x, home_y, home_yaw = self._home_pose
        dx = home_x - x_m
        dy = home_y - y_m
        distance = math.hypot(dx, dy)
        if distance <= 0.03:
            self._set_mock_pose(home_x, home_y, home_yaw)
            self._status = RobotStatus("IDLE", self._status.battery_percent, self._status.connected)
            self._logs.append(self._mock_log("INFO", "mapping", "Return to start completed"))
            return

        step = min(distance, 0.55 * dt)
        next_x = x_m + dx / distance * step
        next_y = y_m + dy / distance * step
        yaw_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        self._set_mock_pose(next_x, next_y, yaw_deg)

    def _advance_manual_mock(self, dt: float) -> None:
        x_speed, y_speed, z_deg = self._manual_velocity
        if x_speed == 0.0 and y_speed == 0.0 and z_deg == 0.0:
            return

        x_m, y_m, yaw_deg = self._mock_pose
        next_yaw = (yaw_deg + z_deg * dt) % 360.0
        yaw_rad = math.radians(next_yaw)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        lateral_x = -math.sin(yaw_rad)
        lateral_y = math.cos(yaw_rad)
        next_x = x_m + (x_speed * forward_x + y_speed * lateral_x) * dt
        next_y = y_m + (x_speed * forward_y + y_speed * lateral_y) * dt
        self._set_mock_pose(next_x, next_y, next_yaw)

    def _set_mock_pose(self, x_m: float, y_m: float, yaw_deg: float) -> None:
        x_m = max(min(x_m, 5.5), -5.5)
        y_m = max(min(y_m, 5.5), -5.5)
        self._mock_pose = (x_m, y_m, yaw_deg % 360.0)
        if not self._mock_path or math.hypot(self._mock_path[-1][0] - x_m, self._mock_path[-1][1] - y_m) >= 0.03:
            self._mock_path.append((x_m, y_m))

    @staticmethod
    def _mode_for_action(normalized_action: str, current_mode: str) -> str:
        return {
            "stop": "STOPPED",
            "start_autonomous": "AUTONOMOUS",
            "return_home": "RETURN_HOME",
            "manual_override": "MANUAL",
            "save_map": current_mode,
        }.get(normalized_action, current_mode)

    @staticmethod
    def _normalize_action(action: str) -> str:
        return {
            "Stop": "stop",
            "Start": "start_autonomous",
            "Zurück zum Start": "return_home",
            "Zurueck zum Start": "return_home",
            "Manuelle Übernahme": "manual_override",
            "Manuelle Uebernahme": "manual_override",
            "Karte speichern": "save_map",
        }.get(action, action.strip().lower().replace(" ", "_"))

    @staticmethod
    def _clamp_speed(speed: float) -> float:
        return max(0.05, min(float(speed), 0.8))

    @staticmethod
    def _manual_velocity_for(direction: str, speed: float) -> tuple[float, float, float]:
        return {
            "forward": (speed, 0.0, 0.0),
            "backward": (-speed, 0.0, 0.0),
            "left": (0.0, speed, 0.0),
            "right": (0.0, -speed, 0.0),
            "turn_left": (0.0, 0.0, 45.0),
            "turn_right": (0.0, 0.0, -45.0),
            "stop": (0.0, 0.0, 0.0),
        }.get(direction, (0.0, 0.0, 0.0))

    @staticmethod
    def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: list[int]) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for idx in range(steps + 1):
            t = idx / steps
            x = round(x0 + (x1 - x0) * t)
            y = round(y0 + (y1 - y0) * t)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = color


class ZenohDashboardDataProvider(QueueDashboardDataProvider):
    """Zenoh-backed provider for the Gradio dashboard.

    Incoming robot data is converted into the queue-backed dashboard models.
    Mock data remains available as a visual fallback until the first real
    samples arrive.
    """

    def __init__(self, config: DashboardConfig | None = None) -> None:
        super().__init__(config)
        self.settings = ZenohDashboardSettings()
        self._status = RobotStatus(self.config.initial_mode, self._status.battery_percent, False)
        self._lock = RLock()
        self._session: Any | None = None
        self._subscribers: list[Any] = []
        self._connecting = False
        self._last_connect_attempt = 0.0
        self._last_connection_error: str | None = None
        self._connect_retry_interval_s = 5.0
        self._last_frame_received_at: float | None = None
        self._pose_history: deque[tuple[float, float]] = deque(maxlen=300)
        self._current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Live occupancy grid published by the PathPlannerAgent (single source of truth).
        self._grid_cell_size: float = 0.0
        self._grid_occupied: list[tuple[int, int]] = []
        self._grid_explored: list[tuple[int, int]] = []
        self._real_home_pose: tuple[float, float, float] | None = None
        self._last_override_publish_at = 0.0
        self._override_publish_interval_s = 0.25
        self._last_lidar_log_at = 0.0
        self._last_obstacle_log_at = 0.0
        self._connect()

    def handle_control(self, action: str) -> str:
        """Publish dashboard control actions to Zenoh."""

        self._drain_realtime_queues()
        normalized = self._normalize_action(action)
        self._apply_control_motion_state(normalized)
        if normalized == "start_autonomous":
            self._reset_real_mission_origin()
        elif normalized == "return_home":
            self._ensure_real_home_pose()
        payload = {
            "action": normalized,
            "label": action,
            "source": "dashboard",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            session = self._ensure_session()
            if normalized == "stop":
                self._publish_stop_command(session)
            session.put(self.settings.topic_robodog_command, json.dumps(payload).encode())
            message = f"Zenoh command published: {payload['action']}"
            level = "INFO"
        except RuntimeError as exc:
            message = f"Zenoh command failed: {exc}"
            level = "WARN"
        except Exception as exc:
            message = f"Zenoh command failed: {exc}"
            level = "ERROR"
            logger.exception("Failed to publish dashboard command")

        with self._lock:
            self._status = RobotStatus(
                self._mode_for_action(normalized, self._status.mode),
                self._status.battery_percent,
                level == "INFO",
            )
        self._safe_put(self.log_queue, LogEntry(self._now(), level, "control", message))
        return f"{self._now()} | {message}"

    def handle_manual_control(self, direction: str, speed: float) -> str:
        """Publish a manual controller movement command to Zenoh."""

        speed = self._clamp_speed(speed)
        self._manual_velocity = self._manual_velocity_for(direction, speed)
        self._last_motion_update = time.monotonic()
        try:
            session = self._ensure_session()
            command = self._movement_command_for_velocity(self._manual_velocity)
            session.put(self.settings.topic_movement_command, command.model_dump_json().encode())
            message = f"Zenoh manual command published: {direction} ({speed:.2f})"
            level = "INFO"
        except RuntimeError as exc:
            message = f"Zenoh manual command failed: {exc}"
            level = "WARN"
        except Exception as exc:
            message = f"Zenoh manual command failed: {exc}"
            level = "ERROR"
            logger.exception("Failed to publish manual dashboard command")

        with self._lock:
            self._status = RobotStatus("MANUAL", self._status.battery_percent, level == "INFO")
        self._safe_put(self.log_queue, LogEntry(self._now(), level, "control", message))
        return f"{self._now()} | {message}"

    def close(self) -> None:
        """Close the Zenoh session when the embedding process supports cleanup."""

        if self._session is not None:
            self._session.close()
            self._session = None

    def _drain_realtime_queues(self) -> None:
        if self._session is None:
            self._connect()
        super()._drain_realtime_queues()
        self._maintain_control_override()

    def _connect(self) -> None:
        if self._connecting or self._session is not None:
            return
        now = time.monotonic()
        if now - self._last_connect_attempt < self._connect_retry_interval_s:
            return
        self._last_connect_attempt = now
        self._connecting = True
        thread = Thread(target=self._connect_blocking, name="dashboard-zenoh-connect", daemon=True)
        thread.start()
        self._safe_put(self.log_queue, LogEntry(self._now(), "INFO", "zenoh", f"Connecting to {self.settings.router}"))

    def _connect_blocking(self) -> None:
        try:
            if not self._router_reachable(self.settings.router):
                raise ConnectionError(f"router not reachable: {self.settings.router}")
            session = self._open_session()
            self._subscribers = [
                session.declare_subscriber(self.settings.topic_frame, self._on_frame),
                session.declare_subscriber(self.settings.topic_visualization, self._on_visualization),
                session.declare_subscriber(self.settings.topic_alert, self._on_alert),
                session.declare_subscriber(self.settings.topic_odometry, self._on_odometry),
                session.declare_subscriber(self.settings.topic_lidar, self._on_lidar),
                session.declare_subscriber(self.settings.topic_obstacle, self._on_obstacle),
                session.declare_subscriber(self.settings.topic_occupancy_grid, self._on_occupancy_grid),
            ]
            with self._lock:
                self._session = session
                self._status = RobotStatus(self._status.mode, self._status.battery_percent, True)
                self._last_connection_error = None
            self._safe_put(
                self.log_queue,
                LogEntry(self._now(), "INFO", "zenoh", f"Connected to {self.settings.router}"),
            )
        except ConnectionError as exc:
            with self._lock:
                self._status = RobotStatus(self._status.mode, self._status.battery_percent, False)
            self._log_connection_error(str(exc))
        except Exception as exc:
            with self._lock:
                self._status = RobotStatus(self._status.mode, self._status.battery_percent, False)
            self._log_connection_error(str(exc))
            logger.exception("Failed to initialize Zenoh dashboard provider")
        finally:
            self._connecting = False

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        self._connect()
        raise RuntimeError("Zenoh is not connected yet")

    def _open_session(self) -> Any:
        if self._session is not None:
            return self._session

        import zenoh

        conf = zenoh.Config()
        if self.settings.router:
            conf.insert_json5("connect/endpoints", json.dumps([self.settings.router]))
        return zenoh.open(conf)

    def _log_connection_error(self, message: str) -> None:
        if message == self._last_connection_error:
            return
        self._last_connection_error = message
        self._safe_put(self.log_queue, LogEntry(self._now(), "WARN", "zenoh", f"Connection failed: {message}"))

    @staticmethod
    def _router_reachable(router: str) -> bool:
        if not router.startswith("tcp/"):
            return True
        target = router.removeprefix("tcp/")
        host, _, port_raw = target.rpartition(":")
        if not host or not port_raw:
            return True
        try:
            with socket.create_connection((host, int(port_raw)), timeout=0.2):
                return True
        except OSError:
            return False

    def _on_frame(self, sample: Any) -> None:
        frame = self._decode_image_sample(sample)
        if frame is not None:
            self._safe_put(self.camera_queue, self._camera_frame(frame))

    def _on_visualization(self, sample: Any) -> None:
        # The visualization topic carries the detector's annotated overlay
        # (camera frame with the recognised litter mask drawn on top).
        frame = self._decode_image_sample(sample)
        if frame is not None:
            self._last_overlay = frame
            self._safe_put(self.camera_queue, self._camera_frame(frame))

    def _annotate_bbox(
        self,
        image: np.ndarray,
        bbox: list[int],
        src_w: int | None,
        src_h: int | None,
    ) -> np.ndarray:
        """Draw a bounding-box rectangle on the (RGB) image, scaled to its display size."""
        img = image.copy()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        if src_w and src_h and (src_w != w or src_h != h):
            x1 = int(x1 * w / src_w)
            y1 = int(y1 * h / src_h)
            x2 = int(x2 * w / src_w)
            y2 = int(y2 * h / src_h)
        # cv2 works on BGR-ordered arrays, but rectangle colour (0,255,0) is green in both orderings
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(
            img, "litter", (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
        )
        return img

    def _on_alert(self, sample: Any) -> None:
        payload = self._decode_json_sample(sample)
        if not payload:
            return

        description = str(payload.get("description") or "litter")
        confidence = self._confidence_value(payload.get("confidence"))
        coverage = payload.get("pixel_coverage")
        label = description[:48] if description else "litter"
        if coverage is not None:
            label = f"{label} ({float(coverage):.1%})"
        camera_image = self._last_camera.image if self._last_camera is not None else None
        fallback_image = self._mock_detection("litter", confidence, 0).image
        image = pick_detection_image(self._last_overlay, camera_image, fallback_image)
        position = self._position_text(payload)
        detection = TrashDetection(
            image=image,
            label=label,
            confidence=confidence,
            timestamp=self._timestamp_from_payload(payload),
            position=position,
        )
        with self._lock:
            self._store_litter_marker_from_payload(payload)
            map_frame = self._render_map_frame()
        self._safe_put(self.detection_queue, detection)
        self._safe_put(self.map_queue, map_frame)
        self._safe_put(self.log_queue, LogEntry(self._now(), "WARN", "detector", f"Litter alert: {label} at {position}"))
        if self._payload_is_obstacle(payload):
            stored = self._ingest_obstacle_payload(payload, default_source="camera")
            if stored:
                self._safe_put(self.log_queue, LogEntry(self._now(), "INFO", "mapping", f"Camera obstacle update: {stored}"))

    def _on_odometry(self, sample: Any) -> None:
        payload = self._decode_json_sample(sample)
        if not payload:
            return

        try:
            from litter_detection.agent.models import OdometryState

            odom = OdometryState.from_raw(payload)
            if odom is None:
                raise ValueError("unsupported odometry payload")
            yaw_deg = self._yaw_from_quaternion(odom.quaternion)
            with self._lock:
                self._current_pose = (odom.x, odom.y, yaw_deg)
                if self._real_home_pose is None:
                    self._real_home_pose = (odom.x, odom.y, yaw_deg)
                self._pose_history.append((odom.x, odom.y))
                self._status = RobotStatus(self._status.mode, self._status.battery_percent, True)
            self._safe_put(self.map_queue, self._render_map_frame())
        except Exception as exc:
            self._safe_put(self.log_queue, LogEntry(self._now(), "WARN", "odometry", f"Could not parse odometry: {exc}"))

    def _on_obstacle(self, sample: Any) -> None:
        payload = self._decode_json_sample(sample)
        if not payload:
            return
        try:
            with self._lock:
                stored = self._ingest_obstacle_payload(payload, default_source="camera")
                frame = self._render_map_frame()
            if stored:
                self._safe_put(self.map_queue, frame)
                now = time.monotonic()
                if now - self._last_obstacle_log_at > 2.0:
                    self._last_obstacle_log_at = now
                    self._safe_put(self.log_queue, LogEntry(self._now(), "INFO", "mapping", f"Camera obstacles mapped: {stored}"))
        except Exception as exc:
            self._safe_put(self.log_queue, LogEntry(self._now(), "WARN", "mapping", f"Could not parse obstacle payload: {exc}"))

    def _on_lidar(self, sample: Any) -> None:
        payload = self._decode_json_sample(sample)
        if not payload:
            return
        try:
            points = self._extract_lidar_points(payload)
            if points is None:
                return
            with self._lock:
                stored = self._update_lidar_obstacles_from_points(points)
                frame = self._render_map_frame()
            if stored:
                self._safe_put(self.map_queue, frame)
                now = time.monotonic()
                if now - self._last_lidar_log_at > 5.0:
                    self._last_lidar_log_at = now
                    self._safe_put(self.log_queue, LogEntry(self._now(), "INFO", "mapping", f"LiDAR obstacles mapped: {stored}"))
        except Exception as exc:
            self._safe_put(self.log_queue, LogEntry(self._now(), "WARN", "lidar", f"Could not parse LiDAR: {exc}"))

    def _on_occupancy_grid(self, sample: Any) -> None:
        payload = self._decode_json_sample(sample)
        if not payload:
            return
        try:
            cell_size = self._float_value(payload.get("cell_size"), 0.0) or 0.0
            occupied = [
                (int(c[0]), int(c[1]))
                for c in payload.get("occupied", [])
                if isinstance(c, (list, tuple)) and len(c) == 2
            ]
            explored = [
                (int(c[0]), int(c[1]))
                for c in payload.get("explored", [])
                if isinstance(c, (list, tuple)) and len(c) == 2
            ]
            with self._lock:
                self._grid_cell_size = cell_size
                self._grid_occupied = occupied
                self._grid_explored = explored
                frame = self._render_map_frame()
            self._safe_put(self.map_queue, frame)
        except Exception as exc:
            self._safe_put(self.log_queue, LogEntry(self._now(), "WARN", "mapping", f"Could not parse occupancy grid: {exc}"))

    def _camera_frame(self, image: np.ndarray) -> CameraFrame:
        now = time.monotonic()
        fps = 0.0
        if self._last_frame_received_at is not None:
            delta = now - self._last_frame_received_at
            fps = 1.0 / delta if delta > 0 else 0.0
        self._last_frame_received_at = now
        return CameraFrame(image=image, timestamp=self._now(), fps=fps)

    def _render_map_frame(self) -> MapFrame:
        with self._lock:
            x_m, y_m, yaw_deg = self._current_pose
            points = list(self._pose_history)
            obstacles = self._current_obstacles()
            grid_cell_size = self._grid_cell_size
            grid_explored = list(self._grid_explored)
            grid_occupied = list(self._grid_occupied)

        size = 520
        image = np.full((size, size, 3), 245, dtype=np.uint8)
        image[::40, :, :] = 226
        image[:, ::40, :] = 226
        center = size // 2
        scale = 40

        # Real occupancy grid (explored footprint + obstacles) drawn underneath
        # everything else so the path, robot and markers stay readable on top.
        self._draw_occupancy_grid(image, center, scale, grid_cell_size, grid_explored, grid_occupied)

        for x, y in points:
            px = center + int(x * scale)
            py = center - int(y * scale)
            if 0 <= px < size and 0 <= py < size:
                image[max(py - 2, 0) : py + 3, max(px - 2, 0) : px + 3] = [247, 184, 25]

        self._draw_obstacles(image, center, scale, obstacles)

        robot_px = center + int(x_m * scale)
        robot_py = center - int(y_m * scale)
        if 0 <= robot_px < size and 0 <= robot_py < size:
            image[max(robot_py - 10, 0) : robot_py + 11, max(robot_px - 10, 0) : robot_px + 11] = [247, 184, 25]
            heading = math.radians(yaw_deg)
            tip_x = robot_px + int(math.cos(heading) * 22)
            tip_y = robot_py - int(math.sin(heading) * 22)
            self._draw_line(image, robot_px, robot_py, tip_x, tip_y, [146, 95, 8])
        self._draw_litter_markers(image, center, scale, list(self._litter_markers))

        camera_count, lidar_count = self._obstacle_source_counts(obstacles)
        explored_cells = len(grid_explored)
        explored_area_m2 = explored_cells * grid_cell_size * grid_cell_size
        return MapFrame(
            image=image,
            x_m=x_m,
            y_m=y_m,
            yaw_deg=yaw_deg,
            camera_obstacles=camera_count,
            lidar_obstacles=lidar_count,
            explored_cells=explored_cells,
            explored_area_m2=explored_area_m2,
        )

    def _draw_occupancy_grid(
        self,
        image: np.ndarray,
        center: int,
        scale: int,
        cell_size: float,
        explored: list[tuple[int, int]],
        occupied: list[tuple[int, int]],
    ) -> None:
        """Render the planner's occupancy grid: explored area (light) + obstacles (dark)."""
        if cell_size <= 0.0:
            return
        half = cell_size / 2.0
        cell_px = max(1, int(round(cell_size * scale)))
        explored_color = [201, 231, 207]  # light green — area the robodog has seen
        occupied_color = [55, 65, 81]      # dark slate — obstacle cells from the real grid
        for cells, color in ((explored, explored_color), (occupied, occupied_color)):
            for cx, cy in cells:
                wx = cx * cell_size + half
                wy = cy * cell_size + half
                px = center + int(wx * scale)
                py = center - int(wy * scale)
                self._fill_cell(image, px, py, cell_px, color)

    @staticmethod
    def _fill_cell(image: np.ndarray, px: int, py: int, size_px: int, color: list[int]) -> None:
        half = size_px // 2
        y0 = max(py - half, 0)
        y1 = min(py + half + 1, image.shape[0])
        x0 = max(px - half, 0)
        x1 = min(px + half + 1, image.shape[1])
        if y1 > y0 and x1 > x0:
            image[y0:y1, x0:x1] = color

    def _publish_stop_command(self, session: Any) -> None:
        from litter_detection.agent.models import MovementCommand, MovementSource

        command = MovementCommand(x=0.0, y=0.0, z_deg=0.0, source=MovementSource.controller)
        session.put(self.settings.topic_movement_command, command.model_dump_json().encode())

    def _maintain_control_override(self) -> None:
        if self._session is None:
            return
        if self._status.mode not in {"STOPPED", "RETURN_HOME", "MANUAL"}:
            return
        now = time.monotonic()
        if now - self._last_override_publish_at < self._override_publish_interval_s:
            return
        self._last_override_publish_at = now

        try:
            if self._status.mode == "STOPPED":
                self._publish_stop_command(self._session)
            elif self._status.mode == "RETURN_HOME":
                command = self._return_home_command()
                self._session.put(self.settings.topic_movement_command, command.model_dump_json().encode())
            elif self._status.mode == "MANUAL":
                command = self._movement_command_for_velocity(self._manual_velocity)
                self._session.put(self.settings.topic_movement_command, command.model_dump_json().encode())
        except Exception:
            logger.debug("Could not maintain dashboard control override", exc_info=True)

    def _return_home_command(self):
        from litter_detection.agent.models import MovementCommand, MovementSource

        self._ensure_real_home_pose()
        with self._lock:
            x_m, y_m, yaw_deg = self._current_pose
            home_x, home_y, _ = self._real_home_pose or self._home_pose

        dx = home_x - x_m
        dy = home_y - y_m
        distance = math.hypot(dx, dy)
        if distance <= 0.15:
            with self._lock:
                self._status = RobotStatus("IDLE", self._status.battery_percent, self._status.connected)
            self._safe_put(self.log_queue, LogEntry(self._now(), "INFO", "control", "Return to start completed"))
            return MovementCommand(source=MovementSource.controller)

        target_heading_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        heading_error = self._shortest_angle(yaw_deg % 360.0, target_heading_deg)
        if abs(heading_error) > 8.0:
            return MovementCommand(
                x=0.0,
                y=0.0,
                z_deg=math.copysign(20.0, heading_error),
                source=MovementSource.controller,
            )
        return MovementCommand(x=min(0.25, distance), y=0.0, z_deg=0.0, source=MovementSource.controller)

    def _movement_command_for_velocity(self, velocity: tuple[float, float, float]):
        from litter_detection.agent.models import MovementCommand, MovementSource

        x_speed, y_speed, z_deg = velocity
        return MovementCommand(x=x_speed, y=y_speed, z_deg=z_deg, source=MovementSource.controller)

    def _reset_real_mission_origin(self) -> None:
        with self._lock:
            self._real_home_pose = self._current_pose
            self._pose_history.clear()
            self._pose_history.append((self._current_pose[0], self._current_pose[1]))

    def _ensure_real_home_pose(self) -> None:
        with self._lock:
            if self._real_home_pose is None:
                self._real_home_pose = self._current_pose

    def _robot_pose_for_obstacles(self) -> tuple[float, float, float]:
        with self._lock:
            return self._current_pose

    @staticmethod
    def _decode_image_sample(sample: Any) -> np.ndarray | None:
        try:
            import cv2

            frame_bgr = cv2.imdecode(np.frombuffer(bytes(sample.payload), np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            logger.exception("Could not decode image sample")
            return None

    @staticmethod
    def _decode_json_sample(sample: Any) -> dict[str, Any] | None:
        try:
            return json.loads(bytes(sample.payload).decode())
        except Exception:
            logger.exception("Could not decode JSON sample")
            return None

    @staticmethod
    def _safe_put(queue: Queue, item: Any) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except Empty:
                pass
        queue.put_nowait(item)

    def _position_text(self, payload: dict[str, Any]) -> str:
        if "x" in payload and "y" in payload:
            return f"x={float(payload['x']):.2f}m, y={float(payload['y']):.2f}m"
        with self._lock:
            x_m, y_m, _ = self._current_pose
        return f"x={x_m:.2f}m, y={y_m:.2f}m"

    @staticmethod
    def _timestamp_from_payload(payload: dict[str, Any]) -> str:
        raw = payload.get("timestamp")
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(raw).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(raw, str):
            return raw
        return QueueDashboardDataProvider._now()

    @staticmethod
    def _yaw_from_quaternion(quaternion: list[float]) -> float:
        qx, qy, qz, qw = quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _shortest_angle(current_deg: float, target_deg: float) -> float:
        return (target_deg - current_deg + 180.0) % 360.0 - 180.0
