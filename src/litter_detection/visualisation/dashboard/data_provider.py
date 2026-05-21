"""Data-provider interfaces and mock queues for the Gradio dashboard."""

from __future__ import annotations

import math
import json
import logging
import os
import socket
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Queue
from threading import RLock, Thread
from typing import Any, Protocol

import numpy as np

from litter_detection.visualisation.dashboard.config import DashboardConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZenohDashboardSettings:
    """Lightweight Zenoh settings for the dashboard runtime."""

    router: str = os.getenv("ZENOH_ROUTER", "tcp/localhost:7447")
    topic_frame: str = os.getenv("LITTER_TOPIC_FRAME", "robodog/sensors/go2_camera")
    topic_visualization: str = os.getenv("LITTER_TOPIC_VISUALIZATION", "litter/visualization")
    topic_alert: str = os.getenv("LITTER_TOPIC_ALERT", "litter/alert")
    topic_odometry: str = os.getenv("LITTER_TOPIC_ODOMETRY", "robodog/system_state/odometry")
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
        self._connect()

    def handle_control(self, action: str) -> str:
        """Publish dashboard control actions to Zenoh."""

        self._drain_realtime_queues()
        payload = {
            "action": self._normalize_action(action),
            "label": action,
            "source": "dashboard",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            session = self._ensure_session()
            if action == "Stop":
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

        mode_by_action = {
            "Stop": "STOPPED",
            "Start": "AUTONOMOUS",
            "Zurück zum Start": "RETURN_HOME",
            "Zurueck zum Start": "RETURN_HOME",
            "Manuelle Übernahme": "MANUAL",
            "Manuelle Uebernahme": "MANUAL",
            "Karte speichern": self._status.mode,
        }
        with self._lock:
            self._status = RobotStatus(
                mode_by_action.get(action, self._status.mode),
                self._status.battery_percent,
                level == "INFO",
            )
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
                self._status = RobotStatus(self.config.initial_mode, self._status.battery_percent, False)
            self._log_connection_error(str(exc))
        except Exception as exc:
            with self._lock:
                self._status = RobotStatus(self.config.initial_mode, self._status.battery_percent, False)
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
        frame = self._decode_image_sample(sample)
        if frame is not None:
            self._safe_put(self.camera_queue, self._camera_frame(frame))

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
        image = self._last_camera.image if self._last_camera is not None else self._mock_detection("litter", confidence, 0).image
        position = self._position_text(payload)
        detection = TrashDetection(
            image=image,
            label=label,
            confidence=confidence,
            timestamp=self._timestamp_from_payload(payload),
            position=position,
        )
        self._safe_put(self.detection_queue, detection)
        self._safe_put(self.log_queue, LogEntry(self._now(), "WARN", "detector", f"Litter alert: {label} at {position}"))

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
                self._pose_history.append((odom.x, odom.y))
                self._status = RobotStatus(self._status.mode, self._status.battery_percent, True)
            self._safe_put(self.map_queue, self._render_map_frame())
        except Exception as exc:
            self._safe_put(self.log_queue, LogEntry(self._now(), "WARN", "odometry", f"Could not parse odometry: {exc}"))

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

        size = 520
        image = np.full((size, size, 3), 245, dtype=np.uint8)
        image[::40, :, :] = 226
        image[:, ::40, :] = 226
        center = size // 2
        scale = 40

        for x, y in points:
            px = center + int(x * scale)
            py = center - int(y * scale)
            if 0 <= px < size and 0 <= py < size:
                image[max(py - 2, 0) : py + 3, max(px - 2, 0) : px + 3] = [247, 184, 25]

        robot_px = center + int(x_m * scale)
        robot_py = center - int(y_m * scale)
        if 0 <= robot_px < size and 0 <= robot_py < size:
            image[max(robot_py - 10, 0) : robot_py + 11, max(robot_px - 10, 0) : robot_px + 11] = [120, 76, 35]
            heading = math.radians(yaw_deg)
            tip_x = robot_px + int(math.cos(heading) * 22)
            tip_y = robot_py - int(math.sin(heading) * 22)
            self._draw_line(image, robot_px, robot_py, tip_x, tip_y, [180, 35, 24])

        return MapFrame(image=image, x_m=x_m, y_m=y_m, yaw_deg=yaw_deg)

    def _publish_stop_command(self, session: Any) -> None:
        from litter_detection.agent.models import MovementCommand, MovementSource

        command = MovementCommand(x=0.0, y=0.0, z_deg=0.0, source=MovementSource.controller)
        session.put(self.settings.topic_movement_command, command.model_dump_json().encode())

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
    def _confidence_value(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return {"high": 0.9, "medium": 0.6, "low": 0.3}.get(str(value).lower(), 0.0)

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
    def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: list[int]) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for idx in range(steps + 1):
            t = idx / steps
            x = round(x0 + (x1 - x0) * t)
            y = round(y0 + (y1 - y0) * t)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = color
