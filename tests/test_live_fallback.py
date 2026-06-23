"""Tests for the live-vs-demo fallback of the dashboard data provider.

Live data must be shown while fresh frames keep arriving; once the feed goes
silent past ``live_timeout_s`` (or before any real frame), the animated demo
(mock) data is shown again.
"""

from __future__ import annotations

import unittest

import numpy as np

from litter_detection.visualisation.dashboard.config import DashboardConfig
from litter_detection.visualisation.dashboard.data_provider import (
    CameraFrame,
    MapFrame,
    QueueDashboardDataProvider,
)


class LiveFallbackTest(unittest.TestCase):
    def _provider(self) -> QueueDashboardDataProvider:
        return QueueDashboardDataProvider(DashboardConfig(provider="mock", live_timeout_s=2.0))

    @staticmethod
    def _real_camera_frame() -> CameraFrame:
        return CameraFrame(image=np.zeros((4, 4, 3), dtype=np.uint8), timestamp="t", fps=10.0)

    @staticmethod
    def _real_map_frame() -> MapFrame:
        return MapFrame(image=np.zeros((4, 4, 3), dtype=np.uint8), x_m=1.0, y_m=2.0, yaw_deg=0.0)

    def test_camera_shows_live_frame_while_fresh(self) -> None:
        provider = self._provider()
        frame = self._real_camera_frame()
        provider.camera_queue.put(frame)
        self.assertIs(provider.get_camera_frame(), frame)

    def test_camera_falls_back_to_demo_when_stale(self) -> None:
        provider = self._provider()
        frame = self._real_camera_frame()
        provider.camera_queue.put(frame)
        self.assertIs(provider.get_camera_frame(), frame)
        # Simulate the live feed going silent past the timeout.
        provider._last_camera_real_at -= provider.config.live_timeout_s + 1.0
        self.assertIsNot(provider.get_camera_frame(), frame)

    def test_camera_demo_before_any_real_frame(self) -> None:
        provider = self._provider()
        self.assertIsNotNone(provider.get_camera_frame())

    def test_map_shows_live_frame_while_fresh(self) -> None:
        provider = self._provider()
        frame = self._real_map_frame()
        provider.map_queue.put(frame)
        self.assertIs(provider.get_map_frame(), frame)

    def test_map_falls_back_to_demo_when_stale(self) -> None:
        provider = self._provider()
        frame = self._real_map_frame()
        provider.map_queue.put(frame)
        self.assertIs(provider.get_map_frame(), frame)
        provider._last_map_real_at -= provider.config.live_timeout_s + 1.0
        self.assertIsNot(provider.get_map_frame(), frame)


if __name__ == "__main__":
    unittest.main()
