"""Tests for the per-detection image source selection."""

from __future__ import annotations

import unittest

from litter_detection.visualisation.dashboard.data_provider import pick_detection_image


class PickDetectionImageTest(unittest.TestCase):
    def test_prefers_overlay(self) -> None:
        self.assertEqual(
            pick_detection_image("overlay", "camera", "fallback"),
            "overlay",
        )

    def test_uses_camera_when_no_overlay(self) -> None:
        self.assertEqual(
            pick_detection_image(None, "camera", "fallback"),
            "camera",
        )

    def test_uses_fallback_when_nothing_else(self) -> None:
        self.assertEqual(
            pick_detection_image(None, None, "fallback"),
            "fallback",
        )


if __name__ == "__main__":
    unittest.main()
