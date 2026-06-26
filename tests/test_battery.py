"""Tests for parsing the robot battery state-of-charge from a Zenoh payload."""

from __future__ import annotations

import unittest

from litter_detection.visualisation.dashboard.data_provider import QueueDashboardDataProvider


class BatteryParsingTest(unittest.TestCase):
    @staticmethod
    def _parse(payload: dict) -> int | None:
        return QueueDashboardDataProvider._battery_percent_from_payload(payload)

    def test_reads_percent_value(self) -> None:
        self.assertEqual(self._parse({"battery_percent": 87}), 87)

    def test_reads_soc_field(self) -> None:
        self.assertEqual(self._parse({"soc": 42}), 42)

    def test_accepts_numeric_string(self) -> None:
        self.assertEqual(self._parse({"battery_percent": "73"}), 73)

    def test_treats_fraction_as_percent(self) -> None:
        self.assertEqual(self._parse({"battery": 0.5}), 50)

    def test_clamps_to_range(self) -> None:
        self.assertEqual(self._parse({"soc": 150}), 100)
        self.assertEqual(self._parse({"soc": -5}), 0)

    def test_returns_none_without_battery_field(self) -> None:
        self.assertIsNone(self._parse({"x": 1.0, "y": 2.0}))


if __name__ == "__main__":
    unittest.main()
