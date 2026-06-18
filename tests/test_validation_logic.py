"""Tests for the pure detection-validation logic (no Gradio dependency)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from litter_detection.visualisation.dashboard.panels.validation_logic import (
    Verdict,
    compute_stats,
    detection_key,
    next_pending_key,
)


def _detection(timestamp: str, label: str, position: str) -> SimpleNamespace:
    return SimpleNamespace(timestamp=timestamp, label=label, position=position)


class ComputeStatsTest(unittest.TestCase):
    def test_empty_input(self) -> None:
        stats = compute_stats([], {})
        self.assertEqual(stats.total, 0)
        self.assertEqual(stats.correct, 0)
        self.assertEqual(stats.incorrect, 0)
        self.assertEqual(stats.unsure, 0)
        self.assertEqual(stats.pending, 0)
        self.assertIsNone(stats.precision)

    def test_mixed_verdicts(self) -> None:
        keys = ["a", "b", "c", "d", "e", "f"]
        verdicts = {
            "a": Verdict.CORRECT,
            "b": Verdict.CORRECT,
            "c": Verdict.CORRECT,
            "d": Verdict.INCORRECT,
            "e": Verdict.UNSURE,
        }
        stats = compute_stats(keys, verdicts)
        self.assertEqual(stats.total, 6)
        self.assertEqual(stats.correct, 3)
        self.assertEqual(stats.incorrect, 1)
        self.assertEqual(stats.unsure, 1)
        self.assertEqual(stats.pending, 1)
        self.assertEqual(stats.precision, 75.0)

    def test_precision_none_without_decisions(self) -> None:
        stats = compute_stats(["a", "b"], {"a": Verdict.UNSURE})
        self.assertEqual(stats.correct, 0)
        self.assertEqual(stats.incorrect, 0)
        self.assertEqual(stats.pending, 1)
        self.assertIsNone(stats.precision)

    def test_ignores_verdicts_for_unknown_keys(self) -> None:
        stats = compute_stats(["a"], {"a": Verdict.CORRECT, "ghost": Verdict.INCORRECT})
        self.assertEqual(stats.total, 1)
        self.assertEqual(stats.correct, 1)
        self.assertEqual(stats.incorrect, 0)
        self.assertEqual(stats.precision, 100.0)


class NextPendingKeyTest(unittest.TestCase):
    def test_returns_first_pending(self) -> None:
        self.assertEqual(next_pending_key(["a", "b", "c"], {}), "a")

    def test_returns_pending_after_key(self) -> None:
        self.assertEqual(
            next_pending_key(["a", "b", "c"], {"a": Verdict.CORRECT}, after_key="a"),
            "b",
        )

    def test_skips_decided_keys(self) -> None:
        self.assertEqual(
            next_pending_key(["a", "b", "c"], {"b": Verdict.CORRECT}, after_key="a"),
            "c",
        )

    def test_wraps_around(self) -> None:
        self.assertEqual(
            next_pending_key(["a", "b", "c"], {"c": Verdict.CORRECT}, after_key="c"),
            "a",
        )

    def test_none_when_all_decided(self) -> None:
        verdicts = {"a": Verdict.CORRECT, "b": Verdict.INCORRECT}
        self.assertIsNone(next_pending_key(["a", "b"], verdicts))


class DetectionKeyTest(unittest.TestCase):
    def test_stable_for_same_fields(self) -> None:
        det = _detection("2026-06-18 10:00:00", "plastic_bottle", "x=1.0m, y=0.5m")
        self.assertEqual(detection_key(det), detection_key(det))

    def test_distinct_for_different_detections(self) -> None:
        a = _detection("2026-06-18 10:00:00", "plastic_bottle", "x=1.0m, y=0.5m")
        b = _detection("2026-06-18 10:00:00", "paper_cup", "x=1.0m, y=0.5m")
        self.assertNotEqual(detection_key(a), detection_key(b))


if __name__ == "__main__":
    unittest.main()
