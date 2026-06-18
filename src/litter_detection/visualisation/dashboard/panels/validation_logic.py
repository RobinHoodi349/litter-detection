"""Pure logic for the detection-validation tab (no Gradio dependency).

Kept separate from the Gradio panel so the verdict bookkeeping and the
precision metric can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class Verdict:
    """Allowed human verdicts for a detection."""

    CORRECT = "müll"
    INCORRECT = "kein_müll"
    UNSURE = "unsicher"


@dataclass(frozen=True)
class ValidationStats:
    """Aggregated counts and precision over a set of detections."""

    total: int
    correct: int
    incorrect: int
    unsure: int
    pending: int
    precision: float | None


def detection_key(detection: Any) -> str:
    """Return a stable key identifying a detection across refreshes."""

    return f"{detection.timestamp}|{detection.label}|{detection.position}"


def compute_stats(keys: Sequence[str], verdicts: Mapping[str, str]) -> ValidationStats:
    """Count verdicts over the currently known detection keys.

    Verdicts whose key is not among ``keys`` are ignored, so stale entries do
    not distort the metric.
    """

    correct = incorrect = unsure = pending = 0
    for key in keys:
        verdict = verdicts.get(key)
        if verdict == Verdict.CORRECT:
            correct += 1
        elif verdict == Verdict.INCORRECT:
            incorrect += 1
        elif verdict == Verdict.UNSURE:
            unsure += 1
        else:
            pending += 1

    decided = correct + incorrect
    precision = (correct / decided * 100.0) if decided else None
    return ValidationStats(
        total=len(keys),
        correct=correct,
        incorrect=incorrect,
        unsure=unsure,
        pending=pending,
        precision=precision,
    )


def next_pending_key(
    ordered_keys: Sequence[str],
    verdicts: Mapping[str, str],
    after_key: str | None = None,
) -> str | None:
    """Return the next key without a verdict, searching cyclically.

    Starts right after ``after_key`` (if present) and wraps around. Returns
    ``None`` when every key already has a verdict.
    """

    count = len(ordered_keys)
    if count == 0:
        return None

    if after_key in ordered_keys:
        start = ordered_keys.index(after_key) + 1
    else:
        start = 0

    for offset in range(count):
        key = ordered_keys[(start + offset) % count]
        if key not in verdicts:
            return key
    return None
