"""Detection-validation panel (UI only).

Lets a human review the litter detections and mark each one as
``Müll`` / ``Kein Müll`` / ``Unsicher``. Verdicts are held in memory only for
now; persistence to disk is a later step. The precision metric and verdict
bookkeeping live in :mod:`validation_logic` so they can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from litter_detection.visualisation.dashboard.data_provider import (
    DashboardDataProvider,
    TrashDetection,
)
from litter_detection.visualisation.dashboard.panels.base import DashboardPanel, PanelTheme
from litter_detection.visualisation.dashboard.panels.validation_logic import (
    ValidationStats,
    Verdict,
    compute_stats,
    detection_key,
    next_pending_key,
)


_FILTER_PENDING = "Nur offene"

_VERDICT_TEXT = {
    Verdict.CORRECT: "✓ Müll",
    Verdict.INCORRECT: "✗ Kein Müll",
    Verdict.UNSURE: "? Unsicher",
}
_VERDICT_BADGE = {
    Verdict.CORRECT: "✓",
    Verdict.INCORRECT: "✗",
    Verdict.UNSURE: "?",
}
_VERDICT_CSS = {
    Verdict.CORRECT: "ok",
    Verdict.INCORRECT: "no",
    Verdict.UNSURE: "unsure",
}


@dataclass(frozen=True)
class ValidationComponents:
    """Rendered components needed to wire callbacks."""

    stats: gr.HTML
    image: gr.Image
    meta: gr.HTML
    gallery: gr.Gallery
    filter: gr.Radio
    current_key: gr.State
    button_correct: gr.Button
    button_incorrect: gr.Button
    button_unsure: gr.Button
    export: gr.Button


class ValidationPanel(DashboardPanel):
    """Review detections and assign a manual verdict (UI only)."""

    def __init__(self, provider: DashboardDataProvider) -> None:
        super().__init__("Validierung", PanelTheme("validation", "panel-validation", "#a78bfa"))
        self.provider = provider
        # In-memory only for this step; persistence follows later.
        self.verdicts: dict[str, str] = {}

    def render(self) -> ValidationComponents:
        """Build the validation tab components."""

        with gr.Column(elem_classes=["validation-root"]):
            stats = gr.HTML(value=self._stats_html(compute_stats([], {})), elem_classes=["val-stats"])
            with gr.Row(elem_classes=["val-body"], equal_height=False):
                with gr.Column(scale=3, elem_classes=["val-current"]):
                    gr.HTML(
                        "<div class='val-img-cap'>Modell-Overlay · erkannte Maske</div>",
                        elem_classes=["val-img-cap-block"],
                    )
                    image = gr.Image(
                        label=None,
                        show_label=False,
                        type="numpy",
                        interactive=False,
                        elem_classes=["val-image"],
                    )
                    meta = gr.HTML(value=self._meta_html(None, None), elem_classes=["val-meta"])
                    with gr.Row(elem_classes=["val-buttons"]):
                        button_correct = gr.Button("✓ Müll", elem_classes=["val-btn", "val-ok"])
                        button_incorrect = gr.Button("✗ Kein Müll", elem_classes=["val-btn", "val-no"])
                        button_unsure = gr.Button("? Unsicher", elem_classes=["val-btn", "val-unsure"])
                    export = gr.Button(
                        "⤓ Export (folgt)",
                        interactive=False,
                        elem_classes=["val-export"],
                    )
                with gr.Column(scale=2, elem_classes=["val-side"]):
                    filter_radio = gr.Radio(
                        ["Alle", _FILTER_PENDING],
                        value="Alle",
                        label="Filter",
                        elem_classes=["val-filter"],
                    )
                    gallery = gr.Gallery(
                        label=None,
                        show_label=False,
                        columns=2,
                        object_fit="cover",
                        elem_classes=["val-gallery"],
                    )
            current_key = gr.State(value=None)

        return ValidationComponents(
            stats=stats,
            image=image,
            meta=meta,
            gallery=gallery,
            filter=filter_radio,
            current_key=current_key,
            button_correct=button_correct,
            button_incorrect=button_incorrect,
            button_unsure=button_unsure,
            export=export,
        )

    # -- handlers -----------------------------------------------------------

    def refresh(self, current_key: str | None, filter_mode: str) -> tuple:
        """Recompute the whole view (used on load and on the timer tick)."""

        return self._view(current_key, filter_mode)

    def vote(self, verdict: str, current_key: str | None, filter_mode: str) -> tuple:
        """Record a verdict for the current detection and advance."""

        detections = self.provider.get_trash_detections()
        keys = [detection_key(item) for item in detections]
        if current_key in keys:
            self.verdicts[current_key] = verdict
            current_key = next_pending_key(keys, self.verdicts, after_key=current_key) or current_key
        return self._view(current_key, filter_mode)

    def on_filter(self, filter_mode: str, current_key: str | None) -> tuple:
        """Re-render when the gallery filter changes."""

        return self._view(current_key, filter_mode)

    def on_select(self, filter_mode: str, event: gr.SelectData) -> tuple:
        """Select a detection from the gallery."""

        detections = self.provider.get_trash_detections()
        keys = [detection_key(item) for item in detections]
        visible = self._visible_keys(detections, keys, filter_mode)
        current_key = None
        if event.index is not None and 0 <= event.index < len(visible):
            current_key = visible[event.index]
        return self._view(current_key, filter_mode)

    # -- view assembly ------------------------------------------------------

    def _view(self, current_key: str | None, filter_mode: str) -> tuple:
        detections = self.provider.get_trash_detections()
        keys = [detection_key(item) for item in detections]
        by_key = dict(zip(keys, detections))

        if current_key not in by_key:
            current_key = next_pending_key(keys, self.verdicts) or (keys[0] if keys else None)

        stats = compute_stats(keys, self.verdicts)
        current = by_key.get(current_key)
        image = current.image if current is not None else None
        meta_html = self._meta_html(current, self.verdicts.get(current_key))
        gallery = self._gallery_items(detections, keys, filter_mode)
        return (
            self._stats_html(stats),
            image,
            meta_html,
            gallery,
            current_key,
        )

    def _visible_keys(
        self,
        detections: list[TrashDetection],
        keys: list[str],
        filter_mode: str,
    ) -> list[str]:
        if filter_mode == _FILTER_PENDING:
            return [key for key in keys if key not in self.verdicts]
        return list(keys)

    def _gallery_items(
        self,
        detections: list[TrashDetection],
        keys: list[str],
        filter_mode: str,
    ) -> list[tuple]:
        items = []
        for detection, key in zip(detections, keys):
            if filter_mode == _FILTER_PENDING and key in self.verdicts:
                continue
            badge = _VERDICT_BADGE.get(self.verdicts.get(key), "•")
            caption = f"{badge} {detection.label} · {detection.confidence:.0%}"
            items.append((detection.image, caption))
        return items

    # -- html ---------------------------------------------------------------

    @staticmethod
    def _stats_html(stats: ValidationStats) -> str:
        precision = "—" if stats.precision is None else f"{stats.precision:.0f}%"
        return (
            "<div class='val-stat-row'>"
            f"<div class='val-stat'><span class='val-num'>{stats.total}</span>"
            "<span class='val-lbl'>Gesamt</span></div>"
            f"<div class='val-stat'><span class='val-num ok'>{stats.correct}</span>"
            "<span class='val-lbl'>Korrekt</span></div>"
            f"<div class='val-stat'><span class='val-num no'>{stats.incorrect}</span>"
            "<span class='val-lbl'>Fehlalarm</span></div>"
            f"<div class='val-stat'><span class='val-num unsure'>{stats.unsure}</span>"
            "<span class='val-lbl'>Unsicher</span></div>"
            f"<div class='val-stat'><span class='val-num'>{stats.pending}</span>"
            "<span class='val-lbl'>Offen</span></div>"
            f"<div class='val-stat val-precision'><span class='val-num'>{precision}</span>"
            "<span class='val-lbl'>Precision</span></div>"
            "</div>"
        )

    @staticmethod
    def _meta_html(detection: TrashDetection | None, verdict: str | None) -> str:
        if detection is None:
            return "<div class='val-meta-empty'>Keine Detektionen vorhanden.</div>"
        verdict_text = _VERDICT_TEXT.get(verdict, "offen")
        verdict_css = _VERDICT_CSS.get(verdict, "pending")
        return (
            "<div class='val-meta-grid'>"
            f"<span>Label</span><strong>{detection.label}</strong>"
            f"<span>Confidence</span><strong>{detection.confidence:.0%}</strong>"
            f"<span>Zeit</span><strong>{detection.timestamp}</strong>"
            f"<span>Position</span><strong>{detection.position}</strong>"
            f"<span>Bewertung</span><strong class='verdict-{verdict_css}'>{verdict_text}</strong>"
            "</div>"
        )
