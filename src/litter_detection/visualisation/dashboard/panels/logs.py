"""Realtime log panel."""

from __future__ import annotations

import gradio as gr

from litter_detection.visualisation.dashboard.data_provider import DashboardDataProvider, LogEntry
from litter_detection.visualisation.dashboard.panels.base import DashboardPanel, PanelTheme


class LogsPanel(DashboardPanel):
    """Shows structured, filterable robot logs."""

    def __init__(self, provider: DashboardDataProvider) -> None:
        super().__init__("Log-Daten", PanelTheme("logs", "panel-logs", "#22b8c7"))
        self.provider = provider

    def render(self) -> list[gr.components.Component]:
        """Build the Gradio components for the panel."""

        with gr.Column(elem_classes=["dashboard-panel", self.theme.css_class, "small-panel"]):
            self.render_header()
            with gr.Row(elem_classes=["compact-row"]):
                gr.Markdown("Level", elem_classes=["filter-label"])
                level = gr.Dropdown(
                    ["ALL", "INFO", "WARN", "ERROR"],
                    value="ALL",
                    label=None,
                    show_label=False,
                    container=False,
                    elem_classes=["level-filter"],
                )
                count = gr.Markdown("Eintraege: **0**", elem_classes=["panel-meta", "log-count"])
            log_html = gr.HTML(elem_classes=["log-frame"])
        return [level, count, log_html]

    def update(self, level_filter: str) -> tuple:
        """Return log count and color-coded HTML entries."""

        entries = self.provider.get_logs(level_filter)
        return f"Eintraege: **{len(entries)}**", self._render_entries(entries)

    @staticmethod
    def _render_entries(entries: list[LogEntry]) -> str:
        rows = []
        for entry in entries:
            level_class = entry.level.lower()
            rows.append(
                f"<div class='log-entry {level_class}'>"
                f"<span class='log-level {level_class}'>{entry.level}</span>"
                f"<span class='log-time'>{entry.timestamp}</span>"
                f"<span class='log-source'>{entry.source}</span>"
                f"<span class='log-message'>{entry.message}</span>"
                "</div>"
            )
        return "<div class='log-scroll'><div class='log-list'>" + "".join(rows) + "</div></div>"
