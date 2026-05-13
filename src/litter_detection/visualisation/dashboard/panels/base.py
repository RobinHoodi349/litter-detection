"""Shared panel helpers."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr


@dataclass(frozen=True)
class PanelTheme:
    """CSS color configuration for a dashboard panel."""

    name: str
    css_class: str
    border_color: str


class DashboardPanel:
    """Base class for panels with a common render contract."""

    def __init__(self, title: str, theme: PanelTheme) -> None:
        self.title = title
        self.theme = theme

    def render_header(self) -> None:
        """Render a themed panel header."""

        gr.Markdown(f"### {self.title}", elem_classes=["panel-title"])
