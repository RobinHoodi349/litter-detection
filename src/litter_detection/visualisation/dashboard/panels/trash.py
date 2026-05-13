"""Trash detection list panel."""

from __future__ import annotations

import gradio as gr

from litter_detection.visualisation.dashboard.data_provider import DashboardDataProvider
from litter_detection.visualisation.dashboard.panels.base import DashboardPanel, PanelTheme


class TrashPanel(DashboardPanel):
    """Shows recent trash detections as a scrollable gallery/list."""

    def __init__(self, provider: DashboardDataProvider) -> None:
        super().__init__("Trash Bilder Liste", PanelTheme("trash", "panel-trash", "#0c3aa6"))
        self.provider = provider

    def render(self) -> list[gr.components.Component]:
        """Build the Gradio components for the panel."""

        with gr.Column(elem_classes=["dashboard-panel", self.theme.css_class, "small-panel"]):
            self.render_header()
            gallery = gr.Gallery(
                label=None,
                show_label=False,
                columns=1,
                rows=3,
                height=260,
                object_fit="cover",
                elem_classes=["scroll-panel"],
            )
        return [gallery]

    def update(self) -> tuple:
        """Return gallery items with metadata captions."""

        items = []
        for detection in self.provider.get_trash_detections():
            caption = (
                f"{detection.label} | conf={detection.confidence:.2f}\n"
                f"{detection.timestamp}\n{detection.position}"
            )
            items.append((detection.image, caption))
        return (items,)
