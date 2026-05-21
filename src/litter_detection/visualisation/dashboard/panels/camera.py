"""Camera panel for the Robodog dashboard."""

from __future__ import annotations

import gradio as gr

from litter_detection.visualisation.dashboard.data_provider import DashboardDataProvider
from litter_detection.visualisation.dashboard.panels.base import DashboardPanel, PanelTheme


class CameraPanel(DashboardPanel):
    """Shows the live robot camera feed with timestamp and FPS."""

    def __init__(self, provider: DashboardDataProvider) -> None:
        super().__init__("Kamera Bild", PanelTheme("camera", "panel-camera", "#9b8cf0"))
        self.provider = provider

    def render(self) -> list[gr.components.Component]:
        """Build the Gradio components for the panel."""

        with gr.Column(elem_classes=["dashboard-panel", self.theme.css_class]):
            self.render_header()
            image = gr.Image(label=None, show_label=False, type="numpy", height=360, elem_classes=["media-fill"])
            meta = gr.Markdown(elem_classes=["panel-meta"])
        return [image, meta]

    def update(self) -> tuple:
        """Return the latest camera image and metadata."""

        frame = self.provider.get_camera_frame()
        return frame.image, f"`{frame.timestamp}` | FPS: **{frame.fps:.1f}**"
