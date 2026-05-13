"""Mapping panel for the Robodog dashboard."""

from __future__ import annotations

import gradio as gr

from litter_detection.visualisation.dashboard.data_provider import DashboardDataProvider
from litter_detection.visualisation.dashboard.panels.base import DashboardPanel, PanelTheme


class MapPanel(DashboardPanel):
    """Shows live map data and the current robot pose."""

    def __init__(self, provider: DashboardDataProvider) -> None:
        super().__init__("Mapping", PanelTheme("mapping", "panel-map", "#f4bd18"))
        self.provider = provider

    def render(self) -> list[gr.components.Component]:
        """Build the Gradio components for the panel."""

        with gr.Column(elem_classes=["dashboard-panel", self.theme.css_class]):
            self.render_header()
            image = gr.Image(label=None, show_label=False, type="numpy", height=360, elem_classes=["media-fill"])
            pose = gr.Markdown(elem_classes=["panel-meta"])
        return [image, pose]

    def update(self) -> tuple:
        """Return the latest map and robot pose summary."""

        frame = self.provider.get_map_frame()
        return frame.image, f"Position: **x={frame.x_m:.2f}m, y={frame.y_m:.2f}m** | Yaw: **{frame.yaw_deg:.0f} deg**"
