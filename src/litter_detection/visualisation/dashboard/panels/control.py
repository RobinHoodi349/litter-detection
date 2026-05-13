"""Robot control panel."""

from __future__ import annotations

import gradio as gr

from litter_detection.visualisation.dashboard.config import DashboardConfig
from litter_detection.visualisation.dashboard.data_provider import DashboardDataProvider
from litter_detection.visualisation.dashboard.panels.base import DashboardPanel, PanelTheme


class ControlPanel(DashboardPanel):
    """Configurable robot control panel."""

    def __init__(self, provider: DashboardDataProvider, config: DashboardConfig) -> None:
        super().__init__("Steuerung", PanelTheme("control", "panel-control", "#765341"))
        self.provider = provider
        self.config = config

    def render(self) -> list[gr.components.Component]:
        """Build the Gradio components for the panel."""

        buttons: list[gr.Button] = []
        with gr.Column(elem_classes=["dashboard-panel", self.theme.css_class, "small-panel"]):
            self.render_header()
            status = gr.HTML(value=self._status_html(), elem_classes=["status-box"])
            for label in self.config.control_buttons:
                variant = "stop" if label == "Stop" else "secondary"
                buttons.append(gr.Button(label, variant=variant, elem_classes=["control-button"]))
            output = gr.HTML(value="", elem_classes=["command-output"])
        return [status, output, *buttons]

    def status_update(self) -> tuple:
        """Return current robot mode, battery and connection state."""

        status = self.provider.get_status()
        connected = "verbunden" if status.connected else "getrennt"
        return (self._status_html(status.mode, status.battery_percent, connected),)

    def handle_button(self, label: str) -> tuple:
        """Handle one configured control button."""

        message = self.provider.handle_control(label)
        return self.status_update()[0], message

    def _status_html(
        self,
        mode: str | None = None,
        battery_percent: int | None = None,
        connected: str | None = None,
    ) -> str:
        """Render robot status as stable HTML instead of a Markdown block."""

        status = self.provider.get_status()
        mode = mode or status.mode
        battery_percent = battery_percent if battery_percent is not None else status.battery_percent
        connected = connected or ("verbunden" if status.connected else "getrennt")
        return (
            "<div class='status-grid'>"
            f"<span>Modus</span><strong>{mode}</strong>"
            f"<span>Batterie</span><strong>{battery_percent}%</strong>"
            f"<span>Verbindung</span><strong>{connected}</strong>"
            "</div>"
        )
