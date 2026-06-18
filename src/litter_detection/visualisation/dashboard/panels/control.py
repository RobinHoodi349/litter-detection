"""Robot control panel."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from litter_detection.visualisation.dashboard.config import DashboardConfig
from litter_detection.visualisation.dashboard.data_provider import DashboardDataProvider
from litter_detection.visualisation.dashboard.panels.base import DashboardPanel, PanelTheme


@dataclass(frozen=True)
class ControlPanelComponents:
    """Rendered Gradio components needed for wiring callbacks."""

    status: gr.HTML
    command_output: gr.HTML
    map_file: gr.File
    manual_group: gr.Group
    action_buttons: dict[str, gr.Button]
    speed: gr.Slider
    manual_buttons: dict[str, gr.Button]


class ControlPanel(DashboardPanel):
    """Configurable robot control panel."""

    def __init__(self, provider: DashboardDataProvider, config: DashboardConfig) -> None:
        super().__init__("Steuerung", PanelTheme("control", "panel-control", "#765341"))
        self.provider = provider
        self.config = config

    def render(self) -> ControlPanelComponents:
        """Build the Gradio components for the panel."""

        buttons: dict[str, gr.Button] = {}
        manual_buttons: dict[str, gr.Button] = {}
        with gr.Column(elem_classes=["dashboard-panel", self.theme.css_class, "small-panel"]):
            self.render_header()
            status = gr.HTML(value=self._status_html(), elem_classes=["status-box"])
            for label in self.config.control_buttons:
                variant = "stop" if label == "Stop" else "secondary"
                classes = ["control-button"]
                if label == "Stop":
                    classes.append("control-stop")
                elif label == "Start":
                    classes.append("control-start")
                else:
                    classes.append("control-secondary")
                buttons[label] = gr.Button(label, variant=variant, elem_classes=classes)
            map_file = gr.File(label="Karten-Download", visible=False, interactive=False, elem_classes=["map-download"])
            with gr.Group(visible=self._manual_visible(), elem_classes=["manual-control"]) as manual_group:
                speed = gr.Slider(
                    minimum=0.05,
                    maximum=0.8,
                    value=0.2,
                    step=0.05,
                    label="Tempo",
                    elem_classes=["manual-speed"],
                )
                with gr.Row(elem_classes=["controller-row"]):
                    manual_buttons["turn_left"] = gr.Button("↶", elem_classes=["controller-button"])
                    manual_buttons["forward"] = gr.Button("↑", elem_classes=["controller-button"])
                    manual_buttons["turn_right"] = gr.Button("↷", elem_classes=["controller-button"])
                with gr.Row(elem_classes=["controller-row"]):
                    manual_buttons["left"] = gr.Button("←", elem_classes=["controller-button"])
                    manual_buttons["stop"] = gr.Button("■", variant="stop", elem_classes=["controller-button", "controller-stop"])
                    manual_buttons["right"] = gr.Button("→", elem_classes=["controller-button"])
                with gr.Row(elem_classes=["controller-row"]):
                    manual_buttons["backward"] = gr.Button("↓", elem_classes=["controller-button"])
            output = gr.HTML(value="", elem_classes=["command-output"])
        return ControlPanelComponents(
            status=status,
            command_output=output,
            map_file=map_file,
            manual_group=manual_group,
            action_buttons=buttons,
            speed=speed,
            manual_buttons=manual_buttons,
        )

    def status_update(self) -> tuple:
        """Return current robot mode, battery and connection state."""

        status = self.provider.get_status()
        connected = "verbunden" if status.connected else "getrennt"
        return (
            self._status_html(status.mode, status.battery_percent, connected),
            gr.update(visible=status.mode == "MANUAL"),
        )

    def handle_button(self, label: str) -> tuple:
        """Handle one configured control button."""

        message = self.provider.handle_control(label)
        map_file_update = gr.update(visible=False)
        if self._normalize_action(label) == "save_map":
            path = self.provider.save_current_map()
            message = f"{message}<br>Karte gespeichert: {path}"
            map_file_update = gr.update(value=path, visible=True)
        status_html, manual_update = self.status_update()
        return status_html, message, map_file_update, manual_update

    def handle_manual_button(self, direction: str, speed: float) -> tuple:
        """Handle one manual controller button."""

        message = self.provider.handle_manual_control(direction, speed)
        status_html, _ = self.status_update()
        return status_html, message

    def _manual_visible(self) -> bool:
        return self.provider.get_status().mode == "MANUAL"

    @staticmethod
    def _normalize_action(action: str) -> str:
        return {
            "Stop": "stop",
            "Start": "start_autonomous",
            "Zurück zum Start": "return_home",
            "Zurueck zum Start": "return_home",
            "Manuelle Übernahme": "manual_override",
            "Manuelle Uebernahme": "manual_override",
            "Karte speichern": "save_map",
        }.get(action, action.strip().lower().replace(" ", "_"))

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

        is_connected = connected.strip().lower().startswith("verbunden")
        conn_state = "ok" if is_connected else "alert"
        batt_pct = max(0, min(100, int(battery_percent)))
        batt_state = "low" if batt_pct <= 20 else "mid" if batt_pct <= 50 else "ok"

        return (
            "<div class='status-grid'>"
            "<span class='stat-label'>Modus</span>"
            f"<span class='stat-cell'><span class='mode-pill'>{mode}</span></span>"
            "<span class='stat-label'>Batterie</span>"
            "<span class='stat-cell'>"
            "<span class='batt-track'>"
            f"<span class='batt-fill {batt_state}' style='width:{batt_pct}%'></span>"
            "</span>"
            f"<span class='batt-pct'>{batt_pct}%</span>"
            "</span>"
            "<span class='stat-label'>Verbindung</span>"
            f"<span class='stat-cell'><span class='led {conn_state}'></span>"
            f"<span class='conn-text'>{connected}</span></span>"
            "</div>"
        )
