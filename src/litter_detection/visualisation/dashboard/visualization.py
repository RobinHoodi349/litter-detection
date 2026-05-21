"""Assemble and launch the modular Gradio Robodog dashboard."""

from __future__ import annotations

import gradio as gr

from litter_detection.visualisation.dashboard.config import DashboardConfig
from litter_detection.visualisation.dashboard.data_provider import DashboardDataProvider, QueueDashboardDataProvider
from litter_detection.visualisation.dashboard.panels.camera import CameraPanel
from litter_detection.visualisation.dashboard.panels.control import ControlPanel
from litter_detection.visualisation.dashboard.panels.logs import LogsPanel
from litter_detection.visualisation.dashboard.panels.map import MapPanel
from litter_detection.visualisation.dashboard.panels.trash import TrashPanel


CSS = """
body, gradio-app { background: #ffffff; }
.dashboard-root { max-width: 1500px; margin: 0 auto; }
.top-row, .bottom-row { gap: 18px; align-items: stretch; }
.dashboard-panel {
    border: 3px solid var(--panel-border);
    border-radius: 8px;
    padding: 12px;
    background: #ffffff;
    min-height: 100%;
    box-sizing: border-box;
}
.small-panel {
    min-height: 390px;
    max-height: none;
    overflow: hidden;
}
.panel-logs.small-panel {
    display: flex;
    flex-direction: column;
    overflow: visible;
}
.panel-title h3 { margin: 0 0 8px 0; font-size: 1.1rem; color: #222; }
.panel-meta { font-size: 0.92rem; margin: 4px 0 0 0; }
.media-fill img { object-fit: cover; }
.panel-camera { --panel-border: #9b8cf0; }
.panel-map { --panel-border: #f4bd18; }
.panel-trash { --panel-border: #0c3aa6; }
.panel-logs { --panel-border: #22b8c7; }
.panel-control { --panel-border: #765341; }
.panel-control.small-panel {
    overflow: visible;
    max-height: none;
}
.panel-control {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.panel-control .wrap,
.panel-control .block,
.panel-control .form,
.panel-control .prose {
    overflow: visible;
}
.scroll-panel { overflow-y: auto; scrollbar-width: thin; }
.panel-trash .scroll-panel,
.panel-trash .scroll-panel * {
    scrollbar-width: none;
    -ms-overflow-style: none;
}
.panel-trash .scroll-panel::-webkit-scrollbar,
.panel-trash .scroll-panel *::-webkit-scrollbar {
    display: none;
    width: 0;
    height: 0;
}
.scroll-panel::-webkit-scrollbar,
.log-scroll::-webkit-scrollbar { width: 8px; }
.scroll-panel::-webkit-scrollbar-button,
.log-scroll::-webkit-scrollbar-button { display: none; height: 0; width: 0; }
.scroll-panel::-webkit-scrollbar-thumb,
.log-scroll::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 999px; }
.scroll-panel::-webkit-scrollbar-track,
.log-scroll::-webkit-scrollbar-track { background: transparent; }
.compact-row {
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    width: 100%;
}
.filter-label {
    flex: 0 0 auto;
    min-width: 44px;
    margin: 0;
    color: #15535e;
    font-weight: 700;
}
.level-filter {
    flex: 1 1 auto;
    min-width: 220px;
    max-width: none;
    padding: 0;
    background: transparent;
    border: 0;
}
.level-filter label,
.level-filter label span {
    color: #1f2933;
}
.level-filter,
.level-filter .wrap,
.level-filter .wrap > div,
.level-filter input,
.level-filter [data-testid="dropdown-input"] {
    background: #ffffff;
    color: #15535e;
    border-color: #a7dde5;
    border-radius: 10px;
    min-height: 42px;
    box-shadow: 0 2px 8px rgba(15, 63, 74, 0.08);
}
.level-filter input {
    font-weight: 700;
    padding-left: 14px;
}
.level-filter button {
    color: #15535e;
}
.log-count {
    flex: 0 0 auto;
    min-width: 112px;
    color: #334155;
    text-align: right;
    white-space: nowrap;
}
.log-frame {
    overflow: hidden;
    border: 1px solid #c7edf1;
    border-radius: 7px;
    background: #f7fcfd;
    padding: 0;
}
.log-frame > .prose {
    max-width: none;
    margin: 0;
}
.log-scroll {
    height: 240px;
    max-height: 240px;
    box-sizing: border-box;
    overflow-y: auto;
    padding: 8px;
    background: #f7fcfd;
    scrollbar-width: thin;
}
.log-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.78rem;
}
.log-entry {
    display: grid;
    grid-template-columns: auto 1fr auto;
    grid-template-areas:
        "level time source"
        "message message message";
    gap: 4px 9px;
    align-items: center;
    padding: 7px 8px;
    border: 1px solid #e2f3f5;
    border-left: 4px solid #22b8c7;
    border-radius: 6px;
    background: #ffffff;
}
.log-level {
    grid-area: level;
    color: #fff;
    border-radius: 4px;
    padding: 2px 7px;
    font-weight: 700;
    text-align: center;
    min-width: 48px;
}
.log-level.info { background: #2374ab; }
.log-level.warn { background: #c88719; }
.log-level.error { background: #b42318; }
.log-entry.info { border-left-color: #2374ab; }
.log-entry.warn { border-left-color: #c88719; }
.log-entry.error { border-left-color: #b42318; }
.log-time { grid-area: time; color: #64748b; white-space: nowrap; }
.log-source { grid-area: source; color: #0f3f4a; font-weight: 700; white-space: nowrap; }
.log-message { grid-area: message; color: #1f2933; overflow-wrap: anywhere; line-height: 1.35; }
.status-box {
    border: 1px solid #e2d7d1;
    border-radius: 6px;
    padding: 8px 10px;
    background: #fffaf7;
    overflow: visible;
}
.status-box,
.status-box * {
    scrollbar-width: none;
    -ms-overflow-style: none;
}
.status-box::-webkit-scrollbar,
.status-box *::-webkit-scrollbar {
    display: none;
    width: 0;
    height: 0;
}
.status-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    color: #3b2a20;
    font-size: 0.9rem;
}
.status-grid span {
    color: #765341;
}
.command-output {
    min-height: 0;
    color: #765341;
    font-size: 0.85rem;
}
.control-button { width: 100%; }
"""


def build_dashboard(
    provider: DashboardDataProvider | None = None,
    config: DashboardConfig | None = None,
) -> gr.Blocks:
    """Build the complete dashboard without launching it."""

    config = config or DashboardConfig()
    provider = provider or QueueDashboardDataProvider(config)

    camera_panel = CameraPanel(provider)
    map_panel = MapPanel(provider)
    trash_panel = TrashPanel(provider)
    logs_panel = LogsPanel(provider)
    control_panel = ControlPanel(provider, config)

    with gr.Blocks(title="Robodog Litter Detection Dashboard") as app:
        with gr.Column(elem_classes=["dashboard-root"]):
            with gr.Row(elem_classes=["top-row"], equal_height=True):
                with gr.Column(scale=1):
                    camera_outputs = camera_panel.render()
                with gr.Column(scale=1):
                    map_outputs = map_panel.render()

            with gr.Row(elem_classes=["bottom-row"], equal_height=True):
                with gr.Column(scale=1):
                    trash_outputs = trash_panel.render()
                with gr.Column(scale=1):
                    log_level, log_count, log_html = logs_panel.render()
                with gr.Column(scale=1):
                    control_components = control_panel.render()

        status_output = control_components[0]
        command_output = control_components[1]
        control_buttons = control_components[2:]

        timer = gr.Timer(config.refresh_interval_s)
        timer.tick(camera_panel.update, outputs=camera_outputs)
        timer.tick(map_panel.update, outputs=map_outputs)
        timer.tick(trash_panel.update, outputs=trash_outputs)
        timer.tick(logs_panel.update, inputs=[log_level], outputs=[log_count, log_html])
        timer.tick(control_panel.status_update, outputs=[status_output])

        log_level.change(logs_panel.update, inputs=[log_level], outputs=[log_count, log_html])
        for button in control_buttons:
            button.click(
                lambda label=button.value: control_panel.handle_button(label),
                outputs=[status_output, command_output],
            )

        app.load(camera_panel.update, outputs=camera_outputs)
        app.load(map_panel.update, outputs=map_outputs)
        app.load(trash_panel.update, outputs=trash_outputs)
        app.load(logs_panel.update, inputs=[log_level], outputs=[log_count, log_html])
        app.load(control_panel.status_update, outputs=[status_output])

    return app


def main() -> None:
    """Launch the Gradio dashboard."""

    config = DashboardConfig()
    app = build_dashboard(config=config)
    app.launch(server_name=config.host, server_port=config.port, share=config.share, css=CSS)


if __name__ == "__main__":
    main()
