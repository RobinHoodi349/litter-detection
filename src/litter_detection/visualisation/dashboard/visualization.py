"""Assemble and launch the modular Gradio Robodog dashboard."""

from __future__ import annotations

import gradio as gr

from litter_detection.visualisation.dashboard.config import DashboardConfig
from litter_detection.visualisation.dashboard.data_provider import (
    DashboardDataProvider,
    QueueDashboardDataProvider,
    ZenohDashboardDataProvider,
)
from litter_detection.visualisation.dashboard.panels.camera import CameraPanel
from litter_detection.visualisation.dashboard.panels.control import ControlPanel
from litter_detection.visualisation.dashboard.panels.logs import LogsPanel
from litter_detection.visualisation.dashboard.panels.map import MapPanel
from litter_detection.visualisation.dashboard.panels.trash import TrashPanel


CSS = """
html,
body,
gradio-app,
.gradio-container {
    height: 100dvh !important;
    min-height: 100dvh !important;
    max-height: 100dvh !important;
    overflow: hidden !important;
    background: #0b1220;
    color: #e5e7eb;
}
body {
    margin: 0 !important;
}
footer,
.footer {
    display: none !important;
}
.gradio-container {
    padding: 0 !important;
    box-sizing: border-box;
    display: block !important;
    background: #0b1220 !important;
}
.gradio-container > *,
.gradio-container main,
.gradio-container .main {
    min-height: 0 !important;
    max-height: 100% !important;
    overflow: hidden !important;
    box-sizing: border-box;
}
.dashboard-root {
    position: fixed;
    top: 12px;
    right: 12px;
    bottom: 12px;
    left: 12px;
    width: auto;
    height: auto;
    max-height: none;
    margin: 0;
    padding: 10px;
    border: 2px solid #475569;
    border-radius: 12px;
    box-sizing: border-box;
    display: grid;
    grid-template-rows: minmax(0, 48fr) minmax(0, 52fr);
    gap: 8px;
    overflow: hidden;
    background: #0b1220;
    box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.18);
}
.dashboard-root,
.dashboard-root * {
    scrollbar-width: none;
    -ms-overflow-style: none;
}
.dashboard-root::-webkit-scrollbar,
.dashboard-root *::-webkit-scrollbar {
    display: none;
    width: 0;
    height: 0;
}
.top-row,
.bottom-row {
    gap: 12px;
    align-items: stretch;
    min-height: 0;
    overflow: hidden;
    width: 100%;
    flex-wrap: nowrap;
}
.dashboard-root .row,
.dashboard-root .column,
.dashboard-root .block,
.dashboard-root .form,
.dashboard-root .wrap,
.dashboard-root .contain {
    min-height: 0 !important;
    max-height: 100%;
    box-sizing: border-box;
}
.top-row {
    height: 100%;
    min-height: 0;
}
.bottom-row {
    height: 100%;
    min-height: 0;
}
.top-row > *,
.bottom-row > * {
    min-height: 0;
    height: 100%;
    overflow: hidden;
}
.dashboard-panel {
    border: 3px solid var(--panel-border);
    border-radius: 8px;
    padding: 8px;
    background: #111827;
    height: 100%;
    min-height: 0;
    box-sizing: border-box;
    overflow: hidden;
    color: #e5e7eb;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
}
.dashboard-root .block,
.dashboard-root .wrap,
.dashboard-root .form,
.dashboard-root .prose,
.dashboard-root .contain,
.dashboard-root .panel,
.dashboard-root input,
.dashboard-root textarea,
.dashboard-root select {
    background: #111827 !important;
    color: #e5e7eb !important;
    border-color: #334155 !important;
}
.dashboard-root button {
    color: #f8fafc !important;
}
.small-panel {
    min-height: 0;
    max-height: 100%;
    overflow: hidden;
}
.panel-logs.small-panel {
    display: grid;
    grid-template-rows: auto auto minmax(0, 1fr);
    overflow: hidden;
}
.panel-title h3 { margin: 0 0 5px 0; font-size: clamp(0.9rem, 1.35vh, 1.05rem); color: #f8fafc; }
.panel-meta { font-size: clamp(0.78rem, 1.08vh, 0.9rem); margin: 4px 0 0 0; color: #cbd5e1; }
.panel-meta code {
    background: #020617;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 2px 4px;
}
.panel-meta strong { color: #ffffff; }
.media-fill,
.media-fill > div,
.media-fill .image-container,
.media-fill [data-testid="image"],
.media-fill [data-testid="image"] > div {
    height: clamp(150px, calc(48dvh - 84px), 410px) !important;
    min-height: 0 !important;
    background: #020617 !important;
}
.media-fill {
    overflow: hidden !important;
}
.media-fill img {
    height: 100% !important;
    max-height: 100% !important;
    object-fit: cover;
}
.panel-camera { --panel-border: #a78bfa; }
.panel-map { --panel-border: #facc15; }
.panel-trash { --panel-border: #60a5fa; }
.panel-logs { --panel-border: #22d3ee; }
.panel-control { --panel-border: #c08457; }
.panel-control.small-panel {
    overflow: hidden;
    max-height: 100%;
}
.panel-control {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.panel-control .wrap,
.panel-control .block,
.panel-control .form,
.panel-control .prose {
    overflow: visible;
}
.scroll-panel {
    height: clamp(90px, calc(52dvh - 112px), 300px) !important;
    min-height: 0 !important;
    overflow-y: auto;
    scrollbar-width: thin;
    background: #020617 !important;
}
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
.log-scroll {
    scrollbar-width: none;
    scrollbar-color: #64748b #020617;
    -ms-overflow-style: none;
    overscroll-behavior: contain;
    pointer-events: auto;
}
.log-scroll::-webkit-scrollbar {
    display: none;
    width: 0;
}
.log-scroll::-webkit-scrollbar-button {
    display: none;
    height: 0;
    width: 0;
}
.log-scroll::-webkit-scrollbar-thumb {
    background: #64748b;
    border-radius: 999px;
}
.log-scroll::-webkit-scrollbar-track {
    background: #020617;
}
.compact-row {
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    width: 100%;
}
.filter-label {
    flex: 0 0 auto;
    min-width: 44px;
    margin: 0;
    color: #a5f3fc;
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
    color: #e5e7eb;
}
.level-filter,
.level-filter .wrap,
.level-filter .wrap > div,
.level-filter input,
.level-filter [data-testid="dropdown-input"] {
    background: #0f172a !important;
    color: #f8fafc !important;
    border-color: #155e75 !important;
    border-radius: 8px;
    height: 34px !important;
    min-height: 34px !important;
    max-height: 34px !important;
    box-shadow: none;
}
.level-filter {
    height: 34px !important;
    min-height: 34px !important;
    max-height: 34px !important;
    overflow: hidden !important;
}
.level-filter .wrap {
    height: 34px !important;
    min-height: 34px !important;
    max-height: 34px !important;
}
.level-filter input {
    font-weight: 700;
    padding-left: 10px;
    color: #f8fafc !important;
    line-height: 30px !important;
}
.level-filter button {
    color: #e5e7eb !important;
}
.log-count {
    flex: 0 0 auto;
    min-width: 112px;
    color: #e5e7eb;
    text-align: right;
    white-space: nowrap;
}
.log-frame {
    overflow: hidden !important;
    border: 1px solid #155e75;
    border-radius: 7px;
    background: #020617;
    padding: 8px;
    min-height: 0 !important;
    align-self: stretch;
    height: 100%;
    max-height: 100%;
    box-sizing: border-box;
}
.log-frame,
.log-frame > div,
.log-frame .wrap,
.log-frame .block,
.log-frame .prose {
    max-width: none;
    margin: 0;
    height: 100% !important;
    min-height: 0 !important;
    max-height: 100% !important;
    overflow: hidden !important;
    box-sizing: border-box;
}
.log-scroll {
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    box-sizing: border-box;
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    padding: 0 6px 0 0;
    background: #020617;
    scrollbar-width: none;
}
.log-scroll,
.log-scroll * {
    pointer-events: auto;
}
.log-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: clamp(0.62rem, 0.95vh, 0.72rem);
}
.log-entry {
    display: grid;
    grid-template-columns: auto 1fr auto;
    grid-template-areas:
        "level time source"
        "message message message";
    gap: 3px 8px;
    align-items: center;
    padding: 3px 8px;
    border: 1px solid #164e63;
    border-left: 4px solid #22b8c7;
    border-radius: 6px;
    background: #0f172a;
    min-height: 46px;
    max-height: 46px;
    box-sizing: border-box;
    overflow: hidden;
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
.log-level.info { background: #0369a1; }
.log-level.warn { background: #b45309; }
.log-level.error { background: #dc2626; }
.log-entry.info { border-left-color: #0ea5e9; }
.log-entry.warn { border-left-color: #f59e0b; }
.log-entry.error { border-left-color: #ef4444; }
.log-time { grid-area: time; color: #cbd5e1; white-space: nowrap; }
.log-source { grid-area: source; color: #67e8f9; font-weight: 700; white-space: nowrap; }
.log-message {
    grid-area: message;
    color: #f8fafc;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    line-height: 1.2;
}
.status-box {
    border: 1px solid #7c4a33;
    border-radius: 6px;
    padding: 5px 8px;
    background: #1f1714;
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
    gap: 2px 12px;
    color: #f8fafc;
    font-size: clamp(0.76rem, 1.08vh, 0.88rem);
}
.status-grid span {
    color: #fed7aa;
}
.command-output {
    min-height: 0;
    color: #fed7aa;
    font-size: clamp(0.7rem, 1vh, 0.82rem);
    overflow: hidden;
}
.control-button { width: 100%; }
.control-button button,
.panel-control button {
    min-height: clamp(26px, 3.75dvh, 36px) !important;
    color: #f8fafc !important;
    font-weight: 700;
    border: 1px solid #475569 !important;
    background: #374151 !important;
}
.panel-control button:hover {
    background: #4b5563 !important;
}
.control-stop button,
.panel-control button.stop {
    background: #b91c1c !important;
    border-color: #ef4444 !important;
}
.control-stop button:hover {
    background: #dc2626 !important;
}
.map-download,
.map-download .wrap,
.map-download .block {
    min-height: 30px !important;
    max-height: 40px !important;
    overflow: hidden !important;
    background: #1f1714 !important;
    border-color: #7c4a33 !important;
}
.manual-control {
    border: 1px solid #7c4a33;
    border-radius: 6px;
    padding: 5px;
    background: #1f1714;
    display: grid;
    gap: 4px;
}
.manual-speed,
.manual-speed .wrap,
.manual-speed .block {
    min-height: 36px !important;
    margin: 0 !important;
}
.manual-speed label {
    color: #fed7aa !important;
    font-size: clamp(0.68rem, 0.95vh, 0.78rem) !important;
}
.controller-row {
    gap: 5px;
    align-items: center;
    justify-content: center;
    min-height: 0;
}
.controller-button,
.controller-button button {
    min-width: 0 !important;
    width: 100% !important;
}
.controller-button button {
    min-height: clamp(24px, 3.3dvh, 32px) !important;
    padding: 0 !important;
    font-size: clamp(0.8rem, 1.25vh, 1.05rem) !important;
    line-height: 1 !important;
}
.controller-stop button {
    background: #991b1b !important;
    border-color: #ef4444 !important;
}
@media (max-height: 760px) {
    .dashboard-root {
        top: 8px;
        right: 8px;
        bottom: 8px;
        left: 8px;
        padding: 7px;
        grid-template-rows: minmax(0, 44fr) minmax(0, 56fr);
        gap: 6px;
    }
    .top-row, .bottom-row { gap: 8px; }
    .dashboard-panel { padding: 6px; border-width: 2px; }
    .compact-row { margin-bottom: 5px; }
    .log-entry { gap: 2px 6px; padding: 3px 6px; }
    .log-entry {
        min-height: 42px;
        max-height: 42px;
    }
    .log-frame {
        height: 100%;
        max-height: 100%;
        padding: 6px;
    }
    .media-fill,
    .media-fill > div,
    .media-fill .image-container,
    .media-fill [data-testid="image"],
    .media-fill [data-testid="image"] > div {
        height: clamp(130px, calc(44dvh - 68px), 310px) !important;
    }
    .scroll-panel {
        height: clamp(80px, calc(56dvh - 104px), 285px) !important;
    }
    .log-scroll {
        height: 100%;
        max-height: 100%;
    }
    .manual-control { padding: 4px; gap: 3px; }
    .controller-row { gap: 4px; }
    .controller-button button {
        min-height: clamp(22px, 3dvh, 28px) !important;
    }
}
@media (max-width: 900px) {
    .dashboard-root {
        top: 6px;
        right: 6px;
        bottom: 6px;
        left: 6px;
        gap: 8px;
        padding: 6px;
    }
    .top-row,
    .bottom-row {
        gap: 8px;
    }
    .level-filter {
        min-width: 140px;
    }
    .log-count {
        min-width: 86px;
    }
}
"""


def build_dashboard(
    provider: DashboardDataProvider | None = None,
    config: DashboardConfig | None = None,
) -> gr.Blocks:
    """Build the complete dashboard without launching it."""

    config = config or DashboardConfig()
    provider = provider or _build_provider(config)

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

        status_output = control_components.status
        command_output = control_components.command_output
        map_file = control_components.map_file
        manual_group = control_components.manual_group

        timer = gr.Timer(config.refresh_interval_s)
        timer.tick(camera_panel.update, outputs=camera_outputs)
        timer.tick(map_panel.update, outputs=map_outputs)
        timer.tick(trash_panel.update, outputs=trash_outputs)
        timer.tick(logs_panel.update, inputs=[log_level], outputs=[log_count, log_html])
        timer.tick(control_panel.status_update, outputs=[status_output, manual_group])

        log_level.change(logs_panel.update, inputs=[log_level], outputs=[log_count, log_html])
        for label, button in control_components.action_buttons.items():
            button.click(
                lambda label=label: control_panel.handle_button(label),
                outputs=[status_output, command_output, map_file, manual_group],
            )
        for direction, button in control_components.manual_buttons.items():
            button.click(
                lambda speed, direction=direction: control_panel.handle_manual_button(direction, speed),
                inputs=[control_components.speed],
                outputs=[status_output, command_output],
            )

        app.load(camera_panel.update, outputs=camera_outputs)
        app.load(map_panel.update, outputs=map_outputs)
        app.load(trash_panel.update, outputs=trash_outputs)
        app.load(logs_panel.update, inputs=[log_level], outputs=[log_count, log_html])
        app.load(control_panel.status_update, outputs=[status_output, manual_group])

    return app


def _build_provider(config: DashboardConfig) -> DashboardDataProvider:
    """Create the dashboard provider selected by configuration."""

    if config.provider.strip().lower() == "mock":
        return QueueDashboardDataProvider(config)
    return ZenohDashboardDataProvider(config)


def main() -> None:
    """Launch the Gradio dashboard."""

    config = DashboardConfig()
    app = build_dashboard(config=config)
    app.launch(server_name=config.host, server_port=config.port, share=config.share, css=CSS)


if __name__ == "__main__":
    main()
