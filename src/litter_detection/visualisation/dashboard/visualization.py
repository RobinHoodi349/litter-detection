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


# ---------------------------------------------------------------------------
# Visual design — "Robodog Control / SCADA Edition"
#
# Design language: modern industrial control system (SCADA/HMI). All panels
# share ONE neutral, uniform frame; the blue / purple / gold accents are
# reserved exclusively for status indicators, active states and live values.
# Layout-critical rules (fixed grid, clamp() heights, overflow hiding, media
# queries) are kept verbatim so the whole dashboard still fits one screen
# without scrolling — only appearance changed.
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root,
.dashboard-root {
    --bg-app: #070b16;
    --bg-panel: #0e1526;
    --bg-panel-2: #0a0f1d;
    --bg-inset: #060a14;
    --border: rgba(148, 163, 184, 0.14);
    --border-strong: rgba(148, 163, 184, 0.26);
    --text: #e6ebf5;
    --text-muted: #8b97ad;
    --text-head: #f8fafc;
    --blue: #4c8dff;
    --blue-soft: rgba(76, 141, 255, 0.16);
    --purple: #a78bfa;
    --purple-soft: rgba(167, 139, 250, 0.16);
    --gold: #f5b94a;
    --gold-soft: rgba(245, 185, 74, 0.16);
    --danger: #ef4444;
    --font-tech: 'Rajdhani', 'Segoe UI', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace;
}

html,
body,
gradio-app,
.gradio-container {
    height: 100dvh !important;
    min-height: 100dvh !important;
    max-height: 100dvh !important;
    overflow: hidden !important;
    background: var(--bg-app);
    color: var(--text);
    font-family: var(--font-tech);
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
    background: var(--bg-app) !important;
}
.gradio-container > *,
.gradio-container main,
.gradio-container .main {
    min-height: 0 !important;
    max-height: 100% !important;
    overflow: hidden !important;
    box-sizing: border-box;
}

/* ---- Root frame: uniform, calm, one continuous control surface ---- */
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
    padding: 12px;
    border: 1px solid var(--border-strong);
    border-radius: 16px;
    box-sizing: border-box;
    display: grid;
    grid-template-rows: auto minmax(0, 48fr) minmax(0, 52fr);
    gap: 10px;
    overflow: hidden;
    background:
        radial-gradient(1100px 680px at 8% 2%, rgba(76, 141, 255, 0.34), transparent 60%),
        radial-gradient(1050px 700px at 98% 100%, rgba(167, 139, 250, 0.32), transparent 62%),
        radial-gradient(820px 560px at 96% 4%, rgba(245, 185, 74, 0.16), transparent 60%),
        var(--bg-app);
    box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.06), 0 24px 60px rgba(0, 0, 0, 0.45);
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
/* ---- Brand title bar ---- */
.app-header-block {
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
    border: 0 !important;
}
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 1px 6px 9px 6px;
    border-bottom: 1px solid var(--border);
}
.app-header .brand {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}
.app-header .brand-mark {
    width: 13px;
    height: 13px;
    border-radius: 3px;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    box-shadow: 0 0 12px rgba(76, 141, 255, 0.7);
    transform: rotate(45deg);
    flex: 0 0 auto;
}
.app-header .brand-name {
    font-family: var(--font-tech);
    font-weight: 700;
    font-size: clamp(1rem, 2.2vh, 1.5rem);
    line-height: 1;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-head);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.app-header .brand-name .accent {
    color: var(--gold);
    text-shadow: 0 0 14px rgba(245, 185, 74, 0.45);
}
.app-header .app-sub {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: clamp(0.6rem, 0.95vh, 0.74rem);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    white-space: nowrap;
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

/* ---- Panels: every panel identical (maximum uniformity, no top accent) ---- */
.dashboard-panel {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 12px;
    background: linear-gradient(180deg, rgba(16, 23, 40, 0.80) 0%, rgba(10, 15, 29, 0.86) 100%);
    backdrop-filter: blur(2px);
    height: 100%;
    min-height: 0;
    box-sizing: border-box;
    overflow: hidden;
    color: var(--text);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 10px 28px rgba(0, 0, 0, 0.35);
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
    background: transparent !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
.dashboard-root button {
    color: var(--text-head) !important;
    font-family: var(--font-tech);
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

/* ---- Panel header: instrument-style label with a uniform status LED ---- */
.panel-title h3 {
    margin: 0 0 8px 0;
    padding-bottom: 7px;
    font-family: var(--font-tech);
    font-size: clamp(0.82rem, 1.3vh, 0.98rem);
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-head);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
}
.panel-title h3::before {
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: 9px;
    border-radius: 50%;
    background: var(--blue);
    box-shadow: 0 0 8px rgba(76, 141, 255, 0.9), 0 0 2px #fff inset;
    flex: 0 0 auto;
}
.panel-meta {
    font-family: var(--font-mono);
    font-size: clamp(0.74rem, 1.04vh, 0.86rem);
    margin: 6px 0 0 0;
    color: var(--text-muted);
    letter-spacing: 0.01em;
}
.panel-meta code {
    background: var(--blue-soft);
    color: #cfe0ff;
    border: 1px solid rgba(76, 141, 255, 0.35);
    border-radius: 5px;
    padding: 2px 6px;
}
.panel-meta strong { color: var(--gold); font-weight: 600; }

/* ---- Media surfaces (camera / map) ---- */
.media-fill,
.media-fill > div,
.media-fill .image-container,
.media-fill [data-testid="image"],
.media-fill [data-testid="image"] > div {
    height: clamp(140px, calc(48dvh - 112px), 400px) !important;
    min-height: 0 !important;
    background: var(--bg-inset) !important;
    border-radius: 8px;
}
.media-fill {
    overflow: hidden !important;
    border: 1px solid var(--border);
    border-radius: 8px;
}
.media-fill img {
    height: 100% !important;
    max-height: 100% !important;
    object-fit: cover;
}

/* All panels share the same neutral frame — accents live in status only. */
.panel-camera,
.panel-map,
.panel-trash,
.panel-logs,
.panel-control { --panel-border: var(--border); }
.panel-control.small-panel {
    overflow: hidden;
    max-height: 100%;
}
.panel-control {
    display: flex;
    flex-direction: column;
    gap: 7px;
}
.panel-control .wrap,
.panel-control .block,
.panel-control .form,
.panel-control .prose {
    overflow: visible;
}

/* ---- Scrollable detection gallery ---- */
.scroll-panel {
    height: clamp(90px, calc(52dvh - 116px), 300px) !important;
    min-height: 0 !important;
    overflow-y: auto;
    scrollbar-width: thin;
    background: var(--bg-inset) !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 4px;
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
.panel-trash .scroll-panel .thumbnail-item,
.panel-trash .scroll-panel .grid-wrap > div {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg-panel) !important;
}
.panel-trash .scroll-panel .caption,
.panel-trash .scroll-panel figcaption {
    font-family: var(--font-mono) !important;
    font-size: clamp(0.6rem, 0.92vh, 0.7rem) !important;
    color: #cfe0ff !important;
    background: rgba(6, 10, 20, 0.82) !important;
    letter-spacing: 0.01em;
}

/* ---- Logs ---- */
.log-scroll {
    scrollbar-width: none;
    scrollbar-color: var(--blue) var(--bg-inset);
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
    background: var(--blue);
    border-radius: 999px;
}
.log-scroll::-webkit-scrollbar-track {
    background: var(--bg-inset);
}
.compact-row {
    align-items: center;
    gap: 8px;
    margin-bottom: 7px;
    width: 100%;
}
.filter-label {
    flex: 0 0 auto;
    min-width: 44px;
    margin: 0;
    color: var(--text-muted);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: clamp(0.64rem, 0.95vh, 0.74rem);
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
    color: var(--text);
}
.level-filter,
.level-filter .wrap,
.level-filter .wrap > div,
.level-filter input,
.level-filter [data-testid="dropdown-input"] {
    background: var(--bg-inset) !important;
    color: var(--text-head) !important;
    border-color: rgba(76, 141, 255, 0.45) !important;
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
    font-family: var(--font-mono);
    font-weight: 600;
    padding-left: 10px;
    color: var(--text-head) !important;
    line-height: 30px !important;
}
.level-filter button {
    color: var(--text) !important;
}
.log-count {
    flex: 0 0 auto;
    min-width: 112px;
    color: var(--text-muted);
    text-align: right;
    white-space: nowrap;
    font-family: var(--font-mono);
}
.log-count strong { color: var(--gold); }
.log-frame {
    overflow: hidden !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-inset);
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
    background: var(--bg-inset);
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
    font-family: var(--font-mono);
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
    padding: 4px 9px;
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue);
    border-radius: 7px;
    background: var(--bg-panel);
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
    letter-spacing: 0.06em;
    text-align: center;
    min-width: 48px;
}
.log-level.info { background: var(--blue); color: #07101f; }
.log-level.warn { background: var(--gold); color: #2a1d00; }
.log-level.error { background: var(--danger); color: #fff; }
.log-entry.info { border-left-color: var(--blue); }
.log-entry.warn { border-left-color: var(--gold); }
.log-entry.error { border-left-color: var(--danger); }
.log-time { grid-area: time; color: var(--text-muted); white-space: nowrap; }
.log-source { grid-area: source; color: var(--purple); font-weight: 700; white-space: nowrap; }
.log-message {
    grid-area: message;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    line-height: 1.25;
}

/* ---- Control: status as SCADA stat tiles with live accent indicators ---- */
.status-box {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 9px 11px;
    background: var(--bg-inset);
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
    align-items: center;
    gap: 7px 14px;
    color: var(--text);
    font-family: var(--font-tech);
    font-size: clamp(0.78rem, 1.1vh, 0.9rem);
}
.stat-label {
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: clamp(0.64rem, 0.92vh, 0.74rem);
    font-weight: 600;
}
.stat-cell {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
}
.mode-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    background: var(--purple-soft);
    border: 1px solid rgba(167, 139, 250, 0.5);
    color: #ddd2ff;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: clamp(0.68rem, 0.98vh, 0.78rem);
}
.batt-track {
    flex: 1 1 auto;
    height: 8px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.16);
    overflow: hidden;
    border: 1px solid var(--border);
}
.batt-fill {
    display: block;
    height: 100%;
    border-radius: 999px;
}
.batt-fill.ok { background: linear-gradient(90deg, var(--blue), #7fb0ff); box-shadow: 0 0 8px rgba(76, 141, 255, 0.6); }
.batt-fill.mid { background: linear-gradient(90deg, var(--gold), #ffd27a); box-shadow: 0 0 8px rgba(245, 185, 74, 0.6); }
.batt-fill.low { background: linear-gradient(90deg, var(--danger), #ff7b7b); box-shadow: 0 0 8px rgba(239, 68, 68, 0.6); }
.batt-pct {
    flex: 0 0 auto;
    font-family: var(--font-mono);
    font-weight: 600;
    color: var(--gold);
    min-width: 38px;
    text-align: right;
}
.led {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex: 0 0 auto;
}
.led.ok { background: var(--blue); box-shadow: 0 0 8px rgba(76, 141, 255, 0.95); }
.led.alert { background: var(--gold); box-shadow: 0 0 9px rgba(245, 185, 74, 0.95); animation: led-pulse 1.1s ease-in-out infinite; }
@keyframes led-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
.conn-text { font-weight: 600; letter-spacing: 0.04em; }
.stat-cell .conn-text { color: var(--text-head); }

.command-output {
    min-height: 0;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: clamp(0.7rem, 1vh, 0.82rem);
    overflow: hidden;
}
.command-output strong,
.command-output b { color: var(--gold); }

/* ---- Buttons: dark glass, blue glow on hover; Start primary, Stop safety ---- */
.control-button { width: 100%; }
.control-button button,
.panel-control button {
    min-height: clamp(26px, 3.75dvh, 36px) !important;
    color: var(--text-head) !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    border: 1px solid var(--border-strong) !important;
    border-radius: 9px !important;
    background: linear-gradient(180deg, #161f33, #0e1626) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.panel-control button:hover {
    background: linear-gradient(180deg, #1b2742, #111b30) !important;
    border-color: rgba(76, 141, 255, 0.6) !important;
    box-shadow: 0 0 0 1px rgba(76, 141, 255, 0.25), 0 0 14px rgba(76, 141, 255, 0.28) !important;
}
.control-start button {
    border-color: rgba(76, 141, 255, 0.55) !important;
    color: #cfe0ff !important;
    background: linear-gradient(180deg, rgba(76, 141, 255, 0.22), rgba(76, 141, 255, 0.08)) !important;
}
.control-start button:hover {
    box-shadow: 0 0 0 1px rgba(76, 141, 255, 0.45), 0 0 16px rgba(76, 141, 255, 0.45) !important;
}
.control-stop button,
.panel-control button.stop {
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.26), rgba(239, 68, 68, 0.1)) !important;
    border-color: rgba(239, 68, 68, 0.65) !important;
    color: #ffe1e1 !important;
}
.control-stop button:hover {
    box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.5), 0 0 16px rgba(239, 68, 68, 0.45) !important;
}
.map-download,
.map-download .wrap,
.map-download .block {
    min-height: 30px !important;
    max-height: 40px !important;
    overflow: hidden !important;
    background: var(--bg-inset) !important;
    border-color: rgba(245, 185, 74, 0.4) !important;
    border-radius: 8px !important;
}

/* ---- Manual joypad: purple frame signals the active manual mode ---- */
.manual-control {
    border: 1px solid rgba(167, 139, 250, 0.45);
    border-radius: 10px;
    padding: 7px;
    background: linear-gradient(180deg, rgba(167, 139, 250, 0.08), rgba(167, 139, 250, 0.02));
    box-shadow: 0 0 14px rgba(167, 139, 250, 0.12);
    display: grid;
    gap: 5px;
}
.manual-speed,
.manual-speed .wrap,
.manual-speed .block {
    min-height: 36px !important;
    margin: 0 !important;
}
.manual-speed label {
    color: #ddd2ff !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: clamp(0.66rem, 0.92vh, 0.76rem) !important;
}
.manual-speed input[type="range"] { accent-color: var(--purple); }
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
    border-color: rgba(167, 139, 250, 0.4) !important;
}
.controller-button button:hover {
    border-color: rgba(167, 139, 250, 0.75) !important;
    box-shadow: 0 0 0 1px rgba(167, 139, 250, 0.3), 0 0 12px rgba(167, 139, 250, 0.3) !important;
}
.controller-stop button {
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.26), rgba(239, 68, 68, 0.1)) !important;
    border-color: rgba(239, 68, 68, 0.65) !important;
}

@media (max-height: 760px) {
    .dashboard-root {
        top: 8px;
        right: 8px;
        bottom: 8px;
        left: 8px;
        padding: 9px;
        grid-template-rows: auto minmax(0, 44fr) minmax(0, 56fr);
        gap: 8px;
    }
    .top-row, .bottom-row { gap: 8px; }
    .dashboard-panel { padding: 8px 9px; }
    .compact-row { margin-bottom: 5px; }
    .log-entry { gap: 2px 6px; padding: 3px 7px; }
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
        height: clamp(120px, calc(44dvh - 96px), 300px) !important;
    }
    .scroll-panel {
        height: clamp(80px, calc(56dvh - 108px), 285px) !important;
    }
    .log-scroll {
        height: 100%;
        max-height: 100%;
    }
    .manual-control { padding: 5px; gap: 4px; }
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
        padding: 7px;
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
            gr.HTML(
                "<div class='app-header'>"
                "<div class='brand'>"
                "<span class='brand-mark'></span>"
                "<span class='brand-name'>The<span class='accent'>BlackBox</span>Society</span>"
                "</div>"
                "<div class='app-sub'><span class='led ok'></span>Robodog · Litter Detection Control</div>"
                "</div>",
                elem_classes=["app-header-block"],
            )
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
