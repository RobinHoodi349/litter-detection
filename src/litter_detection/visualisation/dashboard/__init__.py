"""Gradio dashboard for the Robodog litter-detection system."""

from litter_detection.visualisation.dashboard.visualization import build_dashboard
from litter_detection.visualisation.dashboard.data_provider import ZenohDashboardDataProvider

__all__ = ["build_dashboard", "ZenohDashboardDataProvider"]
