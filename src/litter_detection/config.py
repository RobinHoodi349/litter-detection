'''Configuration for the Litter Detector application'''
import os
from pathlib import Path
from dataclasses import dataclass
from litter_detection.training.train import EfficientNetB1UNet

REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
# Model
    MODEL_NAME:str = "models/checkpoints/best_model_efficientnetb1_model.pth"
    MODEL_CLASS = EfficientNetB1UNet
    DROPOUT:float = 0.1
    FRAME_MAX_AGE_SECONDS:int = 1
    PROCESSING_TIMEOUT_SECONDS:int = 5
    THRESHOLD:float = 0.8
    LITTER_COVERAGE_THRESHOLD:float = 0.0  # min fraction of pixels to trigger reactor


    # Zenoh config — override ZENOH_ROUTER so the same code runs on the Jetson
    # (tcp/localhost:7447) and points at the robot/router from a remote machine.
    ZENOH_ROUTER:str =  os.getenv("ZENOH_ROUTER", "tcp/localhost:7447")
    topic_frame:str = "robodog/sensors/go2_camera"
    topic_mask_binary:str = "litter/mask/binary"
    topic_mask_probabilities:str = "litter/mask/probabilities"
    topic_visualization:str = "litter/visualization"
    topic_alert:str = "litter/alert"
    topic_obstacle:str = "litter/obstacles"
    topic_occupancy_grid:str = "litter/occupancy_grid"
    topic_robodog_command:str = "litter/robodog/command"
    topic_movement_blocked:str = "litter/movement_blocked"
    topic_odometry:str = "robodog/system_state/odometry"
    topic_lidar:str = "robodog/sensors/go2_lidar"
    topic_movement_command:str = "robodog/command/motion/move"
    topic_estop:str = "robodog/command/motion/estop"

    # Camera geometry (used by PathPlannerAgent to compute coverage footprint)
    CAMERA_HEIGHT_M: float = 0.5        # mounting height above ground [m]
    CAMERA_FOV_H_DEG: float = 90.0     # horizontal field of view [degrees]
    CAMERA_FOV_V_DEG: float = 60.0     # vertical field of view [degrees]
    CAMERA_RANGE_M: float = 2.0        # how far ahead the camera sees on the ground [m]
    CAMERA_COVERAGE_OVERLAP: float = 0.20  # min. overlap between adjacent footprints

    # Vision verifier model (must support image input + structured output).
    # qwen2.5vl handles both reliably; the 3b variant keeps latency low on the
    # Jetson (~2-3 GB) and is plenty for the binary litter yes/no decision.
    # Bump to qwen2.5vl:7b for more accuracy, or moondream:latest on tiny hosts.
    VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "qwen2.5vl:3b")
    USE_VERIFIER: bool = _env_bool("USE_VERIFIER", True)
    # Seconds to ignore new detections after an alert (avoids re-triggering on same litter)
    ALERT_COOLDOWN_S: float = 10.0
    # Max seconds to wait for Ollama response before giving up
    VERIFIER_TIMEOUT_S: float = 15.0
    # Min seconds between consecutive inference calls (rate-limits CNN load on laptop)
    INFERENCE_MIN_INTERVAL_S: float = 1.0

    # OpenTelemetry setup
    SERVICE_NAME:str = "litter-detector"
    OTEL_ENDPOINT:str = os.getenv("OTEL_ENDPOINT", "http://127.0.0.1:4317")
