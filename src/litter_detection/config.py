'''Configuration for the Litter Detector application'''
from pathlib import Path
from dataclasses import dataclass
from litter_detection.training.train import EfficientNetB4UNet

REPO_ROOT = Path(__file__).resolve().parents[3]

@dataclass
class Settings:
# Model
    MODEL_NAME:str = "models/best_efficientnetb4.pth"
    MODEL_CLASS = EfficientNetB4UNet
    DROPOUT:float = 0.1
    FRAME_MAX_AGE_SECONDS:int = 1
    PROCESSING_TIMEOUT_SECONDS:int = 5
    THRESHOLD:float = 0.8
    LITTER_COVERAGE_THRESHOLD:float = 0.01  # min fraction of pixels to trigger reactor
    

    # Zenoh config
    ZENOH_ROUTER:str =  "tcp/localhost:7447"
    topic_frame:str = "robodog/sensors/go2_camera"
    topic_mask_binary:str = "litter/mask/binary"
    topic_mask_probabilities:str = "litter/mask/probabilities"
    topic_visualization:str = "litter/visualization"
    topic_alert:str = "litter/alert"
    topic_robodog_command:str = "litter/robodog/command"
    topic_movement_blocked:str = "litter/movement_blocked"
    topic_odometry:str = "robodog/system_state/odometry"
    topic_lidar:str = "robodog/sensors/go2_lidar"
    topic_movement_command:str = "robodog/command/motion/move"

    # Camera geometry (used by PathPlannerAgent to compute coverage footprint)
    CAMERA_HEIGHT_M: float = 0.5        # mounting height above ground [m]
    CAMERA_FOV_H_DEG: float = 90.0     # horizontal field of view [degrees]
    CAMERA_FOV_V_DEG: float = 60.0     # vertical field of view [degrees]
    CAMERA_COVERAGE_OVERLAP: float = 0.20  # min. overlap between adjacent footprints

    # Vision verifier model (must support image input, e.g. llava, moondream)
    VISION_MODEL_NAME: str = "llava:latest"
    USE_VERIFIER: bool = True
    # Seconds to ignore new detections after an alert (avoids re-triggering on same litter)
    ALERT_COOLDOWN_S: float = 10.0

    # OpenTelemetry setup
    SERVICE_NAME:str = "litter-detector"
    OTEL_ENDPOINT:str = "http://127.0.0.1:4317"
