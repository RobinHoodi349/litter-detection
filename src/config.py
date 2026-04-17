'''Configuration for the Litter Detector application''' 
import sys
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from train import *
from dataclasses import dataclass
@dataclass
class Settings:
# Model
    MODEL_NAME:str = "best_efficientnetb4.pth"
    MODEL_CLASS = EfficientNetB4UNet
    DROPOUT:float = 0.1
    FRAME_MAX_AGE_SECONDS:int = 1
    PROCESSING_TIMEOUT_SECONDS:int = 5

    # Zenoh config
    ZENOH_ROUTER:str =  "tcp/localhost:7447"
    topic_frame:str = "litter/frame"
    topic_mask_binary:str = "litter/mask/binary"
    topic_mask_probabilities:str = "litter/mask/probabilities"
    topic_visualization:str = "litter/visualization"

    # OpenTelemetry setup

    SERVICE_NAME:str = "litter-detector"
    OTEL_ENDPOINT:str = "http://127.0.0.1:4317"