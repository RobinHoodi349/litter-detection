'''Configuration for the Litter Detector application''' 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import train
MODEL_NAME = "best_efficientnetb4.pth"
MODEL_CLASS = train.EfficientNetB4UNet
DROPOUT = 0.1

# Zenoh config
ZENOH_ROUTER =  "tcp/localhost:7447"

# OpenTelemetry setup

SERVICE_NAME = "litter-detector"
OTEL_ENDPOINT = "http://127.0.0.1:4317"