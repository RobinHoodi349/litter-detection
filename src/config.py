'''Configuration for the Litter Detector application''' 
from train import *

# Model
MODEL_NAME = "best_efficientnetb4.pth"
MODEL_CLASS = EfficientNetB4UNet
DROPOUT = 0.1

# Zenoh config
ZENOH_ROUTER =  "tcp/localhost:7447"

# OpenTelemetry setup

SERVICE_NAME = "litter-detector"
OTEL_ENDPOINT = "http://127.0.0.1:4317"