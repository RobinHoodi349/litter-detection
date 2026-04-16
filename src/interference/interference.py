import config
from train import *

import zenoh
import torch
import numpy as np
from collections import deque
from pathlib import Path
import threading
import time
import cv2
import json
import logging

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource

# Opentelemetry setup

resource = Resource.create({"service.name": config.SERVICE_NAME})
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=config.OTEL_ENDPOINT, insecure=True)))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(config.SERVICE_NAME)

metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=config.OTEL_ENDPOINT, insecure=True),export_interval_millis=5000)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(config.SERVICE_NAME)

# Metrics

inference_latency = meter.create_histogram("inference_duration_seconds", description="inference latency", unit="s")
confidence_hist = meter.create_histogram("detection_confidence", description="YOLO confidence scores", unit="1")
frames_processed = meter.create_counter("frames_processed_total", description="Total frames processed")


def load_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    logging.info(f"Loading model: {config.MODEL_NAME} on device: {device}")
    CHECKPOINT = "../../" + config.MODEL_NAME
    
    p = Path(CHECKPOINT)
    if p.exists():
        model = config.MODEL_CLASS(dropout=config.DROPOUT).to(device)
        model.load_state_dict(torch.load(p, map_location=device))
        model.eval()
        n = sum(par.numel() for par in model.parameters())
        logging.info(f"Loaded {CHECKPOINT} ({config.MODEL_CLASS.__name__}) on {device}  ({n:,} params)")
    else:
        logging.error(f"{CHECKPOINT} not found")

frame_queue = deque(maxlen=20)
frame_queue_lock = threading.Lock()
frame_available = threading.Event()

def on_frame_received(sample: zenoh.Sample):
    try:
        frame_bytes = bytes(sample.payload) 
        with tracer.start_as_current_span("receive_frame"):
            with frame_queue_lock:
                frame_queue.append(frame_bytes)
                frame_available.set()
    except Exception as e:
        logging.exception(f"Failed to enqueue frame from Zenoh: {e}")

def preprocess_frame(frame_bytes):

    img_rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor.to(device)

def inference(frame_bytes: bytes):
   
        with tracer.start_as_current_span("inference")as span:
            span.set_attribute("frame_size_bytes", len(frame_bytes))
            span.set_attribute("model_name", config.MODEL_NAME)
            start_time = time.perf_counter()

            
            try:

                img_tensor = preprocess_frame(frame_bytes)
                with torch.no_grad():
                    output = model(img_tensor)
                
                duration = time.perf_counter() - start_time

                if isinstance(output, torch.Tensor):
                    # Für binäre Segmentierung: Sigmoid anwenden, um Wahrscheinlichkeiten zu bekommen
                    mask_probabilities = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
                    # Optional: Threshold anwenden, um binäre Maske zu bekommen (z.B. > 0.5)
                    mask_binary = (mask_probabilities > 0.5).astype(np.uint8)
                else:
                    mask_probabilities = None
                    mask_binary = None
                
                span.set_attribute("inference_duration_seconds", round(duration*1000,1))
                inference_latency.record(duration,{"model_name": config.MODEL_NAME})
                frames_processed.add(1)
                
                logging.info(f"Maske erhalten: Shape {mask_probabilities.shape}, Binary Mask Shape {mask_binary.shape}")
                

            except Exception as e:
                logging.exception(f"Error during inference: {e}")
            
            return mask_probabilities, mask_binary

def visualize_mask(frame_bytes: bytes, mask_binary: np.ndarray, alpha: float = 0.5):
 
    try:
        # Dekodiere Frame zu OpenCV-Format (BGR)
        frame_bgr = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize Maske auf Frame-Größe falls nötig
        frame_height, frame_width = frame_rgb.shape[:2]
        mask_height, mask_width = mask_binary.shape[:2]
        
        if (mask_height, mask_width) != (frame_height, frame_width):
            mask_binary = cv2.resize(mask_binary, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        
        # Erstelle rotes Overlay für erkannte Litter-Bereiche
        overlay = frame_rgb.copy()
        # Rot im RGB-Format
        overlay[mask_binary == 1] = [255, 0, 0]
        
        # Blende Overlay mit Original zusammen
        visualized = cv2.addWeighted(frame_rgb, 1 - alpha, overlay, alpha, 0)
        
        logging.debug(f"Visualisierung erstellt: {visualized.shape}")
        return visualized
        
    except Exception as e:
        logging.exception(f"Error during mask visualization: {e}")
        return None

def main():
    load_model()


    z = zenoh.open(zenoh.Config().insert_json5("connect/endpoints", json.dumps([config.ZENOH_ROUTER])))
    frame_sub = z.declare_subscriber("litter/frame", on_frame_received)
    logging.info("Subscribed to Zenoh topic: litter/frame")
    
    try:
        while True:
            if not frame_available.wait(timeout=1):
                continue
            with frame_queue_lock:
                if not frame_queue:
                    frame_available.clear()
                    continue
                Frame_bytes = frame_queue.popleft()
                if not frame_queue:
                    frame_available.clear()
            
            with tracer.start_as_current_span("process_frame") as root_span:
                overall_start = time.perf_counter()

                mask_probabilities, mask_binary = inference(Frame_bytes)
                
                # Visualisiere die Maske
                visualized_img = visualize_mask(Frame_bytes, mask_binary, alpha=0.5) if mask_binary is not None else None
                
                # Sende Ergebnisse über Zenoh
                if mask_probabilities is not None:
                    z.put("litter/mask_probabilities", mask_probabilities.tobytes())
                    logging.debug(f"Sent probability mask: {mask_probabilities.shape}")
                
                if mask_binary is not None:
                    z.put("litter/mask_binary", mask_binary.tobytes())
                    logging.debug(f"Sent binary mask: {mask_binary.shape}")
                
                if visualized_img is not None:
                    # Konvertiere RGB zu BGR für JPEG Encoding
                    visualized_bgr = cv2.cvtColor(visualized_img, cv2.COLOR_RGB2BGR)
                    encoded = zenoh.Encoding.IMAGE_JPEG.ZENOH_BYTES(visualized_bgr)
                    z.put("litter/visualization", encoded.tobytes())
                    logging.debug(f"Sent visualization: {visualized_bgr.shape}")
                
                overall_duration = time.perf_counter() - overall_start
                root_span.set_attribute("overall_processing_duration_seconds", round(overall_duration*1000,1))
                logging.info(f"Frame verarbeitet und Ergebnisse veröffentlicht (Gesamtdauer: {overall_duration:.3f}s)")


       
    except KeyboardInterrupt:
        logging.info("Shutting down...")