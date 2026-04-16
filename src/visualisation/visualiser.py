"""
Visualizer for litter detection masks.

Receives segmentation masks from inference and displays them with different modes:
- Binary mask (red overlay)
- Probability heatmap
- Original frame with visualization overlay

Controls:
- 'b': Toggle binary mask view
- 'p': Toggle probability heatmap view  
- 'v': Toggle visualization overlay view
- 'q': Quit application
"""

import cv2
import numpy as np
import zenoh
import logging
import sys
import json
from pathlib import Path
import threading
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("visualiser")

# Display mode constants
MODE_BINARY = 0
MODE_PROBABILITY = 1
MODE_VISUALIZATION = 2

# Global state
current_mode = MODE_VISUALIZATION
display_lock = threading.Lock()

# Received data buffers
latest_frame = None
latest_mask_binary = None
latest_mask_probabilities = None
latest_visualization = None
data_ready = threading.Event()


def colormap_heatmap(probabilities: np.ndarray) -> np.ndarray:
    """
    Convert probability map to color heatmap (blue -> green -> red).
    
    Args:
        probabilities: Probability map with values 0-1, shape (H, W)
    
    Returns:
        heatmap: RGB heatmap image
    """
    # Normalize to 0-255
    heatmap_uint8 = (probabilities * 255).astype(np.uint8)
    
    # Apply colormap (COLORMAP_JET: blue -> cyan -> green -> yellow -> red)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    return heatmap_rgb


def visualize_binary_mask(mask_binary: np.ndarray, frame_shape: tuple = None) -> np.ndarray:
    """
    Create red overlay visualization for binary mask.
    
    Args:
        mask_binary: Binary mask (0 or 1), shape (H, W)
        frame_shape: Optional frame shape to resize mask to
    
    Returns:
        RGB image with red overlay
    """
    if frame_shape is not None:
        h, w = frame_shape[:2]
        mask_binary = cv2.resize(mask_binary.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create RGB image with red overlay on detected areas
    h, w = mask_binary.shape[:2]
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # White background
    rgb_image[:, :] = [220, 220, 220]
    
    # Red for detected litter
    rgb_image[mask_binary == 1] = [255, 0, 0]
    
    return rgb_image


def on_frame(sample: zenoh.Sample):
    """Zenoh callback for frame reception."""
    global latest_frame
    try:
        latest_frame = bytes(sample.payload)
        data_ready.set()
    except Exception as e:
        logger.exception(f"Error receiving frame: {e}")


def on_mask_binary(sample: zenoh.Sample):
    """Zenoh callback for binary mask reception."""
    global latest_mask_binary
    try:
        payload = bytes(sample.payload)
        if payload:
            # Mask is sent as numpy array bytes
            latest_mask_binary = np.frombuffer(payload, dtype=np.uint8).reshape(-1)
            data_ready.set()
    except Exception as e:
        logger.exception(f"Error receiving binary mask: {e}")


def on_mask_probabilities(sample: zenoh.Sample):
    """Zenoh callback for probability mask reception."""
    global latest_mask_probabilities
    try:
        payload = bytes(sample.payload)
        if payload:
            # Probabilities are sent as numpy array bytes
            latest_mask_probabilities = np.frombuffer(payload, dtype=np.float32).reshape(-1)
            data_ready.set()
    except Exception as e:
        logger.exception(f"Error receiving probability mask: {e}")


def on_visualization(sample: zenoh.Sample):
    """Zenoh callback for visualization reception."""
    global latest_visualization
    try:
        payload = bytes(sample.payload)
        if payload:
            latest_visualization = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            data_ready.set()
    except Exception as e:
        logger.exception(f"Error receiving visualization: {e}")


def display_loop():
    """Main display loop."""
    global current_mode, latest_frame, latest_mask_binary, latest_mask_probabilities, latest_visualization
    
    window_name = "Litter Detection Visualization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    mode_names = {
        MODE_BINARY: "Binary Mask (Red Overlay)",
        MODE_PROBABILITY: "Probability Heatmap",
        MODE_VISUALIZATION: "Visualization Overlay",
    }
    
    logger.info("Display loop started. Press:")
    logger.info("  'b' - Binary mask view")
    logger.info("  'p' - Probability heatmap view")
    logger.info("  'v' - Visualization overlay view")
    logger.info("  'q' - Quit")
    
    while True:
        data_ready.wait(timeout=0.1)
        
        with display_lock:
            if current_mode == MODE_BINARY and latest_mask_binary is not None:
                try:
                    # Reshape binary mask back to 2D
                    mask_binary = latest_mask_binary.reshape(384, 384)
                    display_image = visualize_binary_mask(mask_binary)
                except Exception as e:
                    logger.error(f"Error reshaping binary mask: {e}")
                    display_image = np.zeros((384, 384, 3), dtype=np.uint8)
                    
            elif current_mode == MODE_PROBABILITY and latest_mask_probabilities is not None:
                try:
                    # Reshape probability mask back to 2D
                    mask_probs = latest_mask_probabilities.reshape(384, 384)
                    display_image = colormap_heatmap(mask_probs)
                except Exception as e:
                    logger.error(f"Error reshaping probability mask: {e}")
                    display_image = np.zeros((384, 384, 3), dtype=np.uint8)
                    
            elif current_mode == MODE_VISUALIZATION and latest_visualization is not None:
                display_image = cv2.cvtColor(latest_visualization, cv2.COLOR_BGR2RGB)
            else:
                display_image = np.zeros((384, 384, 3), dtype=np.uint8)
        
        # Add text overlay with current mode
        display_with_text = display_image.copy()
        cv2.putText(
            display_with_text,
            f"Mode: {mode_names[current_mode]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_with_text,
            "b=Binary | p=Probability | v=Visualization | q=Quit",
            (10, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        
        # Convert RGB to BGR for OpenCV display
        display_bgr = cv2.cvtColor(display_with_text, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, display_bgr)
        
        # Handle keyboard input (non-blocking)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quitting...")
            break
        elif key == ord('b'):
            with display_lock:
                current_mode = MODE_BINARY
            logger.info("Switched to Binary Mask view")
        elif key == ord('p'):
            with display_lock:
                current_mode = MODE_PROBABILITY
            logger.info("Switched to Probability Heatmap view")
        elif key == ord('v'):
            with display_lock:
                current_mode = MODE_VISUALIZATION
            logger.info("Switched to Visualization Overlay view")
    
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    logger.info(f"Connecting to Zenoh router at {config.ZENOH_ROUTER}...")
    
    z = zenoh.open(
        zenoh.Config().insert_json5(
            "connect/endpoints", json.dumps([config.ZENOH_ROUTER])
        )
    )
    
    logger.info("Subscribing to Zenoh topics...")
    
    # Subscribe to all relevant topics
    frame_sub = z.declare_subscriber("litter/frame", on_frame)
    mask_binary_sub = z.declare_subscriber("litter/mask_binary", on_mask_binary)
    mask_probs_sub = z.declare_subscriber("litter/mask_probabilities", on_mask_probabilities)
    viz_sub = z.declare_subscriber("litter/visualization", on_visualization)
    
    logger.info("Subscribed to topics: litter/frame, litter/mask_binary, litter/mask_probabilities, litter/visualization")
    
    try:
        display_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        z.close()
        logger.info("Zenoh session closed")


if __name__ == "__main__":
    main()
