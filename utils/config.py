"""
Configuration module for Local YOLO PPE Detection System
Loads environment variables from .env file
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local YOLO Model Configuration
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "model/best.pt")

# Default Detection Parameters
CONF_THRESH_DEFAULT = float(os.getenv("CONF_THRESH_DEFAULT", "0.5"))
DETECTION_INTERVAL_DEFAULT = float(os.getenv("DETECTION_INTERVAL_DEFAULT", "2.0"))

# Live Detection Transformer Parameters
TRANSFORMER_CONF_THRESH = float(os.getenv("TRANSFORMER_CONF_THRESH", "0.01"))  # Very low for testing  # Very low threshold for better detection
TRANSFORMER_DETECTION_INTERVAL = float(os.getenv("TRANSFORMER_DETECTION_INTERVAL", "0.5"))
TRANSFORMER_PERSISTENCE_FRAMES = int(os.getenv("TRANSFORMER_PERSISTENCE_FRAMES", "20"))

# Webcam Stream Configuration
WEBCAM_WIDTH = int(os.getenv("WEBCAM_WIDTH", "640"))
WEBCAM_HEIGHT = int(os.getenv("WEBCAM_HEIGHT", "400"))

# YOLO Server Configuration (optional)
YOLO_SERVER_HOST = os.getenv("YOLO_SERVER_HOST", "127.0.0.1")
YOLO_SERVER_PORT = int(os.getenv("YOLO_SERVER_PORT", "8888"))

def get_model_info():
    """Get information about the local model"""
    if os.path.exists(LOCAL_MODEL_PATH):
        size = os.path.getsize(LOCAL_MODEL_PATH) / (1024 * 1024)  # Size in MB
        return {
            "path": LOCAL_MODEL_PATH,
            "exists": True,
            "size_mb": round(size, 2)
        }
    return {
        "path": LOCAL_MODEL_PATH,
        "exists": False,
        "size_mb": 0
    }

def validate_config():
    """Validate configuration on startup (optional validation)"""
    model_info = get_model_info()
    if not model_info["exists"]:
        print(f"Warning: Local YOLO model not found at {LOCAL_MODEL_PATH}")
    return True

# Validate config on import
validate_config()
