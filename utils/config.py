"""
Configuration module for PPE Detection System
Loads environment variables from .env file
"""

import os
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
load_dotenv()

# Roboflow API Configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com")

# Parse available models
API_MODELS_STR = os.getenv("API_MODELS", "ppe-factory-bmdcj/2,pbe-detection/4,safety-pyazl/1")
API_MODELS: List[str] = [model.strip() for model in API_MODELS_STR.split(",")]

# Default Detection Parameters
MODEL_ID_DEFAULT = os.getenv("MODEL_ID_DEFAULT", "ppe-factory-bmdcj/2")
CONF_THRESH_DEFAULT = float(os.getenv("CONF_THRESH_DEFAULT", "0.5"))
OVERLAP_THRESH_DEFAULT = float(os.getenv("OVERLAP_THRESH_DEFAULT", "0.3"))
DETECTION_INTERVAL_DEFAULT = float(os.getenv("DETECTION_INTERVAL_DEFAULT", "4.0"))

def validate_config():
    """Validate that required environment variables are set"""
    if not ROBOFLOW_API_KEY:
        raise ValueError(
            "ROBOFLOW_API_KEY not found in environment variables. "
            "Please copy .env.example to .env and set your API key."
        )
    return True

# Validate config on import
validate_config()
