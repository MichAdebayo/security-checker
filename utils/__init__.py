"""
Utilities package for PPE Detection System
Contains YOLO model management, inference servers, and configuration
"""

from .config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_API_URL, 
    API_MODELS,
    MODEL_ID_DEFAULT,
    CONF_THRESH_DEFAULT,
    OVERLAP_THRESH_DEFAULT,
    DETECTION_INTERVAL_DEFAULT,
    validate_config
)
from .yolo_model_manager import YOLOModelManager, run_optimized_local_inference, check_yolo_dependencies
from .yolo_server import YOLOServer, run_server

__all__ = [
    # Configuration
    'ROBOFLOW_API_KEY',
    'ROBOFLOW_API_URL',
    'API_MODELS',
    'MODEL_ID_DEFAULT',
    'CONF_THRESH_DEFAULT',
    'OVERLAP_THRESH_DEFAULT',
    'DETECTION_INTERVAL_DEFAULT',
    'validate_config',
    # YOLO Management
    'YOLOModelManager',
    'run_optimized_local_inference', 
    'check_yolo_dependencies',
    'YOLOServer',
    'run_server'
]
