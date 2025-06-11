"""
Utilities package for Local YOLO PPE Detection System
Contains configuration management for the live PPE detection app
"""

from .config import (
    LOCAL_MODEL_PATH,
    CONF_THRESH_DEFAULT,
    DETECTION_INTERVAL_DEFAULT,
    TRANSFORMER_CONF_THRESH,
    TRANSFORMER_DETECTION_INTERVAL,
    TRANSFORMER_PERSISTENCE_FRAMES,
    WEBCAM_WIDTH,
    WEBCAM_HEIGHT,
    get_model_info,
    validate_config
)

__all__ = [
    # Configuration
    'LOCAL_MODEL_PATH',
    'CONF_THRESH_DEFAULT',
    'DETECTION_INTERVAL_DEFAULT',
    'TRANSFORMER_CONF_THRESH',
    'TRANSFORMER_DETECTION_INTERVAL',
    'TRANSFORMER_PERSISTENCE_FRAMES',
    'WEBCAM_WIDTH',
    'WEBCAM_HEIGHT',
    'get_model_info',
    'validate_config'
]
