#!/usr/bin/env python3
"""
Optimized YOLO model manager for in-process inference.
Loads model once and keeps it in memory for fast real-time processing.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['YOLO_VERBOSE'] = 'False'


class YOLOModelManager:
    """Singleton class to manage YOLO model loading and inference"""
    
    _instance = None
    _model = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str) -> bool:
        """Load YOLO model if not already loaded or if path changed"""
        try:
            # Check if we need to load/reload model
            if self._model is None or self._model_path != model_path:
                # Import here to avoid conflicts in main streamlit process
                from ultralytics import YOLO
                
                # Verify model file exists
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Load model with verbose=False
                print(f"Loading YOLO model from {model_path}...")
                self._model = YOLO(model_path, verbose=False)
                self._model_path = model_path
                print("Model loaded successfully!")
                
            return True
            
        except ImportError as e:
            print(f"YOLO dependencies not available: {str(e)}")
            return False
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._model is not None
    
    def get_color_for_class(self, name: str) -> tuple:
        """Get BGR color tuple for detection class"""
        colors = {
            "hard-hat": (0, 255, 0),        # Green
            "helmet": (0, 255, 0),          # Green
            "safety-vest": (0, 255, 255),   # Yellow
            "person": (255, 0, 0),          # Blue
            "no-hard-hat": (0, 0, 255),     # Red
            "no-helmet": (0, 0, 255),       # Red
            "no-safety-vest": (0, 165, 255) # Orange
        }
        return colors.get(name.lower(), (128, 128, 128))  # Gray default
    
    def run_inference(self, image: np.ndarray, conf_thresh: float = 0.5) -> Tuple[List[Dict], np.ndarray]:
        """
        Run YOLO inference on image
        
        Args:
            image: Input image as numpy array (BGR format)
            conf_thresh: Confidence threshold for detections
            
        Returns:
            Tuple of (detections_list, annotated_image)
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Validate inputs
            if image is None or image.size == 0:
                raise ValueError("Invalid image data")
            
            if not (0.0 <= conf_thresh <= 1.0):
                raise ValueError("Confidence threshold must be between 0 and 1")
            
            # Convert BGR to RGB for YOLO
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference with suppressed output
            results = self._model(rgb_image, conf=conf_thresh, verbose=False)
            
            # Extract detections
            detections = []
            annotated = image.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates and info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self._model.names[class_id]
                        
                        # Convert to center format for consistency with Roboflow API
                        cx = float((x1 + x2) / 2)
                        cy = float((y1 + y2) / 2)
                        width = float(x2 - x1)
                        height = float(y2 - y1)
                        
                        # Create detection dict in same format as Roboflow
                        detection = {
                            "x": cx,
                            "y": cy,
                            "width": width,
                            "height": height,
                            "confidence": confidence,
                            "class": class_name
                        }
                        detections.append(detection)
                        
                        # Draw bounding box on image
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        color = self.get_color_for_class(class_name)
                        
                        # Draw rectangle
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name} ({confidence:.2f})"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Draw label background
                        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 8), 
                                    (x1 + label_size[0], y1), color, -1)
                        
                        # Draw label text
                        cv2.putText(annotated, label, (x1, y1 - 4), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return detections, annotated
            
        except Exception as e:
            print(f"Inference error: {str(e)}")
            return [], image


# Global instance for easy access
yolo_manager = YOLOModelManager()


def run_optimized_local_inference(image: np.ndarray, model_path: str, conf_thresh: float = 0.5) -> Tuple[List[Dict], np.ndarray]:
    """
    Optimized local YOLO inference function
    
    Args:
        image: Input image as numpy array (BGR format)
        model_path: Path to YOLO model file
        conf_thresh: Confidence threshold
        
    Returns:
        Tuple of (detections_list, annotated_image)
    """
    try:
        # Load model if needed (only happens once or when path changes)
        if not yolo_manager.load_model(model_path):
            raise RuntimeError("Failed to load YOLO model")
        
        # Run inference
        return yolo_manager.run_inference(image, conf_thresh)
        
    except Exception as e:
        print(f"Optimized inference failed: {str(e)}")
        return [], image


# Backward compatibility function
def check_yolo_dependencies() -> bool:
    """Check if YOLO dependencies are available"""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        return False
