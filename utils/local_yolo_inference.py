#!/usr/bin/env python3
"""
Standalone script for local YOLO inference.
This script runs in a separate process to avoid dependency conflicts.

Usage:
    python local_yolo_inference.py <model_path> <image_b64> <conf_thresh>

Returns:
    JSON with detections and annotated image (base64 encoded)
"""

import sys
import json
import os
import cv2
import numpy as np
import base64
import warnings
from typing import Dict, List, Any

# Suppress all warnings for clean subprocess output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['YOLO_VERBOSE'] = 'False'  # Suppress YOLO verbose output

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    try:
        _, buffer = cv2.imencode('.png', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

def decode_image_from_base64(img_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image data")
        return img
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

def get_color_for_class(name: str) -> tuple:
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

def load_model(model_path: str):
    """Load YOLO model with error handling"""
    try:
        from ultralytics import YOLO 
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model with verbose=False to suppress output
        model = YOLO(model_path, verbose=False)
        return model
        
    except ImportError as e:
        raise ImportError(f"YOLO dependencies not available: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def run_inference(model_path: str, image_b64: str, conf_thresh: float) -> Dict[str, Any]:
    """Run YOLO inference on image and return results"""
    try:
        # Validate inputs
        if not model_path or not image_b64:
            raise ValueError("Model path and image data are required")
        
        if not (0.0 <= conf_thresh <= 1.0):
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        # Load model
        model = load_model(model_path)
        
        # Decode image
        image = decode_image_from_base64(image_b64)
        
        # Convert BGR to RGB for YOLO
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference with suppressed output
        results = model(rgb_image, conf=conf_thresh, verbose=False)
        
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
                    class_name = model.names[class_id]
                    
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
                    color = get_color_for_class(class_name)
                    
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
        
        # Encode annotated image back to base64
        annotated_b64 = encode_image_to_base64(annotated)
        
        return {
            "detections": detections,
            "annotated_image": annotated_b64,
            "model_path": model_path,
            "confidence_threshold": conf_thresh,
            "num_detections": len(detections),
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

def main():
    """Main function for command line usage"""
    try:
        # Check command line arguments - now expect model_path, temp_file_path, conf_thresh
        if len(sys.argv) != 4:
            error_msg = {
                "error": "Invalid arguments",
                "usage": "python local_yolo_inference.py <model_path> <temp_image_file> <conf_thresh>",
                "provided": sys.argv[1:] if len(sys.argv) > 1 else [],
                "success": False
            }
            print(json.dumps(error_msg))
            sys.exit(1)
        
        # Parse arguments
        model_path = sys.argv[1]
        temp_image_file = sys.argv[2]
        try:
            conf_thresh = float(sys.argv[3])
        except ValueError:
            error_msg = {
                "error": f"Invalid confidence threshold: {sys.argv[3]}. Must be a number between 0 and 1.",
                "success": False
            }
            print(json.dumps(error_msg))
            sys.exit(1)
        
        # Read image from temporary file
        try:
            with open(temp_image_file, 'r') as f:
                image_b64 = f.read().strip()
        except Exception as e:
            error_msg = {
                "error": f"Failed to read image file {temp_image_file}: {str(e)}",
                "success": False
            }
            print(json.dumps(error_msg))
            sys.exit(1)
        
        # Run inference
        result = run_inference(model_path, image_b64, conf_thresh)
        
        # Output result as JSON
        print(json.dumps(result))
        
        # Exit with appropriate code
        sys.exit(0 if result.get("success", False) else 1)
        
        # Output result as JSON
        print(json.dumps(result))
        
        # Exit with appropriate code
        sys.exit(0 if result.get("success", False) else 1)
        
    except KeyboardInterrupt:
        error_msg = {"error": "Inference interrupted by user", "success": False}
        print(json.dumps(error_msg))
        sys.exit(1)
    except Exception as e:
        error_msg = {"error": f"Unexpected error: {str(e)}", "success": False}
        print(json.dumps(error_msg))
        sys.exit(1)

if __name__ == "__main__":
    main()
