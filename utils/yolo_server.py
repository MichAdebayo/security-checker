#!/usr/bin/env python3
"""
Persistent YOLO inference server for real-time performance.
Keeps the model loaded in memory and processes requests via HTTP.
"""

import os
import sys
import json
import base64
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import signal
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

class YOLOServer:
    def __init__(self, model_path, port=8888):
        self.model_path = model_path
        self.port = port
        self.model = None
        self.server = None
        
    def load_model(self):
        """Load YOLO model once at startup"""
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path, verbose=False)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def decode_image_from_base64(self, img_str: str) -> np.ndarray:
        """Convert base64 string to OpenCV image"""
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    def encode_image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.png', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    
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
    
    def run_inference(self, image: np.ndarray, conf_thresh: float = 0.5):
        """Run YOLO inference on image"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Convert BGR to RGB for YOLO
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_image, conf=conf_thresh, verbose=False)
            
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
                        class_name = self.model.names[class_id]
                        
                        # Convert to center format for consistency
                        cx = float((x1 + x2) / 2)
                        cy = float((y1 + y2) / 2)
                        width = float(x2 - x1)
                        height = float(y2 - y1)
                        
                        # Create detection dict
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
            print(f"Inference error: {e}")
            return [], image

class YOLORequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            try:
                # Read request data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Parse JSON
                data = json.loads(post_data.decode('utf-8'))
                image_b64 = data.get('image_b64', '')
                conf_thresh = float(data.get('conf_thresh', 0.5))
                
                # Decode image
                image = self.server.yolo_server.decode_image_from_base64(image_b64)
                
                # Run inference
                detections, annotated = self.server.yolo_server.run_inference(image, conf_thresh)
                
                # Encode result
                annotated_b64 = self.server.yolo_server.encode_image_to_base64(annotated)
                
                # Send response
                response = {
                    "detections": detections,
                    "annotated_image": annotated_b64,
                    "success": True
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                # Send error response
                error_response = {
                    "error": str(e),
                    "success": False
                }
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "ok", "model_loaded": self.server.yolo_server.model is not None}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server(model_path, port=8888):
    """Run the YOLO server"""
    yolo_server = YOLOServer(model_path, port)
    
    # Load model
    if not yolo_server.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Create HTTP server
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, YOLORequestHandler)
    httpd.yolo_server = yolo_server
    
    print(f"YOLO server running on http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop")
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        httpd.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python yolo_server.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    run_server(model_path)
