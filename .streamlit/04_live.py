import streamlit as st
import cv2
import numpy as np
import time
import os
import sys
import av
from typing import List, Tuple
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import (
    CONF_THRESH_DEFAULT, 
    DETECTION_INTERVAL_DEFAULT,
    TRANSFORMER_CONF_THRESH,
    TRANSFORMER_DETECTION_INTERVAL,
    TRANSFORMER_PERSISTENCE_FRAMES,
    WEBCAM_WIDTH,
    WEBCAM_HEIGHT
)

# Check if local model file exists
def check_local_model_available():
    """Check if the local YOLO model file exists"""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "model", "best.pt"),
        "model/best.pt",
        os.path.join("model", "best.pt"),
        "best.pt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return True, path
    return False, None

# Check model availability
LOCAL_MODEL_AVAILABLE, LOCAL_MODEL_PATH = check_local_model_available()

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    """Load YOLO model once and cache it"""
    if not LOCAL_MODEL_PATH:
        st.error("❌ Local model not found!")
        return None
    try:
        model = YOLO(LOCAL_MODEL_PATH, verbose=False)
        # Model loaded silently - no success message needed
        return model
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="Live PPE Detection – Smart Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for consistent styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .stats-panel {
        background: #ffffff;
        border: 2px solid #e1e8ed;
        border-radius: 15px;
        padding: 1.5rem;
        margin-left: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        height: fit-content;
    }
    .compliance-alert {
        background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(255, 71, 87, 0.4);
        border: 3px solid #ff1744;
    }
    .compliance-good {
        background: linear-gradient(135deg, #2ed573 0%, #17c0eb 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(46, 213, 115, 0.4);
        border: 3px solid #00c851;
    }
    @keyframes pulse {
        0% { 
            opacity: 1; 
            transform: scale(1);
            box-shadow: 0 6px 20px rgba(255, 71, 87, 0.4);
        }
        50% { 
            opacity: 0.8; 
            transform: scale(1.02);
            box-shadow: 0 8px 30px rgba(255, 71, 87, 0.6);
        }
        100% { 
            opacity: 1; 
            transform: scale(1);
            box-shadow: 0 6px 20px rgba(255, 71, 87, 0.4);
        }
    }
    .ppe-stats-container {
        background: #f8fafe;
        border: 1px solid #e3f2fd;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .ppe-count-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem;
        margin: 0.3rem 0;
        background: white;
        border-radius: 8px;
        border-left: 4px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .ppe-count-item:hover {
        transform: translateX(2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .ppe-count-item.positive {
        border-left-color: #4caf50;
    }
    .ppe-count-item.negative {
        border-left-color: #f44336;
    }
    .ppe-count-item.neutral {
        border-left-color: #2196f3;
    }
    .ppe-icon {
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    .ppe-name {
        flex-grow: 1;
        font-weight: 500;
        color: #333;
    }
    .ppe-count {
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        background: #f5f5f5;
        color: #333;
    }
    .quick-metrics {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
    }
    .quick-metric {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        flex: 1;
        margin: 0 0.3rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .stats-header {
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e1e8ed;
    }
    .stats-header h3 {
        color: #333;
        margin: 0;
        font-size: 1.3rem;
    }
    .refresh-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
        margin: 0.5rem 0;
        width: 100%;
    }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #4caf50;
        border-radius: 50%;
        animation: blink 1s infinite;
        margin-right: 5px;
    }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color:white;margin:0;font-size:2.5rem;">📹 Live YOLO PPE Detection</h1>
    <p style="color:rgba(255,255,255,0.9);margin:0.5rem 0 0 0;font-size:1.1rem;">Real-time webcam monitoring with local PPE detection model</p>
</div>
""", unsafe_allow_html=True)

# Color mapping for classes
def get_color_for_class(name: str) -> Tuple[int,int,int]:
    """Get BGR color for detection class"""
    return {
        "Hardhat": (0,255,0),
        "Helmet": (0,255,0),
        "Mask": (0,255,128),
        "Safety Vest": (0,255,255),
        "Person": (255,0,0),
        "NO-Hardhat": (0,0,255),
        "NO-Helmet": (0,0,255),
        "NO-Mask": (255,0,255),
        "NO-Safety Vest": (255,165,0),
        "Safety Cone": (255,255,0),
        "Machinery": (128,0,128),
        "Vehicle": (0,128,255)
    }.get(name, (128,128,128))

def get_display_name(class_name: str) -> str:
    """Convert YOLO class names to display names"""
    name_mapping = {
        "Hardhat": "Helmet",
        "NO-Hardhat": "NO-Helmet", 
        "Mask": "Mask",
        "NO-Mask": "NO-Mask",
        "NO-Safety Vest": "NO-Safety Vest",
        "Person": "Person",
        "Safety Vest": "Safety Vest",
        "Safety Cone": "Safety Cone",
        "Machinery": "Machinery",
        "Vehicle": "Vehicle"
    }
    return name_mapping.get(class_name, class_name)

def calculate_compliance_percentage(detections: List) -> float:
    """Calculate PPE compliance percentage based on current detections"""
    if not detections:
        return 100.0  # No detections = assume compliant
    
    # Count people using the raw YOLO class name
    people_count = sum(1 for d in detections if d['class'] == 'Person')
    
    if people_count == 0:
        return 100.0  # No people detected
    
    # Count positive PPE detections using raw YOLO class names
    helmets = sum(1 for d in detections if d['class'] in ['Hardhat', 'Helmet'])
    masks = sum(1 for d in detections if d['class'] == 'Mask')
    safety_vests = sum(1 for d in detections if d['class'] == 'Safety Vest')
    
    # Count explicit violations using raw YOLO class names
    no_helmets = sum(1 for d in detections if d['class'] in ['NO-Hardhat', 'NO-Helmet'])
    no_masks = sum(1 for d in detections if d['class'] == 'NO-Mask')
    no_safety_vests = sum(1 for d in detections if d['class'] == 'NO-Safety Vest')
    
    # Calculate compliance per person (each person needs helmet, mask, safety vest)
    total_violations = no_helmets + no_masks + no_safety_vests
    
    # For missing PPE (when person is detected but no corresponding PPE)
    missing_helmets = max(0, people_count - helmets - no_helmets)
    missing_masks = max(0, people_count - masks - no_masks) 
    missing_vests = max(0, people_count - safety_vests - no_safety_vests)
    
    total_violations += missing_helmets + missing_masks + missing_vests
    
    # Maximum possible violations = people_count * 3 (helmet, mask, vest per person)
    max_possible_violations = people_count * 3
    
    # Calculate compliance percentage
    if max_possible_violations == 0:
        return 100.0
    
    compliance = max(0, ((max_possible_violations - total_violations) / max_possible_violations) * 100)
    return compliance

def run_yolo_inference(image: np.ndarray, conf_thresh: float, model) -> Tuple[List, np.ndarray]:
    """Run YOLO inference on image with improved bounding box precision"""
    try:
        if model is None:
            return [], image
        
        # Convert BGR to RGB for YOLO
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference with better NMS settings
        results = model(rgb_image, conf=conf_thresh, iou=0.4, verbose=False)
        
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
                    
                    # Apply additional filtering for box size reasonableness
                    box_width = x2 - x1
                    box_height = y2 - y1
                    image_area = image.shape[0] * image.shape[1]
                    box_area = box_width * box_height
                    
                    # Skip boxes that are too large (likely false positives)
                    if box_area > image_area * 0.7:  # Skip if box covers >70% of image
                        continue
                    
                    # Skip very small boxes that might be noise
                    if box_width < 20 or box_height < 20:
                        continue
                    
                    # Create detection dict with RAW class name for logic, display name for UI
                    detection = {
                        "x": float((x1 + x2) / 2),
                        "y": float((y1 + y2) / 2),
                        "width": float(box_width),
                        "height": float(box_height),
                        "confidence": confidence,
                        "class": class_name,  # Keep RAW YOLO class name for logic
                        "display_name": get_display_name(class_name)  # Display name for UI
                    }
                    detections.append(detection)
                    
                    # Draw more precise bounding box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    color = get_color_for_class(class_name)
                    
                    # Use thinner lines for better precision
                    thickness = 2 if confidence > 0.7 else 1
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw smaller, more precise label
                    display_name = get_display_name(class_name)
                    label = f"{display_name} {confidence:.2f}"
                    font_scale = 0.5  # Smaller font
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                    
                    # Draw compact label background
                    label_y = max(y1 - 5, label_size[1] + 5)
                    cv2.rectangle(annotated, (x1, label_y - label_size[1] - 4), 
                                (x1 + label_size[0] + 4, label_y + 2), color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated, label, (x1 + 2, label_y - 2), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        return detections, annotated
        
    except Exception as e:
        st.error(f"Inference error: {e}")
        return [], image

# Video transformer class with improved persistence and tracking
class PPEDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.conf_thresh = TRANSFORMER_CONF_THRESH  # From environment variable
        self.detection_interval = TRANSFORMER_DETECTION_INTERVAL  # From environment variable
        self.persistence_frames = TRANSFORMER_PERSISTENCE_FRAMES  # From environment variable
        self.last_detection_time = 0
        self.current_detections = []
        self.frame_count = 0
        self.detection_history = []  # Track detections over time
        self.debug_mode = False
        self.instant_detect = False
        
    def set_model(self, model):
        self.model = model
        
    def set_conf_thresh(self, conf_thresh):
        self.conf_thresh = conf_thresh
        
    def set_detection_interval(self, interval):
        self.detection_interval = interval
        
    def set_persistence_frames(self, frames):
        self.persistence_frames = frames
    
    def recv(self, frame):
        """New recv method to replace deprecated transform method"""
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # Debug: Print every 30 frames to show it's working
            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count}: Model available: {self.model is not None}, Shape: {img.shape}")
            
            # SIMPLIFIED DETECTION: Run detection every few frames instead of time-based
            should_detect = (self.frame_count % max(1, int(self.detection_interval * 10)) == 0 and 
                           self.model is not None) or self.instant_detect
            
            if should_detect and self.model is not None:
                if self.debug_mode:
                    print(f"Running detection on frame {self.frame_count} with conf_thresh: {self.conf_thresh}")
                
                # SIMPLIFIED DETECTION CALL - Remove complex filtering for now
                try:
                    # Ensure image is in correct format
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        # Convert BGR to RGB for YOLO
                        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Run inference with very permissive settings
                        results = self.model(rgb_image, conf=self.conf_thresh, iou=0.1, verbose=False)
                        
                        detections = []
                        annotated = img.copy()
                        
                        # Process all results
                        for result in results:
                            boxes = result.boxes
                            if boxes is not None and len(boxes) > 0:
                                if self.debug_mode:
                                    print(f"Raw YOLO found {len(boxes)} boxes")
                                for box in boxes:
                                    # Get box data
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    confidence = float(box.conf[0].cpu().numpy())
                                    class_id = int(box.cls[0].cpu().numpy())
                                    
                                    # Ensure model.names exists and has the class_id
                                    if hasattr(self.model, 'names') and class_id in self.model.names:
                                        class_name = self.model.names[class_id]
                                    else:
                                        class_name = f"class_{class_id}"  # Fallback name
                                    
                                    # MINIMAL FILTERING - only check if box is reasonable
                                    box_width = x2 - x1
                                    box_height = y2 - y1
                                    
                                    # Very permissive size check
                                    if box_width > 10 and box_height > 10:
                                        detection = {
                                            "x": float((x1 + x2) / 2),
                                            "y": float((y1 + y2) / 2),
                                            "width": float(box_width),
                                            "height": float(box_height),
                                            "confidence": confidence,
                                            "class": class_name,
                                            "display_name": get_display_name(class_name)
                                        }
                                        detections.append(detection)
                                        
                                        # Draw bounding box
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                        color = get_color_for_class(class_name)
                                        
                                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                                        
                                        # Draw label
                                        display_name = get_display_name(class_name)
                                        label = f"{display_name} {confidence:.2f}"
                                        cv2.putText(annotated, label, (x1, y1-10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Debug: Always print detection results
                        if detections:
                            if self.debug_mode:
                                print(f"SUCCESS: Found {len(detections)} detections: {[d['class'] for d in detections]}")
                        else:
                            if self.debug_mode:
                                model_classes = len(self.model.names) if hasattr(self.model, 'names') else 0
                                print(f"NO DETECTIONS: Image shape {img.shape}, Model classes: {model_classes}")
                        
                        # Update current detections
                        self.current_detections = detections
                        
                        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                    
                    else:
                        if self.debug_mode:
                            print(f"Invalid image shape: {img.shape}")
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                        
                except Exception as detection_error:
                    if self.debug_mode:
                        print(f"Detection error: {detection_error}")
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            else:
                # Show persistent detections
                annotated = img.copy()
                
                if self.current_detections:
                    for detection in self.current_detections:
                        # Draw bounding box
                        x1 = int(detection["x"] - detection["width"]/2)
                        y1 = int(detection["y"] - detection["height"]/2)
                        x2 = int(detection["x"] + detection["width"]/2)
                        y2 = int(detection["y"] + detection["height"]/2)
                        
                        color = get_color_for_class(detection["class"])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{detection['display_name']} {detection['confidence']:.2f}"
                        cv2.putText(annotated, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            
        except Exception as e:
            print(f"Recv error: {e}")
            return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Detection Parameters")
    
    # Show model info
    if not LOCAL_MODEL_AVAILABLE or not LOCAL_MODEL_PATH:
        st.error("❌ Local model not found")
        st.stop()
    
    # Add debug mode toggle
    debug_mode = st.checkbox("🔍 Debug Mode", value=False, help="Enable verbose debugging output")
    
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, TRANSFORMER_CONF_THRESH, 0.01, 
                            help="Lower values = more detections, may include false positives. Start with 0.01 for maximum sensitivity.")
    
    st.info(f"💡 **Tip:** For testing, try confidence = 0.01 first")
    
    detection_interval = st.slider("Detection Interval (seconds)", 0.1, 2.0, TRANSFORMER_DETECTION_INTERVAL, 0.1, 
                                  help="How often to run detection (lower = more responsive)")
    
    # Add persistence control
    persistence_frames = st.slider("Detection Persistence (frames)", 5, 60, TRANSFORMER_PERSISTENCE_FRAMES, 5,
                                 help="How many frames to keep showing detections")
    
    # Add reset button for troubleshooting
    if st.button("🔄 Reset Detection System", help="Clear cache and restart detection"):
        if 'transformer' in st.session_state:
            del st.session_state.transformer
        st.rerun()
        
    # Add instant detection toggle
    instant_detect = st.checkbox("⚡ Instant Detection", value=False, 
                                help="Run detection on every frame (may slow down stream)")

# Load model
model = load_yolo_model()
if model is None:
    st.error("❌ Failed to load YOLO model. Please check the model file.")
    st.stop()

# Initialize session state for transformer BEFORE using it
if 'transformer' not in st.session_state or not hasattr(st.session_state.transformer, 'set_persistence_frames'):
    # Create new transformer if it doesn't exist or is missing new methods
    st.session_state.transformer = PPEDetectionTransformer()

# Update transformer settings safely
st.session_state.transformer.set_model(model)
st.session_state.transformer.set_conf_thresh(conf_thresh)
st.session_state.transformer.set_detection_interval(detection_interval)
if hasattr(st.session_state.transformer, 'set_persistence_frames'):
    st.session_state.transformer.set_persistence_frames(persistence_frames)

# Add debug and instant detection settings
if hasattr(st.session_state.transformer, 'debug_mode'):
    st.session_state.transformer.debug_mode = debug_mode
if hasattr(st.session_state.transformer, 'instant_detect'):
    st.session_state.transformer.instant_detect = instant_detect

# Main content
st.markdown("### 📹 Live PPE Detection Stream")

# Create main layout: webcam on left, stats on right
webcam_col, stats_col = st.columns([2, 1])

with webcam_col:
    # WebRTC streamer with improved transformer handling
    try:
        # Use session state transformer directly instead of factory
        if 'transformer' not in st.session_state:
            st.session_state.transformer = PPEDetectionTransformer()
        
        # Ensure transformer has the model and current settings
        st.session_state.transformer.set_model(model)
        st.session_state.transformer.set_conf_thresh(conf_thresh)
        st.session_state.transformer.set_detection_interval(detection_interval)
        st.session_state.transformer.set_persistence_frames(persistence_frames)
        
        webrtc_ctx = webrtc_streamer(
            key="ppe-detection-stream",
            video_transformer_factory=PPEDetectionTransformer,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]}
                ]
            },
            media_stream_constraints={
                "video": {"width": WEBCAM_WIDTH, "height": WEBCAM_HEIGHT},
                "audio": False
            },
            async_processing=True,
        )
        
        # Display connection status with improved debugging
        if webrtc_ctx.state.playing:
            st.success("🟢 Camera stream active")
            if webrtc_ctx.video_transformer:
                transformer = webrtc_ctx.video_transformer
                
                # Update transformer with current settings
                transformer.set_model(model)
                transformer.set_conf_thresh(conf_thresh)
                transformer.set_detection_interval(detection_interval)
                transformer.set_persistence_frames(persistence_frames)
                transformer.debug_mode = debug_mode
                transformer.instant_detect = instant_detect
                
                model_status = "✅ Ready" if hasattr(transformer, 'model') and transformer.model is not None else "❌ No Model"
                st.info(f"🔧 Detection transformer: {model_status}")
                
                # Debug info
                if hasattr(transformer, 'frame_count'):
                    st.caption(f"Frames processed: {transformer.frame_count}")
                if hasattr(transformer, 'current_detections'):
                    st.caption(f"Current detections: {len(transformer.current_detections) if transformer.current_detections else 0}")
            else:
                st.warning("⚠️ Detection transformer not ready")
        elif webrtc_ctx.state.signalling:
            st.info("🟡 Connecting to camera...")
        else:
            st.info("⚪ Camera stream stopped. Click 'Start' to begin.")
            
        # Add immediate test button
        if st.button("🚀 Force Detection Test", help="Test detection on current frame"):
            if webrtc_ctx.video_transformer and hasattr(webrtc_ctx.video_transformer, 'model'):
                transformer = webrtc_ctx.video_transformer
                if transformer.model:
                    st.success("Model is available in transformer!")
                    st.write(f"Confidence threshold: {transformer.conf_thresh}")
                    st.write(f"Model classes: {len(transformer.model.names)}")
                else:
                    st.error("Model not available in transformer")
            else:
                st.warning("Transformer not ready for testing")

    except Exception as e:
        st.error(f"❌ WebRTC Error: {str(e)}")
        st.info("""
        **Troubleshooting Tips:**
        - Ensure your browser supports WebRTC (Chrome, Firefox, Safari, Edge)
        - Allow camera access when prompted
        - Try refreshing the page
        - Check if camera is being used by another application
        - On macOS: System Preferences > Security & Privacy > Camera
        - Try using a different browser
        """)
        
        # Fallback: Show camera test button
        st.markdown("### 📸 Alternative: Camera Test")
        if st.button("📱 Test Camera Access"):
            st.info("If this button works, your browser supports camera access. The WebRTC issue may be temporary.")
            st.markdown("""
            **Manual Steps:**
            1. Refresh the browser page
            2. Clear browser cache 
            3. Try in an incognito/private window
            4. Check browser permissions for camera access
            """)

with stats_col:
    st.markdown('<div class="stats-panel">', unsafe_allow_html=True)
    
    # Statistics Header
    st.markdown("""
    <div class="stats-header">
        <h3><span class="live-indicator"></span>📊 Live PPE Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Manual refresh button
    col_refresh1, col_refresh2 = st.columns([1, 1])
    with col_refresh1:
        if st.button("🔄 Refresh Stats", key="refresh_stats", help="Force refresh statistics"):
            st.rerun()
    with col_refresh2:
        # Remove auto-refresh checkbox that was causing infinite loops
        st.write("Manual refresh only")
    
    # Live metrics display
    if 'webrtc_ctx' in locals() and webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
        transformer = webrtc_ctx.video_transformer
        
        if hasattr(transformer, 'current_detections') and transformer.current_detections:
            detections = transformer.current_detections
            
            # Calculate compliance
            compliance = calculate_compliance_percentage(detections)
            
            # Show compliance alert or success
            if compliance < 60:
                st.markdown(f"""
                <div class="compliance-alert">
                    🚨 COMPLIANCE ALERT 🚨<br>
                    <div style="font-size: 1.5rem; margin-top: 0.5rem;">{compliance:.1f}% Compliant</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="compliance-good">
                    ✅ COMPLIANT WORKPLACE<br>
                    <div style="font-size: 1.5rem; margin-top: 0.5rem;">{compliance:.1f}% Compliant</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Count detections - include all model classes
            ppe_counts = {
                "Person": 0,
                "Helmet": 0,
                "Mask": 0,
                "Safety Vest": 0,
                "NO-Helmet": 0,
                "NO-Mask": 0,
                "NO-Safety Vest": 0,
                "Safety Cone": 0,
                "Machinery": 0,
                "Vehicle": 0
            }
            
            for detection in detections:
                display_name = get_display_name(detection.get('class', ''))
                if display_name in ppe_counts:
                    ppe_counts[display_name] += 1
            
            # Show quick metrics
            people = ppe_counts["Person"]
            violations = ppe_counts["NO-Helmet"] + ppe_counts["NO-Mask"] + ppe_counts["NO-Safety Vest"]
            
            st.markdown("""
            <div class="quick-metrics">
                <div class="quick-metric">
                    <div class="metric-value">👥 {}</div>
                    <div class="metric-label">People Detected</div>
                </div>
                <div class="quick-metric">
                    <div class="metric-value">⚠️ {}</div>
                    <div class="metric-label">Violations</div>
                </div>
            </div>
            """.format(people, violations), unsafe_allow_html=True)
            
            # Display detailed statistics
            st.markdown('<div class="ppe-stats-container">', unsafe_allow_html=True)
            st.markdown("**Detection Results:**")
            
            # Define icons and categories for all model classes
            ppe_categories = [
                ("Person", "👤", "neutral"),
                ("Helmet", "⛑️", "positive"),
                ("Mask", "😷", "positive"),
                ("Safety Vest", "🦺", "positive"),
                ("NO-Helmet", "🚫⛑️", "negative"),
                ("NO-Mask", "🚫😷", "negative"),
                ("NO-Safety Vest", "🚫🦺", "negative"),
                ("Safety Cone", "🚧", "neutral"),
                ("Machinery", "⚙️", "neutral"),
                ("Vehicle", "🚗", "neutral")
            ]
            
            for category, icon, status in ppe_categories:
                count = ppe_counts.get(category, 0)
                st.markdown(f"""
                <div class="ppe-count-item {status}">
                    <div>
                        <span class="ppe-icon">{icon}</span>
                        <span class="ppe-name">{category}</span>
                    </div>
                    <span class="ppe-count">{count}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show debug info in expandable section
            with st.expander("🔍 Debug Information", expanded=False):
                st.json({
                    'detection_count': len(detections),
                    'raw_classes': [d.get('class', 'unknown') for d in detections],
                    'compliance': compliance
                })
            
        elif hasattr(transformer, 'current_detections'):
            # No detections
            st.markdown(f"""
            <div class="compliance-good">
                ✅ NO DETECTIONS<br>
                <div style="font-size: 1.5rem; margin-top: 0.5rem;">100.0% Compliant</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="quick-metrics">
                <div class="quick-metric">
                    <div class="metric-value">👥 0</div>
                    <div class="metric-label">People Detected</div>
                </div>
                <div class="quick-metric">
                    <div class="metric-value">⚠️ 0</div>
                    <div class="metric-label">Violations</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🔧 Transformer not properly initialized")
    else:
        st.info("📷 Start camera stream to see live statistics")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Show live feed status
st.markdown("---")
if 'webrtc_ctx' in locals() and hasattr(webrtc_ctx, 'state'):
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "🟢 Active" if webrtc_ctx.state.playing else "⚪ Stopped"
        st.write(f"**Stream Status:** {status}")
    with col2:
        signalling = "🟡 Connected" if webrtc_ctx.state.signalling else "⚪ Disconnected"
        st.write(f"**Signalling:** {signalling}")
    with col3:
        st.write(f"**Model:** {'✅ Loaded' if model else '❌ Error'}")

# Performance test section
st.markdown("---")
st.markdown("### 🧪 Model Performance Test")

if st.button("Test Model Performance"):
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    with st.spinner("Testing model inference speed..."):
        start_time = time.time()
        num_tests = 5
        for _ in range(num_tests):
            rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            results = model(rgb_image, conf=conf_thresh, verbose=False)
        avg_time = (time.time() - start_time) / num_tests * 1000
    
    st.success(f"✅ Model Performance Test Complete!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Inference Time", f"{avg_time:.1f}ms")
    with col2:
        st.metric("Theoretical Max FPS", f"{1000/avg_time:.1f}")
    with col3:
        st.metric("Status", "Ready" if avg_time < 200 else "Slow")

# Model class verification
st.markdown("---")
st.markdown("### 🔍 Model Testing & Verification")

col_test1, col_test2 = st.columns(2)

with col_test1:
    if st.button("Show Model Classes"):
        if model:
            st.markdown("**Available YOLO Classes:**")
            class_info = {}
            for class_id, class_name in model.names.items():
                display_name = get_display_name(class_name)
                class_info[f"ID {class_id}"] = f"{class_name} → {display_name}"
            
            st.json(class_info)
            
            st.markdown("**Expected PPE Categories:**")
            expected_categories = ["Helmet", "Mask", "NO-Helmet", "NO-Mask", "NO-Safety Vest", "Person", "Safety Vest"]
            st.write(", ".join(expected_categories))
        else:
            st.error("Model not loaded")

with col_test2:
    if st.button("🧪 Test Detection"):
        if model:
            with st.spinner("Testing model detection..."):
                # Create test image
                test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
                
                # Test with current confidence threshold
                results = model(test_image, conf=conf_thresh, verbose=False)
                detection_count = 0
                for result in results:
                    if result.boxes is not None:
                        detection_count += len(result.boxes)
                
                if detection_count > 0:
                    st.success(f"✅ Model working! Found {detection_count} detections on test image")
                    st.info("If you're not seeing detections on camera, try:")
                    st.write("• Lower the confidence threshold")
                    st.write("• Point camera at different objects")
                    st.write("• Ensure good lighting")
                    st.write("• Move closer to objects")
                else:
                    st.warning(f"⚠️ No detections at confidence {conf_thresh:.2f}")
                    st.info("Try lowering the confidence threshold in the sidebar")
        else:
            st.error("Model not loaded")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6c757d;padding:1rem;'>🛡️ Smart Safety Monitor - Local YOLO Live Detection</div>", unsafe_allow_html=True)

