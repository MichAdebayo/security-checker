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

# Load audio file
@st.cache_data
def load_audio_bytes():
    """Load emergency alarm audio file using correct path resolution"""
    # Try multiple possible paths to find the audio file
    from pathlib import Path
    possible_paths = [
        Path(__file__).parent.parent / "assets" / "emergency-alarm.mp3",  # From .streamlit/ to project root
        Path(__file__).parent / "assets" / "emergency-alarm.mp3",        # If assets is in .streamlit/
        Path("assets") / "emergency-alarm.mp3",                          # Relative to current directory
        Path("/Users/michaeladebayo/Documents/Simplon/brief_projects/security-checker/assets/emergency-alarm.mp3")  # Absolute path
    ]
    
    for audio_path in possible_paths:
        try:
            if audio_path.exists():
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                    file_size = audio_path.stat().st_size / 1024  # KB
                    return audio_bytes, file_size
        except Exception as e:
            continue
    
    # If no path worked, return None silently
    return None, 0

# Page configuration
st.set_page_config(
    page_title="Live PPE Detection – Smart Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for audio (must be before sidebar)
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = False

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
        background: #f44336;  /* Red by default */
        border-radius: 50%;
        margin-right: 5px;
    }
    .live-indicator.active {
        background: #4caf50;  /* Green when active */
        animation: blink 1s infinite;
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
    """Calculate PPE compliance percentage based on presence of all 3 PPE items (same as video.py)"""
    if not detections:
        return 100.0  # No detections = assume compliant

    # Count positive PPE detections using normalized class names (same as video.py)
    helmet_count = sum(1 for d in detections if d['class'].lower() in ['hardhat', 'helmet'])
    mask_count = sum(1 for d in detections if d['class'].lower() == 'mask')
    vest_count = sum(1 for d in detections if d['class'].lower() == 'safety vest')
    
    # Calculate compliance for each PPE item (1 if present, 0 if not)
    helmet_compliance = 1 if helmet_count > 0 else 0
    mask_compliance = 1 if mask_count > 0 else 0
    vest_compliance = 1 if vest_count > 0 else 0
    
    # Total compliance is sum of individual compliances divided by 3
    total_compliance = (helmet_compliance + mask_compliance + vest_compliance) / 3 * 100
    
    return round(total_compliance, 1)


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
        import time
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # Debug: Print every 30 frames to show it's working
            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count}: Model available: {self.model is not None}, Shape: {img.shape}")
            
            # IMPROVED DETECTION: Run detection more frequently for better real-time metrics
            # Run detection every 5-10 frames instead of every few seconds for responsive metrics
            detection_frame_interval = max(5, int(self.detection_interval * 5))  # Much more frequent
            should_detect = (self.frame_count % detection_frame_interval == 0 and 
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
                        
                        # Update current detections - ALWAYS update, even if empty
                        self.current_detections = detections
                        
                        # Add timestamp to track when detections were last updated
                        self.last_detection_time = time.time()
                        
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
                # Show persistent detections, but clear them if too old
                current_time = time.time()
                detection_age = current_time - getattr(self, 'last_detection_time', current_time)
                max_detection_age = 3.0  # Clear detections after 3 seconds of no new detections
                
                annotated = img.copy()
                
                if self.current_detections and detection_age < max_detection_age:
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
                elif detection_age >= max_detection_age:
                    # Clear old detections to ensure metrics reflect current state
                    self.current_detections = []
                
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
    
    
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, TRANSFORMER_CONF_THRESH, 0.01, 
                            help="Lower values = more detections, may include false positives. Start with 0.01 for maximum sensitivity.")
    
    detection_interval = st.slider("Detection Interval (seconds)", 0.1, 2.0, TRANSFORMER_DETECTION_INTERVAL, 0.1, 
                                  help="How often to run detection (lower = more responsive)")
    
    # Add persistence control
    persistence_frames = st.slider("Detection Persistence (frames)", 5, 60, TRANSFORMER_PERSISTENCE_FRAMES, 5,
                                 help="How many frames to keep showing detections")
    
    st.markdown("---")
    st.header("🔊 Audio Alerts")
    
    # Audio toggle control
    audio_enabled = st.toggle(
        "Enable Audio Alerts", 
        value=st.session_state.audio_enabled,
        help="Automatically play alarm sound when violations are detected"
    )
    
    # Update session state
    if audio_enabled != st.session_state.audio_enabled:
        st.session_state.audio_enabled = audio_enabled

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
        
        # Create a factory function that returns our session state transformer
        def create_transformer() -> PPEDetectionTransformer:
            # Create a new transformer with the current settings
            transformer = PPEDetectionTransformer()
            transformer.set_model(model)
            transformer.set_conf_thresh(conf_thresh)
            transformer.set_detection_interval(detection_interval)
            transformer.set_persistence_frames(persistence_frames)
            
            # Store reference in session state for statistics access
            st.session_state.active_transformer = transformer
            return transformer
        
        webrtc_ctx = webrtc_streamer(
            key="ppe-detection-stream",
            video_processor_factory=create_transformer,
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
            if webrtc_ctx.video_processor:
                transformer = webrtc_ctx.video_processor
                
                # Update transformer with current settings
                transformer.set_model(model)
                transformer.set_conf_thresh(conf_thresh)
                transformer.set_detection_interval(detection_interval)
                transformer.set_persistence_frames(persistence_frames)
            else:
                st.warning("⚠️ Detection transformer not ready")
        elif webrtc_ctx.state.signalling:
            st.info("🟡 Connecting to camera...")
        else:
            st.info("⚪ Camera stream stopped. Click 'Start' to begin.")

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
    
    # Statistics Header - indicator changes based on camera status
    camera_active = 'webrtc_ctx' in locals() and webrtc_ctx.state.playing
    indicator_class = "live-indicator active" if camera_active else "live-indicator"
    
    st.markdown(f"""
        <h3><span class="{indicator_class}"></span>📊 Live PPE Statistics</h3>
    """, unsafe_allow_html=True)
    
    # Manual refresh button
    if st.button("🔄 Refresh Stats", key="refresh_stats", help="Force refresh statistics"):
        st.rerun()
    
    # Auto-refresh mechanism for live stats
    if 'webrtc_ctx' in locals() and webrtc_ctx.state.playing:
        # Add auto-refresh when camera is active
        import time
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        # Auto-refresh every 0.3 seconds when camera is active for more responsive updates
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 0.3:
            st.session_state.last_refresh = current_time
            st.rerun()
    
    # Live metrics display - Use both webrtc transformer and session state
    active_transformer = None
    if 'webrtc_ctx' in locals() and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        active_transformer = webrtc_ctx.video_processor
    elif 'active_transformer' in st.session_state:
        active_transformer = st.session_state.active_transformer
    elif 'transformer' in st.session_state:
        active_transformer = st.session_state.transformer
    
    if active_transformer and hasattr(active_transformer, 'current_detections'):
        # Force fresh read of detections to avoid caching issues
        detections = getattr(active_transformer, 'current_detections', [])
        
        # Only show detailed statistics if there are actual detections
        if detections and len(detections) > 0:
            
            # Calculate and show compliance
            compliance = calculate_compliance_percentage(detections)
            
            # Show compliance alert or success - use consistent threshold
            is_non_compliant = compliance < 50
            if is_non_compliant:
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
            
            # Count detections for PPE-related classes only (exclude Safety Cone)
            ppe_counts = {
                "Person": 0,
                "Helmet": 0,
                "Mask": 0,
                "Safety Vest": 0,
                "NO-Helmet": 0,
                "NO-Mask": 0,
                "NO-Safety Vest": 0
            }
            
            # Process each detection
            for detection in detections:
                raw_class = detection.get('class', '')
                display_name = get_display_name(raw_class)
                
                # Only count PPE-related classes (ignore Vehicle and Machinery)
                if display_name in ppe_counts:
                    ppe_counts[display_name] += 1
            
            # Display detailed statistics
            st.markdown("**Detection Results:**")
            
            # Define icons and categories for positive PPE only (same as video.py)
            ppe_categories = [
                ("Helmet", "🪖", "positive"),
                ("Mask", "😷", "positive"),
                ("Safety Vest", "🦺", "positive")
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
            
            # Add audio alert if enabled and compliance is low (same threshold as alert)
            if st.session_state.get('audio_enabled', False) and is_non_compliant:
                audio_bytes, _ = load_audio_bytes()
                if audio_bytes:
                    import base64
                    # Create hidden audio element with loop for continuous play
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                    
                    st.markdown(f"""
                    <audio autoplay loop style="display: none;">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                    """, unsafe_allow_html=True)
        else:
            # Check if camera is actually active
            camera_playing = 'webrtc_ctx' in locals() and webrtc_ctx.state.playing
            if camera_playing:
                # Camera is active but no detections yet
                st.markdown("""
                <div style="text-align: center; padding: 2rem; color: #666;">
                    📹 <strong>Camera Active</strong><br>
                    <em>Waiting for detections...</em>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Camera not active
                st.markdown("""
                <div style="text-align: center; padding: 2rem; color: #666;">
                    📹 <strong>Camera Not Active</strong><br>
                    <em>Waiting for detections to compute stats...</em>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📷 Start camera stream to see live statistics")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6c757d;padding:1rem;'>🛡️ Smart Safety Monitor - Live PPE Detection System</div>", unsafe_allow_html=True)