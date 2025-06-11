import streamlit as st
import time
import os
import subprocess
import json
import base64
import sys
import tempfile
import requests
from typing import List, Tuple
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import (
    ROBOFLOW_API_KEY, 
    ROBOFLOW_API_URL, 
    API_MODELS,
    MODEL_ID_DEFAULT,
    CONF_THRESH_DEFAULT, 
    OVERLAP_THRESH_DEFAULT, 
    DETECTION_INTERVAL_DEFAULT
)

# Import optimized YOLO model manager
try:
    from utils.yolo_model_manager import run_optimized_local_inference, check_yolo_dependencies
    YOLO_AVAILABLE = check_yolo_dependencies()
except ImportError:
    YOLO_AVAILABLE = False
    run_optimized_local_inference = None

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

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def decode_image_from_base64(img_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    img_data = base64.b64decode(img_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def run_local_inference(image: np.ndarray, conf_thresh: float) -> Tuple[List, np.ndarray]:
    """Run local YOLO inference using HTTP server for optimal live performance"""
    try:
        if not LOCAL_MODEL_PATH:
            return [], image
        
        # Try HTTP server first (much faster for live detection)
        try:
            return run_http_inference(image, conf_thresh)
        except Exception:
            # Fallback to subprocess if server not available
            return run_subprocess_inference(image, conf_thresh)
        
    except Exception:
        # Silent error handling for live detection
        return [], image

def run_http_inference(image: np.ndarray, conf_thresh: float) -> Tuple[List, np.ndarray]:
    """Fast HTTP-based inference using persistent YOLO server"""
    try:
        # Encode image to base64
        image_b64 = encode_image_to_base64(image)
        
        # Prepare request data
        data = {
            "image_b64": image_b64,
            "conf_thresh": conf_thresh
        }
        
        # Send request to local YOLO server with very short timeout for live video
        response = requests.post(
            "http://127.0.0.1:8888/predict",
            json=data,
            timeout=0.5  # Very aggressive timeout for live detection
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                # Decode annotated image
                annotated = decode_image_from_base64(result["annotated_image"])
                return result["detections"], annotated
        
        return [], image
        
    except Exception as e:
        raise e  # Re-raise to trigger fallback

def run_subprocess_inference(image: np.ndarray, conf_thresh: float) -> Tuple[List, np.ndarray]:
    """Optimized subprocess inference for live detection"""
    try:
        # Encode image to base64
        image_b64 = encode_image_to_base64(image)
        
        # Get paths
        script_path = os.path.join(os.path.dirname(__file__), "..", "utils", "local_yolo_inference.py")
        
        # Look for separate YOLO environment
        yolo_env_path = os.path.join(os.path.dirname(__file__), "..", "yolo_env")
        yolo_python = os.path.join(yolo_env_path, "bin", "python")
        
        # Choose Python executable
        if os.path.exists(yolo_python):
            python_executable = yolo_python
        else:
            python_executable = sys.executable
        
        # Use temporary file to avoid "Argument list too long" error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.b64', delete=False) as temp_file:
            temp_file.write(image_b64)
            temp_file_path = temp_file.name
        
        try:
            # Run subprocess with very aggressive timeout for live detection
            cmd = [python_executable, script_path, LOCAL_MODEL_PATH, temp_file_path, str(conf_thresh)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5)  # Ultra-short timeout
            
            if result.returncode != 0:
                return [], image
            
            # Parse result
            output = json.loads(result.stdout)
            if "error" in output:
                return [], image
            
            # Decode annotated image
            annotated = decode_image_from_base64(output["annotated_image"])
            
            return output["detections"], annotated
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass  # Ignore cleanup errors
        
    except subprocess.TimeoutExpired:
        # Return empty result on timeout - don't freeze the video
        return [], image
    except Exception as e:
        return [], image

st.set_page_config(
    page_title="PPE Detection – Smart Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and video size control
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
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #333333 !important;
    }
    
    .metric-card h4 {
        color: #333333 !important;
        margin-bottom: 1rem;
    }
    
    .metric-card p {
        color: #555555 !important;
        margin: 0.5rem 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .status-active {
        background: linear-gradient(45deg, #28a745, #20c997);
    }
    
    .status-inactive {
        background: linear-gradient(45deg, #6c757d, #495057);
    }
    
    /* Control video size */
    .stVideo > div {
        max-width: 640px !important;
        max-height: 480px !important;
        margin: 0 auto;
    }
    
    /* WebRTC video container */
    div[data-testid="stWebRtcVideo"] > div {
        max-width: 640px !important;
        max-height: 480px !important;
        margin: 0 auto !important;
    }
    
    .video-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background: #f8f9fa;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1 style="color:white;margin:0;font-size:2.5rem;">🛡️ Smart Safety Monitor</h1>
    <p style="color:rgba(255,255,255,0.9);margin:0.5rem 0 0 0;font-size:1.2rem;">
        Real-time PPE Detection with AI-powered Computer Vision
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("⚙️ Detection Settings")
    
    # Model selection
    model_options = API_MODELS.copy()
    if LOCAL_MODEL_AVAILABLE:
        model_options.append("best.pt (Local)")
    
    model_id = st.selectbox(
        "🎯 Model Selection", 
        model_options,
        index=0
    )
    
    st.markdown("#### 🎚️ Detection Parameters")
    conf_thresh = st.slider(
        "Confidence Threshold", 
        0.0, 1.0, CONF_THRESH_DEFAULT, 0.05,
        help="Minimum confidence for detections"
    )
    overlap_thresh = st.slider(
        "IoU Threshold (NMS)", 
        0.0, 1.0, OVERLAP_THRESH_DEFAULT, 0.05,
        help="Non-maximum suppression threshold"
    )
    detection_interval = st.number_input(
        "⏱️ Detection Interval (seconds)",
        min_value=0.1,
        max_value=30.0,
        value=DETECTION_INTERVAL_DEFAULT,
        step=0.1,
        format="%0.1f",
        help="Time between API calls to save quota"
    )
    
    st.markdown("#### 📹 Camera Control")
    start_webcam = st.toggle("🎥 Enable Live Detection", value=False)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    **💡 Tips:**
    - Higher confidence = fewer false positives
    - Lower interval = more frequent detection
    - Adjust thresholds for optimal performance
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    if start_webcam:
        st.markdown("### 📺 Live Video Feed")
    else:
        st.markdown("### 📺 Video Feed")
        st.info("👆 Enable live detection in the sidebar to start monitoring")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Detection Status")
    
    # Status metrics
    status_class = "status-active" if start_webcam else "status-inactive"
    status_text = "🟢 ACTIVE" if start_webcam else "🔴 INACTIVE"
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>System Status</h4>
        <span class="status-badge {status_class}">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>Current Settings</h4>
        <p><strong>Model:</strong> {model_id}</p>
        <p><strong>Confidence:</strong> {conf_thresh:.2f}</p>
        <p><strong>IoU Threshold:</strong> {overlap_thresh:.2f}</p>
        <p><strong>Interval:</strong> {detection_interval:.1f}s</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detection classes info
    st.markdown("""
    <div class="metric-card">
        <h4>🎯 Detection Classes</h4>
        <p>🦺 Safety Vest</p>
        <p>⛑️ Hard Hat</p>
        <p>🔧 Safety Equipment</p>
    </div>
    """, unsafe_allow_html=True)

# Utility: NMS
def apply_nms(preds: List[dict], conf_thresh: float, overlap_thresh: float) -> List[dict]:
    boxes, confidences, raw_indices = [], [], []
    for idx, p in enumerate(preds):
        conf = p["confidence"]
        if conf >= conf_thresh:
            cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            boxes.append([x1, y1, int(w), int(h)])
            confidences.append(float(conf))
            raw_indices.append(idx)
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, overlap_thresh)
    kept = []
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        kept.append(preds[raw_indices[i]])
    return kept

# Roboflow client (cached)
@st.cache_resource(show_spinner=False)
def get_rf_client():
    return InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY)

# Video transformer
class PPETransformer(VideoTransformerBase):
    def __init__(self, api_client, model_id, conf_thresh, overlap_thresh, detection_interval):
        self.client = api_client
        self.model_id = model_id
        self.conf_thresh = conf_thresh
        self.overlap_thresh = overlap_thresh
        self.detection_interval = detection_interval
        self._last_preds: List[dict] = []
        self._last_time = 0.0

    def _infer(self, image: np.ndarray):
        current = time.time()
        if current - self._last_time >= self.detection_interval:
            try:
                if self.model_id == "best.pt (Local)":
                    # Use local model via subprocess
                    detections, _ = run_local_inference(image, self.conf_thresh)
                    self._last_preds = detections
                else:
                    # Use Roboflow API - ensure we send BGR image (Roboflow expects BGR)
                    response = self.client.infer(image, model_id=self.model_id)
                    self._last_preds = response.get("predictions", [])
                self._last_time = current
            except Exception as e:
                # Draw error message on a copy to avoid color conflicts
                error_img = image.copy()
                cv2.putText(
                    error_img,
                    f"API error: {str(e)[:50]}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),  # Red text in BGR
                    2,
                )
                # Don't return the error image, just log it
                print(f"Inference Error: {e}")
        return self._last_preds

    def transform(self, frame):
        # Get the image array - streamlit-webrtc provides BGR format
        img = frame.to_ndarray(format="bgr24")
        
        # Make a copy to avoid modifying the original
        display_img = img.copy()
        
        # Get predictions
        preds = self._infer(img)
        preds = apply_nms(preds, self.conf_thresh, self.overlap_thresh)

        # Draw bounding boxes and labels
        for p in preds:
            x1 = int(p["x"] - p["width"] / 2)
            y1 = int(p["y"] - p["height"] / 2)
            x2 = int(p["x"] + p["width"] / 2)
            y2 = int(p["y"] + p["height"] / 2)
            
            # Draw green rectangle (BGR: B=0, G=255, R=0)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{p['class']} ({p['confidence']:.2f})"
            cv2.putText(
                display_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),  # Green text in BGR
                2,
            )
        
        # Return the BGR image directly (as per working friend's implementation)
        # Streamlit webrtc can handle BGR format correctly
        return display_img

# Stream launch
if start_webcam:
    with col1:
        rf_client = get_rf_client()
        webrtc_streamer(
            key="ppe-detection-stream",
            video_processor_factory=lambda: PPETransformer(
                api_client=rf_client,
                model_id=model_id,
                conf_thresh=conf_thresh,
                overlap_thresh=overlap_thresh,
                detection_interval=detection_interval,
            ),
            media_stream_constraints={
                "video": {"width": 640, "height": 480}, 
                "audio": False
            },
            async_processing=True,
        )
else:
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #dee2e6;">
            <h3 style="color: #6c757d;">🎥 Camera Ready</h3>
            <p style="color: #6c757d; font-size: 1.1rem;">Enable live detection to start monitoring PPE compliance</p>
            <p style="color: #6c757d;">Adjust settings in the sidebar before starting</p>
        </div>
        """, unsafe_allow_html=True)

