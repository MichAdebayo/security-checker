import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from typing import List, Tuple
import tempfile
import os
import subprocess
import json
import base64
import sys

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import ROBOFLOW_API_KEY, ROBOFLOW_API_URL, API_MODELS

# Import optimized YOLO model manager
try:
    from utils.yolo_model_manager import run_optimized_local_inference, check_yolo_dependencies
    YOLO_AVAILABLE = check_yolo_dependencies()
except ImportError:
    YOLO_AVAILABLE = False
    run_optimized_local_inference = None

# Import optimized YOLO model manager  
try:
    from yolo_model_manager import run_optimized_local_inference, check_yolo_dependencies
    YOLO_AVAILABLE = check_yolo_dependencies()
except ImportError:
    YOLO_AVAILABLE = False
    run_optimized_local_inference = None

# Check if local model file exists
def check_local_model_available():
    """Check if the local YOLO model file exists (independent of YOLO dependencies)"""
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

# Page configuration
st.set_page_config(
    page_title="PPE Image Analysis – Smart Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Copy CSS for consistent styling
st.markdown("""
<style>
    .main-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);padding: 2rem;border-radius: 15px;margin-bottom: 2rem;text-align: center;box-shadow: 0 8px 32px rgba(0,0,0,0.1);}
    .metric-card {background: white;padding: 1.5rem;border-radius: 10px;box-shadow: 0 4px 16px rgba(0,0,0,0.1);border-left: 4px solid #667eea;margin: 1rem 0;color: #333333 !important;}
    .metric-card h4 {color: #333333 !important;margin-bottom: 1rem;}
    .metric-card p {color: #555555 !important;margin: 0.5rem 0;}
    .status-badge {display: inline-block;padding: 0.5rem 1rem;border-radius: 20px;color: white;font-weight: bold;margin: 0.5rem 0;}
    .status-safe {background: linear-gradient(45deg, #28a745, #20c997);} .status-warning {background: linear-gradient(45deg, #ffc107, #fd7e14);} .status-danger {background: linear-gradient(45deg, #dc3545, #e74c3c);}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color:white;margin:0;font-size:2.5rem;">🖼️ Image PPE Analysis</h1>
    <p style="color:rgba(255,255,255,0.9);margin:0.5rem 0 0 0;font-size:1.1rem;">Upload an image for PPE detection and compliance analysis</p>
</div>
""", unsafe_allow_html=True)

# Hidden API credentials
@st.cache_resource(show_spinner=False)
def get_rf_client() -> InferenceHTTPClient:
    return InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)

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
    """Run local YOLO inference using optimized approach or subprocess fallback"""
    try:
        if not LOCAL_MODEL_PATH:
            st.error("Local model path not available")
            return [], image
        
        # Always use subprocess approach to avoid NumPy compatibility issues
        # The optimized approach has NumPy 2.x conflicts with YOLO dependencies
        return run_subprocess_inference(image, conf_thresh)
        
    except Exception as e:
        st.error(f"Local inference failed: {str(e)}")
        return [], image

def run_subprocess_inference(image: np.ndarray, conf_thresh: float) -> Tuple[List, np.ndarray]:
    """Fallback subprocess inference for when YOLO dependencies aren't in main environment"""
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
            st.info("🐍 Using separate YOLO environment")
        else:
            python_executable = sys.executable
            st.warning("⚠️ YOLO environment not found, using current environment")
        
        # Use temporary file to avoid "Argument list too long" error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.b64', delete=False) as temp_file:
            temp_file.write(image_b64)
            temp_file_path = temp_file.name
        
        try:
            # Run subprocess with temp file path instead of large string
            cmd = [python_executable, script_path, LOCAL_MODEL_PATH, temp_file_path, str(conf_thresh)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                st.error(f"Local inference failed: {result.stderr}")
                return [], image
            
            # Parse result
            output = json.loads(result.stdout)
            if "error" in output:
                st.error(f"Local inference error: {output['error']}")
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
        st.error("Local inference timed out")
        return [], image
    except Exception as e:
        st.error(f"Local inference failed: {str(e)}")
        return [], image

# Non-Maximum Suppression utility
from typing import List, Dict

def apply_nms(preds: List[Dict], conf_thresh: float, overlap_thresh: float) -> List[Dict]:
    boxes, confidences, idxs = [], [], []
    for i, p in enumerate(preds):
        if p["confidence"] >= conf_thresh:
            cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
            x1, y1 = int(cx - w/2), int(cy - h/2)
            boxes.append([x1, y1, int(w), int(h)])
            confidences.append(float(p["confidence"]))
            idxs.append(i)
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, overlap_thresh)
    result = []
    for k in keep:
        idx = k[0] if isinstance(k, (list, tuple, np.ndarray)) else k
        result.append(preds[idxs[idx]])
    return result

# Color mapping for classes
from typing import Tuple

def get_color_for_class(name: str) -> Tuple[int,int,int]:
    return {
        "hard-hat": (0,255,0),
        "helmet": (0,255,0),
        "safety-vest": (0,255,255),
        "person": (255,0,0),
        "no-hard-hat": (0,0,255),
        "no-helmet": (0,0,255),
        "no-safety-vest": (0,165,255)
    }.get(name.lower(), (128,128,128))

def analyze_image(image: np.ndarray, conf_thresh: float, overlap_thresh: float, model_id: str):
    if model_id == "best.pt (Local)":
        # Use local model via subprocess
        return run_local_inference(image, conf_thresh)
    else:
        # Use Roboflow API
        client = get_rf_client()
        resp = client.infer(image, model_id=model_id)
        preds = resp.get("predictions", []) # type: ignore
        filtered = apply_nms(preds, conf_thresh, overlap_thresh)
        out = image.copy()
        for p in filtered:
            x1 = int(p["x"] - p["width"]/2)
            y1 = int(p["y"] - p["height"]/2)
            x2 = int(p["x"] + p["width"]/2)
            y2 = int(p["y"] + p["height"]/2)
            col = get_color_for_class(p["class"])
            cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
            label = f"{p['class']} ({p['confidence']:.2f})"
            sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(out, (x1,y1-sz[1]-8), (x1+sz[0], y1), col, -1)
            cv2.putText(out, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return filtered, out

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Detection Parameters")
    # Model selection
    model_options = API_MODELS.copy()
    if LOCAL_MODEL_AVAILABLE:
        model_options.append("best.pt (Local)")
    
    model_id = st.selectbox(
        "🎯 Model Selection",
        model_options,
        index=1,
        help="Choose the PPE detection model"
    )
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    overlap_thresh = st.slider("IoU Threshold", 0.0, 1.0, 0.3, 0.05)

# Main content
st.markdown("### Upload Image for PPE Analysis")
img_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp"], help="Upload an image showing workers with or without PPE")
if img_file:
    # Read image
    data = img_file.read()
    img_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Show original
    st.image(image, channels="BGR", caption="Original Image", use_container_width=True)

    # Analyze
    with st.spinner("🔍 Detecting PPE..."):
        detections, annotated = analyze_image(image, conf_thresh, overlap_thresh, model_id)

    # Show annotated
    st.image(annotated, channels="BGR", caption=f"Detections: {len(detections)}", use_container_width=True)

    # Metrics
# sourcery skip: simplify-constant-sum
    violations = sum(1 for d in detections if 'no-' in d['class'].lower())
    compliance = ((len(detections) - violations) / len(detections) * 100) if detections else 0
    helmets = sum(1 for d in detections if d['class'].lower() == 'helmet')
    vests = sum(1 for d in detections if d['class'].lower() == 'safety-vest')
    st.markdown(f"**🪖 Helmets:** {helmets}    **🦺 Vests:** {vests}    **⚠️ Violations:** {violations}    **✅ Compliance:** {compliance:.1f}%")

    # Download annotated image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        cv2.imwrite(tmp.name, annotated)
        tmp.seek(0)
        btn = st.download_button(
            label="📥 Download Annotated Image",
            data=tmp.read(),
            file_name="ppe_analysis.png",
            mime="image/png"
        )
        os.unlink(tmp.name)

    # Details expander
    if detections:
        with st.expander("🔍 Detection Details"):
            for i, d in enumerate(detections, start=1):
                st.write(f"{i}. **{d['class']}** at ({int(d['x'])}, {int(d['y'])}), Confidence: {d['confidence']:.2f}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6c757d;padding:1rem;'>🛡️ Smart Safety Monitor - Image PPE Analysis</div>", unsafe_allow_html=True)