import streamlit as st
import cv2
import numpy as np
from typing import List, Tuple
import tempfile
import os
import sys
from ultralytics import YOLO

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import CONF_THRESH_DEFAULT

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
        st.success(f"✅ YOLO model loaded from {LOCAL_MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        return None

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
    <h1 style="color:white;margin:0;font-size:2.5rem;">🖼️ Local YOLO PPE Analysis</h1>
    <p style="color:rgba(255,255,255,0.9);margin:0.5rem 0 0 0;font-size:1.1rem;">Upload an image for local PPE detection and compliance analysis</p>
</div>
""", unsafe_allow_html=True)

# Color mapping for classes
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

def run_yolo_inference(image: np.ndarray, conf_thresh: float, model) -> Tuple[List, np.ndarray]:
    """Run YOLO inference on image"""
    try:
        if model is None:
            st.error("Model not loaded")
            return [], image
        
        # Convert BGR to RGB for YOLO
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
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
        
        return detections, annotated
        
    except Exception as e:
        st.error(f"Inference error: {e}")
        return [], image

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Detection Parameters")
    
    # Show model info
    if LOCAL_MODEL_AVAILABLE and LOCAL_MODEL_PATH:
        st.success("🎯 Local YOLO Model Ready")
        st.info(f"📁 Model: {os.path.basename(LOCAL_MODEL_PATH)}")
    else:
        st.error("❌ Local model not found")
        st.stop()
    
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Load model
model = load_yolo_model()
if model is None:
    st.error("❌ Failed to load YOLO model. Please check the model file.")
    st.stop()

# Main content
st.markdown("### Upload Image for Local PPE Analysis")
img_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp"], help="Upload an image showing workers with or without PPE")
if img_file:
    # Read image
    data = img_file.read()
    img_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Show original
    st.image(image, channels="BGR", caption="Original Image", use_container_width=True)

    # Analyze
    with st.spinner("🔍 Detecting PPE with local YOLO model..."):
        detections, annotated = run_yolo_inference(image, conf_thresh, model)

    # Show annotated
    st.image(annotated, channels="BGR", caption=f"Detections: {len(detections)}", use_container_width=True)

    # Metrics
    violations = sum(1 for d in detections if 'no-' in d['class'].lower())
    compliance = ((len(detections) - violations) / len(detections) * 100) if detections else 0
    helmets = sum(1 for d in detections if d['class'].lower() in ['helmet', 'hard-hat'])
    vests = sum(1 for d in detections if d['class'].lower() == 'safety-vest')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🪖 Helmets", helmets)
    with col2:
        st.metric("🦺 Vests", vests)
    with col3:
        st.metric("⚠️ Violations", violations)
    with col4:
        st.metric("✅ Compliance", f"{compliance:.1f}%")

    # Download annotated image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        cv2.imwrite(tmp.name, annotated)
        with open(tmp.name, 'rb') as f:
            btn = st.download_button(
                label="📥 Download Annotated Image",
                data=f.read(),
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
st.markdown("<div style='text-align:center;color:#6c757d;padding:1rem;'>🛡️ Smart Safety Monitor - Local YOLO PPE Analysis</div>", unsafe_allow_html=True)