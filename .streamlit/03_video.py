import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import sys
from typing import List, Dict, Tuple
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
        return model
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="PPE Video Analysis – Smart Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for consistent styling
st.markdown("""
<style>
    .main-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);padding: 2rem;border-radius: 15px;margin-bottom: 2rem;text-align: center;box-shadow: 0 8px 32px rgba(0,0,0,0.1);}
    .metric-card {background: white;padding: 1.5rem;border-radius: 10px;box-shadow: 0 4px 16px rgba(0,0,0,0.1);border-left: 4px solid #667eea;margin: 1rem 0;color: #333333 !important;}
    .metric-card h4 {color: #333333 !important;margin-bottom: 1rem;}
    .metric-card p {color: #555555 !important;margin: 0.5rem 0;}
    .progress-container {background: #f0f2f6;border-radius: 10px;padding: 1rem;margin: 1rem 0;}
    .compliance-alert {
        background: #ff4757;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        animation: blink 1s infinite;
        margin: 1rem 0;
    }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    .stats-panel {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .ppe-count {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
    .ppe-count:last-child {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color:white;margin:0;font-size:2.5rem;">🎥 Local YOLO Video Analysis</h1>
    <p style="color:rgba(255,255,255,0.9);margin:0.5rem 0 0 0;font-size:1.1rem;">Process video files with local PPE detection model</p>
</div>
""", unsafe_allow_html=True)

# YOLO class mapping
YOLO_CLASSES = {
    0: "Hardhat",
    1: "Mask", 
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "Machinery",
    9: "Vehicle"
}

# PPE categories to track for compliance
PPE_CATEGORIES = [0, 1, 2, 3, 4, 5, 7]  # Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Vest

# Color mapping for classes
def get_color_for_class(name: str) -> Tuple[int,int,int]:
    color_map = {
        "hardhat": (0,255,0),
        "mask": (0,255,128),
        "no-hardhat": (0,0,255),
        "no-mask": (255,0,255),
        "no-safety vest": (255,165,0),
        "person": (255,0,0),
        "safety vest": (0,255,255),
        "safety cone": (255,255,0),
        "machinery": (128,128,128),
        "vehicle": (64,64,64)
    }
    return color_map.get(name.lower(), (128,128,128))

def calculate_compliance_percentage(ppe_counts: Dict[str, int]) -> float:
    """Calculate PPE compliance percentage"""
    total_people = ppe_counts.get("Person", 0)
    if total_people == 0:
        return 100.0  # No people detected, assume compliant
    
    # Count violations
    violations = (
        ppe_counts.get("NO-Hardhat", 0) + 
        ppe_counts.get("NO-Mask", 0) + 
        ppe_counts.get("NO-Safety Vest", 0)
    )
    
    # Calculate compliance (fewer violations = higher compliance)
    compliance = max(0, (total_people * 3 - violations) / (total_people * 3) * 100)
    return compliance

def run_yolo_inference(image: np.ndarray, conf_thresh: float, model) -> Tuple[List, np.ndarray]:
    """Run YOLO inference on image"""
    try:
        if model is None:
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
                    class_name = YOLO_CLASSES.get(class_id, f"class_{class_id}")
                    
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

def process_video_with_local_model(input_path: str, output_path: str, conf_thresh: float, model, progress_bar, status_text) -> Dict:
    """Process video using local YOLO model"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    total_detections = 0
    total_violations = 0
    
    # PPE category counts
    ppe_counts = {
        "Hardhat": 0,
        "Mask": 0, 
        "NO-Hardhat": 0,
        "NO-Mask": 0,
        "NO-Safety Vest": 0,
        "Person": 0,
        "Safety Vest": 0
    }
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        detections, annotated_frame = run_yolo_inference(frame, conf_thresh, model)
        
        # Update statistics and PPE counts
        total_detections += len(detections)
        for detection in detections:
            class_name = detection['class']
            if class_name in ppe_counts:
                ppe_counts[class_name] += 1
            if 'no-' in class_name.lower():
                total_violations += 1
        
        # Write frame
        out.write(annotated_frame)
        
        frame_count += 1
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps_current = frame_count / elapsed_time
            eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
            status_text.text(f"Processing frame {frame_count}/{total_frames} | FPS: {fps_current:.1f} | ETA: {eta:.1f}s")
    
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    compliance_percentage = calculate_compliance_percentage(ppe_counts)
    
    return {
        "success": True,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "total_violations": total_violations,
        "processing_time": processing_time,
        "avg_fps": frame_count / processing_time if processing_time > 0 else 0,
        "ppe_counts": ppe_counts,
        "compliance_percentage": compliance_percentage
    }

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Processing Parameters")
    
    # Simple model check without detailed info
    if not LOCAL_MODEL_AVAILABLE or not LOCAL_MODEL_PATH:
        st.error("❌ Local model not found")
        st.stop()
    
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, float(CONF_THRESH_DEFAULT), 0.05)

# Load model
model = load_yolo_model()
if model is None:
    st.error("❌ Failed to load YOLO model. Please check the model file.")
    st.stop()

# Main content
st.markdown("### Upload Video for Local PPE Analysis")
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"], help="Upload a video file showing workers with or without PPE")

if video_file:
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
        tmp_input.write(video_file.read())
        input_path = tmp_input.name
    
    # Show video info
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📐 Resolution", f"{width}x{height}")
        with col2:
            st.metric("⏱️ Duration", f"{duration:.1f}s")
        with col3:
            st.metric("🎬 Frames", frame_count)
        with col4:
            st.metric("📊 FPS", f"{fps:.1f}")
    
    # Process video button
    if st.button("🚀 Process Video", type="primary"):
        # Create output path
        with tempfile.NamedTemporaryFile(delete=False, suffix='_processed.mp4') as tmp_output:
            output_path = tmp_output.name
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔍 Processing video with local YOLO model..."):
            result = process_video_with_local_model(
                input_path, output_path, conf_thresh, model, progress_bar, status_text
            )
        
        if result.get("success"):
            st.success("✅ Video processing completed!")
            
            # Create layout with main results and PPE statistics panel
            main_col, stats_col = st.columns([2, 1])
            
            with main_col:
                # Show main results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Total Detections", result["total_detections"])
                with col2:
                    st.metric("⚠️ Violations", result["total_violations"])
                with col3:
                    st.metric("⏱️ Processing Time", f"{result['processing_time']:.1f}s")
                with col4:
                    st.metric("🚀 Avg FPS", f"{result['avg_fps']:.1f}")
                
                # Download processed video
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f.read(),
                        file_name="ppe_processed_video.mp4",
                        mime="video/mp4"
                    )
            
            with stats_col:
                # PPE Statistics Panel
                st.markdown("### 📊 PPE Detection Statistics")
                
                # Compliance percentage
                compliance = result["compliance_percentage"]
                if compliance < 60:
                    st.markdown(f"""
                    <div class="compliance-alert">
                        🚨 COMPLIANCE ALERT 🚨<br>
                        {compliance:.1f}% Compliance
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ Compliance: {compliance:.1f}%")
                
                # PPE counts panel
                st.markdown('<div class="stats-panel">', unsafe_allow_html=True)
                st.markdown("**PPE Category Counts:**")
                
                ppe_counts = result["ppe_counts"]
                for category in ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Vest"]:
                    count = ppe_counts.get(category, 0)
                    icon = "🟢" if "NO-" not in category else "🔴"
                    if category == "Person":
                        icon = "👤"
                    
                    st.markdown(f"""
                    <div class="ppe-count">
                        <span>{icon} {category}</span>
                        <span><strong>{count}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6c757d;padding:1rem;'>🛡️ Smart Safety Monitor - Local YOLO Video Analysis</div>", unsafe_allow_html=True)
