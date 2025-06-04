import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from typing import List, Dict, Tuple
from inference_sdk import InferenceHTTPClient
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(
    page_title="PPE Video Analysis – Smart Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
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
    
    .status-safe {
        background: linear-gradient(45deg, #28a745, #20c997);
    }
    
    .status-warning {
        background: linear-gradient(45deg, #ffc107, #fd7e14);
    }
    
    .status-danger {
        background: linear-gradient(45deg, #dc3545, #e74c3c);
    }
    
    .upload-area {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .analysis-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .progress-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1 style="color:white;margin:0;font-size:2.5rem;">🎬 Video PPE Analysis</h1>
    <p style="color:rgba(255,255,255,0.9);margin:0.5rem 0 0 0;font-size:1.2rem;">
        Upload videos for comprehensive safety equipment detection and compliance analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Configuration
API_KEY_DEFAULT = "mDauQAfDrFWieIsSqti6"  # Hidden from UI
MODEL_ID_DEFAULT = "pbe-detection/4"
CONF_THRESH_DEFAULT = 0.25
OVERLAP_THRESH_DEFAULT = 0.3

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None
if 'show_realtime' not in st.session_state:
    st.session_state.show_realtime = False

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("⚙️ Analysis Settings")
    
    model_id = st.selectbox(
        "🎯 Model Selection", 
        ["ppe-factory-bmdcj/2", "pbe-detection/4"],
        index=0,
        help="Choose the PPE detection model"
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
    
    st.markdown("#### 📊 Analysis Options")
    
    # Frame skip only for batch analysis
    st.markdown("**🎬 Batch Analysis Settings:**")
    frame_skip = st.slider(
        "Frame Skip Interval (Full Analysis only)",
        1, 30, 5,
        help="For Full Analysis: Process every Nth frame (higher = faster but less accurate)"
    )
    
    save_annotated_video = st.checkbox(
        "💾 Save Annotated Video",
        value=True,
        help="Generate video with detection annotations (Full Analysis only)"
    )
    
    detailed_analysis = st.checkbox(
        "📈 Detailed Statistics",
        value=True,
        help="Generate comprehensive compliance reports"
    )
    
    st.markdown("**🎬 Smooth Detection Settings:**")
    st.info("Smooth Detection processes every frame for natural playback")
    st.markdown("- **Processing**: Every frame analyzed")  
    st.markdown("- **Speed**: Optimized for real-time viewing")
    st.markdown("- **Quality**: Maximum detection accuracy")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    **💡 Analysis Tips:**
    - **⚡ Instant Preview:** Quick first-frame analysis
    - **🎬 Smooth Detection:** Natural playback, every frame processed
    - **📊 Full Analysis:** Batch processing with frame skip for speed
    - Upload MP4, AVI, or MOV files
    - Optimal resolution: 640x480 to 1920x1080
    - Lower confidence = more detections (may include false positives)
    """)

# Utility functions
def apply_nms(preds: List[dict], conf_thresh: float, overlap_thresh: float) -> List[dict]:
    """Apply Non-Maximum Suppression to filter predictions"""
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

@st.cache_resource(show_spinner=False)
def get_rf_client():
    """Get cached Roboflow client"""
    return InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY_DEFAULT)

def analyze_frame(client, frame: np.ndarray, model_id: str, conf_thresh: float, overlap_thresh: float) -> Tuple[List[dict], np.ndarray]:
    """Analyze a single frame for PPE detection"""
    try:
        # API call for detection
        response = client.infer(frame, model_id=model_id)
        predictions = response.get("predictions", [])
        filtered_preds = apply_nms(predictions, conf_thresh, overlap_thresh)
        
        # Draw annotations on frame
        annotated_frame = frame.copy()
        for pred in filtered_preds:
            x1 = int(pred["x"] - pred["width"] / 2)
            y1 = int(pred["y"] - pred["height"] / 2)
            x2 = int(pred["x"] + pred["width"] / 2)
            y2 = int(pred["y"] + pred["height"] / 2)
            
            # Color coding based on safety equipment
            color = get_color_for_class(pred["class"])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{pred['class']} ({pred['confidence']:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return filtered_preds, annotated_frame
    
    except Exception as e:
        st.error(f"Frame analysis error: {str(e)}")
        return [], frame

def get_color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Get BGR color for different PPE classes"""
    colors = {
        "hard-hat": (0, 255, 0),      # Green
        "helmet": (0, 255, 0),        # Green alias for helmet
        "safety-vest": (0, 255, 255), # Yellow
        "person": (255, 0, 0),        # Blue
        "no-hard-hat": (0, 0, 255),   # Red
        "no-helmet": (0, 0, 255),     # Red alias for no-helmet
        "no-safety-vest": (0, 165, 255)  # Orange
    }
    return colors.get(class_name.lower(), (128, 128, 128))  # Gray default

def calculate_compliance_metrics(all_detections: List[Dict]) -> Dict:
    """Calculate safety compliance metrics from all detections"""
    frame_stats = defaultdict(lambda: {"persons": 0, "hard_hats": 0, "safety_vests": 0, "violations": 0})
    
    for detection in all_detections:
        frame_num = detection["frame"]
        class_name = detection["class"].lower()
        
        if "person" in class_name:
            frame_stats[frame_num]["persons"] += 1
        elif "hard-hat" in class_name or "helmet" in class_name:
            frame_stats[frame_num]["hard_hats"] += 1
        elif "safety-vest" in class_name or "vest" in class_name:
            frame_stats[frame_num]["safety_vests"] += 1
        elif "no-" in class_name:
            frame_stats[frame_num]["violations"] += 1
    
    # Calculate overall metrics
    total_frames = len(frame_stats)
    compliant_frames = 0
    total_persons = 0
    total_violations = 0
    
    for frame_data in frame_stats.values():
        persons = frame_data["persons"]
        hats = frame_data["hard_hats"]
        vests = frame_data["safety_vests"]
        violations = frame_data["violations"]
        
        total_persons += persons
        total_violations += violations
        
        # Frame is compliant if all persons have safety equipment
        if persons > 0 and (hats >= persons or violations == 0):
            compliant_frames += 1
    
    compliance_rate = (compliant_frames / total_frames * 100) if total_frames > 0 else 0
    
    return {
        "total_frames_analyzed": total_frames,
        "compliant_frames": compliant_frames,
        "compliance_rate": compliance_rate,
        "total_persons_detected": total_persons,
        "total_violations": total_violations,
        "frame_stats": dict(frame_stats)
    }

def process_video(uploaded_file, client, model_id: str, conf_thresh: float, overlap_thresh: float, 
                 frame_skip: int, save_annotated: bool) -> Dict:
    """Process uploaded video for PPE detection"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.unlink(video_path)
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    # Setup video writer for annotated output
    output_path = None
    writer = None
    if save_annotated:
        output_path = tempfile.mktemp(suffix='_annotated.mp4')
        # Use more compatible fourcc
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            # Try with mp4v if XVID fails
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_detections = []
    frame_num = 0
    processed_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if frame_num % frame_skip == 0:
                status_text.text(f"Analyzing frame {frame_num + 1}/{frame_count}")
                
                # Analyze frame
                detections, annotated_frame = analyze_frame(
                    client, frame, model_id, conf_thresh, overlap_thresh
                )
                
                # Store detections with frame info
                for det in detections:
                    det["frame"] = frame_num
                    det["timestamp"] = frame_num / fps
                    all_detections.append(det)
                
                # Write annotated frame
                if writer is not None and writer.isOpened():
                    writer.write(annotated_frame)
                elif save_annotated and output_path:
                    # If we want to save but writer failed, try to recreate it
                    if writer is None or not writer.isOpened():
                        fourcc = cv2.VideoWriter.fourcc(*'XVID')
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        if writer.isOpened():
                            writer.write(annotated_frame)
                
                processed_frames += 1
            else:
                # Write original frame if saving video
                if writer is not None and writer.isOpened():
                    writer.write(frame)
            
            # Update progress
            progress = (frame_num + 1) / frame_count
            progress_bar.progress(progress)
            frame_num += 1
    
    finally:
        cap.release()
        if writer is not None and writer.isOpened():
            writer.release()
        os.unlink(video_path)
        progress_bar.empty()
        status_text.empty()
    
    # Verify output file was created if we intended to save it
    if save_annotated and output_path and not os.path.exists(output_path):
        output_path = None  # Don't return invalid path
    
    # Calculate metrics
    metrics = calculate_compliance_metrics(all_detections)
    
    return {
        "detections": all_detections,
        "metrics": metrics,
        "video_info": {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
            "processed_frames": processed_frames
        },
        "annotated_video_path": output_path if save_annotated else None
    }

def run_realtime_detection(uploaded_file, model_id: str, conf_thresh: float, overlap_thresh: float):
    """Run real-time PPE detection on video frames with smooth playback"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Setup client
        client = get_rf_client()
        
        # Create container for smooth video display
        st.markdown("### 🎬 Live PPE Detection Feed")
        
        # Control panel
        col_info, col_controls = st.columns([2, 1])
        with col_info:
            st.info(f"📹 **Video:** {frame_count} frames | **Duration:** {duration:.1f}s | **FPS:** {fps}")
        
        with col_controls:
            # Process every N frames for performance
            skip_frames = st.selectbox("Processing Speed", [1, 2, 3, 5], index=1, 
                                     help="Process every Nth frame (higher = faster but less accuracy)")
        
        # Display containers
        video_container = st.container()
        progress_container = st.container()
        metrics_container = st.container()
        
        with video_container:
            frame_placeholder = st.empty()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
        
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            frames_processed = col1.empty()
            detections_count = col2.empty()
            current_violations = col3.empty()
            compliance_rate = col4.empty()
        
        # Initialize tracking variables
        all_detections = []
        frame_num = 0
        processed_count = 0
        violation_count = 0
        
        # Smooth frame processing
        frame_delay = max(0.03, 1.0 / (fps * 2))  # Adaptive delay based on original FPS
        
        # Start processing
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = (frame_num + 1) / frame_count
            progress_bar.progress(progress)
            
            # Process frame (skip some for performance)
            should_process = (frame_num % skip_frames == 0)
            
            if should_process:
                try:
                    # Analyze frame
                    detections, annotated_frame = analyze_frame(
                        client, frame, model_id, conf_thresh, overlap_thresh
                    )
                    
                    # Store detections
                    for det in detections:
                        det["frame"] = frame_num
                        det["timestamp"] = frame_num / fps
                        all_detections.append(det)
                        
                        # Count violations
                        if "no-" in det["class"].lower():
                            violation_count += 1
                    
                    processed_count += 1
                    
                    # Display annotated frame
                    frame_placeholder.image(
                        annotated_frame, 
                        channels="BGR", 
                        caption=f"🎯 Frame {frame_num + 1}/{frame_count} | Detections: {len(detections)}",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    # Show original frame if detection fails
                    frame_placeholder.image(
                        frame, 
                        channels="BGR", 
                        caption=f"⚠️ Frame {frame_num + 1}/{frame_count} | Detection Error",
                        use_container_width=True
                    )
            else:
                # Show original frame for skipped frames (smoother playback)
                frame_placeholder.image(
                    frame, 
                    channels="BGR", 
                    caption=f"📹 Frame {frame_num + 1}/{frame_count} | Skipped",
                    use_container_width=True
                )
            
            # Update live metrics
            elapsed_time = time.time() - start_time
            processing_fps = processed_count / elapsed_time if elapsed_time > 0 else 0
            current_compliance = ((processed_count - violation_count) / processed_count * 100) if processed_count > 0 else 100
            
            frames_processed.metric("Frames Processed", f"{processed_count:,}")
            detections_count.metric("Total Detections", f"{len(all_detections):,}")
            current_violations.metric("Violations Found", f"{violation_count:,}")
            compliance_rate.metric("Live Compliance", f"{current_compliance:.1f}%")
            
            # Status update
            status_placeholder.text(f"🔄 Processing at {processing_fps:.1f} FPS | {progress*100:.1f}% complete")
            
            # Smooth delay for natural playback
            time.sleep(frame_delay)
            
            frame_num += 1
            
            # Safety break for very long videos
            if frame_num > 1000:  # Process max 1000 frames for demo
                st.warning("⏱️ Reached processing limit of 1000 frames for performance")
                break
    
    finally:
        cap.release()
        os.unlink(video_path)
        
        # Store final results
        if all_detections:
            metrics = calculate_compliance_metrics(all_detections)
            st.session_state.analysis_results = {
                "detections": all_detections,
                "metrics": metrics,
                "video_info": {
                    "duration": len(all_detections) / fps if all_detections else 0,
                    "processed_frames": processed_count,
                    "total_frames": frame_num
                }
            }
            
            # Success message with summary
            st.success(f"""
            ✅ **Live Detection Complete!**
            - Processed {processed_count:,} frames out of {frame_num:,}
            - Found {len(all_detections):,} total detections
            - Identified {violation_count:,} safety violations
            - Final compliance rate: {metrics['compliance_rate']:.1f}%
            """)
            
            # Auto-scroll to results
            st.balloons()
        else:
            st.info("No detections found in the processed frames.")

# Add real-time video processing functions
def create_annotated_video_realtime(uploaded_file, client, model_id: str, conf_thresh: float, overlap_thresh: float):
    """Create a real-time annotated video for display"""
    # Save uploaded file temporarily for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_video_path = tmp_file.name
    
    # Create output path for annotated video
    output_path = tempfile.mktemp(suffix='_realtime_annotated.mp4')
    
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        os.unlink(input_video_path)
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer with more compatible codec
    fourcc = cv2.VideoWriter.fourcc(*'XVID')  # More compatible codec
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        # Try with different codec if first one fails
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    all_detections = []
    frame_num = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = (frame_num + 1) / frame_count
            progress_placeholder.progress(progress)
            status_placeholder.text(f"Processing frame {frame_num + 1}/{frame_count}")
            
            # Analyze frame for PPE detection
            try:
                response = client.infer(frame, model_id=model_id)
                predictions = response.get("predictions", [])
                filtered_preds = apply_nms(predictions, conf_thresh, overlap_thresh)
                
                # Store detections
                for det in filtered_preds:
                    det["frame"] = frame_num
                    det["timestamp"] = frame_num / fps
                    all_detections.append(det)
                
                # Draw annotations on frame
                annotated_frame = frame.copy()
                for pred in filtered_preds:
                    x1 = int(pred["x"] - pred["width"] / 2)
                    y1 = int(pred["y"] - pred["height"] / 2)
                    x2 = int(pred["x"] + pred["width"] / 2)
                    y2 = int(pred["y"] + pred["height"] / 2)
                    
                    # Color coding
                    color = get_color_for_class(pred["class"])
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background
                    label = f"{pred['class']} ({pred['confidence']:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame to output video
                if writer.isOpened():
                    writer.write(annotated_frame)
                
            except Exception as e:
                # If API fails, write original frame
                if writer.isOpened():
                    writer.write(frame)
                print(f"Frame {frame_num} API error: {e}")
            
            frame_num += 1
    
    finally:
        cap.release()
        if writer.isOpened():
            writer.release()
        os.unlink(input_video_path)
        progress_placeholder.empty()
        status_placeholder.empty()
    
    # Verify output file was created
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ValueError("Failed to create output video file")
    
    return output_path, all_detections

def run_smooth_detection(uploaded_file, model_id: str, conf_thresh: float, overlap_thresh: float):
    """Run smooth PPE detection with natural video playback - processes every frame"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Setup client
        client = get_rf_client()
        
        st.markdown("### 🎬 Smooth PPE Detection Player")
        st.info("🎯 **Natural Playback**: Processing every frame for maximum accuracy")
        
        # Playback controls
        col_info, col_controls = st.columns([3, 1])
        with col_info:
            st.success(f"📹 **{duration:.1f}s video** | {frame_count} frames @ {fps} FPS")
        
        with col_controls:
            playback_speed = st.selectbox("Playback Speed", ["0.5x", "1x", "1.5x", "2x"], index=1)
            speed_multiplier = float(playback_speed.replace('x', ''))
            
            # Option to process every frame or every 2nd frame for performance
            processing_mode = st.radio("Quality", ["Max Quality (every frame)", "Balanced (every 2nd frame)"])
            frame_skip_smooth = 1 if "Max Quality" in processing_mode else 2
        
        # Create containers for smooth display
        video_col, metrics_col = st.columns([3, 1])
        
        with video_col:
            frame_display = st.empty()
            progress_display = st.empty()
            
        with metrics_col:
            status_display = st.empty()
            detection_counter = st.empty()
            compliance_display = st.empty()
            fps_display = st.empty()
        
        # Initialize tracking
        all_detections = []
        frame_num = 0
        detection_count = 0
        violation_count = 0
        processed_count = 0
        
        # Calculate display timing for smooth playback
        target_frame_time = (1.0 / fps) / speed_multiplier
        
        # Start smooth playback
        start_time = time.time()
        last_frame_time = start_time
        
        st.info("▶️ Starting smooth video playback with PPE detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = (frame_num + 1) / frame_count
            progress_display.progress(progress)
            
            # Process frame for PPE detection (every frame or every 2nd frame)
            should_process = (frame_num % frame_skip_smooth == 0)
            annotated_frame = frame.copy()
            
            if should_process:
                try:
                    # Analyze frame
                    detections, annotated_frame = analyze_frame(
                        client, frame, model_id, conf_thresh, overlap_thresh
                    )
                    
                    # Store detections
                    for det in detections:
                        det["frame"] = frame_num
                        det["timestamp"] = frame_num / fps
                        all_detections.append(det)
                        
                        # Count violations
                        if "no-" in det["class"].lower():
                            violation_count += 1
                    
                    detection_count += len(detections)
                    processed_count += 1
                    
                except Exception as e:
                    # If processing fails, use original frame
                    annotated_frame = frame
                    status_display.error(f"⚠️ Frame {frame_num} processing failed")
            
            # Display current frame (annotated or original)
            frame_display.image(
                annotated_frame, 
                channels="BGR", 
                caption=f"🎬 Frame {frame_num + 1}/{frame_count} | {len(all_detections)} total detections",
                use_container_width=True
            )
            
            # Update live metrics
            current_time = time.time()
            elapsed_time = current_time - start_time
            actual_fps = frame_num / elapsed_time if elapsed_time > 0 else 0
            current_compliance = ((processed_count - violation_count) / processed_count * 100) if processed_count > 0 else 100
            
            # Update metric displays
            detection_counter.metric("🎯 Live Detections", f"{detection_count:,}")
            compliance_display.metric("✅ Compliance Rate", f"{current_compliance:.1f}%")
            fps_display.metric("📊 Processing FPS", f"{actual_fps:.1f}")
            
            # Status update
            status_display.info(f"🔄 Frame {frame_num + 1}/{frame_count} | {progress*100:.1f}% complete")
            
            # Timing control for smooth playback
            frame_time = current_time - last_frame_time
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)
            
            last_frame_time = time.time()
            frame_num += 1
            
            # Safety break for very long videos
            if frame_num > 2000:  # Process max 2000 frames
                st.warning("⏱️ Reached processing limit of 2000 frames for smooth playback")
                break
        
        # Final results
        if all_detections:
            metrics = calculate_compliance_metrics(all_detections)
            st.session_state.analysis_results = {
                "detections": all_detections,
                "metrics": metrics,
                "video_info": {
                    "duration": frame_num / fps,
                    "processed_frames": processed_count,
                    "total_frames": frame_num,
                    "processing_mode": "smooth_detection"
                }
            }
            
            # Success summary
            st.success(f"""
            ✅ **Smooth Detection Complete!**
            - 🎬 Played {frame_num:,} frames smoothly
            - 🔍 Processed {processed_count:,} frames for detection  
            - 🎯 Found {len(all_detections):,} total detections
            - ⚠️ Identified {violation_count:,} safety violations
            - ✅ Final compliance rate: {metrics['compliance_rate']:.1f}%
            """)
            
            st.balloons()
        else:
            st.info("🔍 No detections found during smooth playback.")
    
    finally:
        cap.release()
        os.unlink(video_path)

def run_smooth_detection_live(uploaded_file, model_id: str, conf_thresh: float, overlap_thresh: float):
    """Run smooth PPE detection with live setting updates - more responsive approach"""
    
    # Initialize or get existing session state for smooth detection
    if 'smooth_video_data' not in st.session_state:
        # Save video data once to avoid re-reading
        uploaded_file.seek(0)
        video_data = uploaded_file.read()
        uploaded_file.seek(0)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_data)
            st.session_state.smooth_video_path = tmp_file.name
        
        # Open video and get properties
        cap = cv2.VideoCapture(st.session_state.smooth_video_path)
        if not cap.isOpened():
            st.error("Could not open video file")
            return
            
        st.session_state.smooth_video_props = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
        }
        cap.release()
        
        # Initialize playback state
        st.session_state.smooth_current_frame = 0
        st.session_state.smooth_is_playing = False
        st.session_state.smooth_detections = []
    
    props = st.session_state.smooth_video_props
    
    # Control panel with live settings
    st.info(f"📹 **{props['duration']:.1f}s video** | {props['frame_count']} frames @ {props['fps']} FPS")
    
    # Simple control buttons in a single row
    if st.button("▶️ Play", key="play_btn"):
        st.session_state.smooth_is_playing = True
        
    if st.button("⏸️ Pause", key="pause_btn"):
        st.session_state.smooth_is_playing = False
        
    if st.button("⏮️ Reset", key="reset_btn"):
        st.session_state.smooth_current_frame = 0
        st.session_state.smooth_detections = []
        st.session_state.smooth_is_playing = False
    
    # Live settings that don't require restart
    playback_speed = st.selectbox("Speed", ["0.5x", "1x", "1.5x", "2x"], index=1, key="live_speed")
    processing_quality = st.selectbox("Quality", ["Every Frame", "Every 2nd", "Every 5th"], index=1, key="live_quality")
    
    speed_multiplier = float(playback_speed.replace('x', ''))
    frame_skip = {"Every Frame": 1, "Every 2nd": 2, "Every 5th": 5}[processing_quality]
    
    # Display containers (no nested columns to avoid Streamlit nesting errors)
    video_container = st.container()
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(st.session_state.smooth_current_frame / props['frame_count'])
        status_text = st.empty()
    
    # Metrics in simple layout
    st.markdown("**Live Metrics:**")
    metrics_text = st.empty()
    
    # Update metrics display function
    def update_metrics():
        violations = len([d for d in st.session_state.smooth_detections if "no-" in d['class'].lower()])
        compliance_rate = ((len(st.session_state.smooth_detections) - violations) / len(st.session_state.smooth_detections) * 100) if st.session_state.smooth_detections else 100
        
        metrics_info = f"""
        📍 **Frame:** {st.session_state.smooth_current_frame}/{props['frame_count']} | 
        🎯 **Detections:** {len(st.session_state.smooth_detections)} | 
        ✅ **Compliance:** {compliance_rate:.1f}% | 
        ⚠️ **Violations:** {violations}
        """
        metrics_text.markdown(metrics_info)
    
    # Only process if playing
    if st.session_state.smooth_is_playing and st.session_state.smooth_current_frame < props['frame_count']:
        try:
            # Open video and seek to current frame
            cap = cv2.VideoCapture(st.session_state.smooth_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.smooth_current_frame)
            
            ret, frame = cap.read()
            if ret:
                # Process frame if it's time (based on frame_skip)
                should_process = (st.session_state.smooth_current_frame % frame_skip == 0)
                annotated_frame = frame.copy()
                
                if should_process:
                    try:
                        client = get_rf_client()
                        detections, annotated_frame = analyze_frame(
                            client, frame, model_id, conf_thresh, overlap_thresh
                        )
                        
                        # Store detections with current settings
                        for det in detections:
                            det["frame"] = st.session_state.smooth_current_frame
                            det["timestamp"] = st.session_state.smooth_current_frame / props['fps']
                            st.session_state.smooth_detections.append(det)
                            
                    except Exception as e:
                        status_text.warning(f"⚠️ Frame {st.session_state.smooth_current_frame} processing failed")
                
                # Display frame
                with video_container:
                    st.image(
                        annotated_frame,
                        channels="BGR",
                        caption=f"🎬 Frame {st.session_state.smooth_current_frame + 1}/{props['frame_count']} | Processing: {processing_quality}",
                        use_container_width=True
                    )
                
                # Update progress
                progress_bar.progress((st.session_state.smooth_current_frame + 1) / props['frame_count'])
                status_text.info(f"▶️ Playing at {playback_speed} | {(st.session_state.smooth_current_frame + 1) / props['frame_count'] * 100:.1f}% complete")
                
                # Calculate and update live metrics
                violations = len([d for d in st.session_state.smooth_detections if "no-" in d['class'].lower()])
                compliance_rate = ((len(st.session_state.smooth_detections) - violations) / len(st.session_state.smooth_detections) * 100) if st.session_state.smooth_detections else 100
                
                # Update metrics display
                metrics_info = f"""
                📍 **Frame:** {st.session_state.smooth_current_frame + 1}/{props['frame_count']} | 
                🎯 **Detections:** {len(st.session_state.smooth_detections)} | 
                ✅ **Compliance:** {compliance_rate:.1f}% | 
                ⚠️ **Violations:** {violations}
                """
                metrics_text.markdown(metrics_info)
                
                # Advance frame
                st.session_state.smooth_current_frame += 1
                
                # Control playback speed
                time.sleep(max(0.05, (1.0 / props['fps']) / speed_multiplier))
                
                # Auto-refresh for smooth playback
                st.rerun()
                
            cap.release()
            
        except Exception as e:
            st.error(f"Playback error: {str(e)}")
            st.session_state.smooth_is_playing = False
            
    elif st.session_state.smooth_current_frame >= props['frame_count']:
        # Video finished
        st.session_state.smooth_is_playing = False
        status_text.success("✅ Video playback completed!")
        
        if st.session_state.smooth_detections:
            # Calculate final metrics
            metrics = calculate_compliance_metrics(st.session_state.smooth_detections)
            st.session_state.analysis_results = {
                "detections": st.session_state.smooth_detections,
                "metrics": metrics,
                "video_info": {
                    "duration": props['duration'],
                    "processed_frames": len(st.session_state.smooth_detections),
                    "total_frames": props['frame_count'],
                    "processing_mode": "smooth_live"
                }
            }
            
            st.success(f"""
            🎬 **Smooth Detection Complete!**
            - 📺 Played {props['frame_count']:,} frames
            - 🎯 Found {len(st.session_state.smooth_detections):,} detections
            - ✅ Final compliance rate: {metrics['compliance_rate']:.1f}%
            """)
            st.balloons()
    
    # Display current frame even when paused
    elif not st.session_state.smooth_is_playing:
        try:
            cap = cv2.VideoCapture(st.session_state.smooth_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.smooth_current_frame)
            ret, frame = cap.read()
            if ret:
                with video_container:
                    st.image(
                        frame,
                        channels="BGR",
                        caption=f"⏸️ Paused at Frame {st.session_state.smooth_current_frame + 1}/{props['frame_count']}",
                        use_container_width=True
                    )
            cap.release()
            status_text.info("⏸️ Playback paused - Click Play to continue")
        except:
            pass

# Main interface - Single column layout to avoid nesting issues
st.markdown("### 📤 Upload Video for Analysis")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Upload a video file showing workers with or without PPE"
)

if uploaded_file is not None:
    # If live detection recently triggered, show live UI only
    if st.session_state.get('show_realtime', False):
        uploaded_file.seek(0)
        run_smooth_detection_live(uploaded_file, model_id, conf_thresh, overlap_thresh)
        st.stop()
    # Two column layout for main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display video info
        st.markdown("#### 📹 Video Information")
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }
        
        # Display file info in a simple format without nested columns
        st.write(f"**Filename:** {file_details['Filename']}")
        st.write(f"**File size:** {file_details['File size']}")
        st.write(f"**File type:** {file_details['File type']}")
        
        # Show original video immediately
        st.markdown("### 📹 Original Video Preview")
        video_bytes = uploaded_file.read()
        st.video(video_bytes)
        
        # Reset file pointer for further processing
        uploaded_file.seek(0)
        
        # Analysis options - Simplified controls
        st.markdown("### 🔧 Analysis Options")
        st.markdown("**Choose your analysis type:**")
        
        # Instant Preview
        if st.button("⚡ Instant Preview", type="primary", use_container_width=True):
            st.markdown("### 📹 Video Preview at 4 Seconds")
            uploaded_file.seek(0)
            data = uploaded_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(data)
                video_path = tmp.name
            cap2 = cv2.VideoCapture(video_path)
            fps_val = cap2.get(cv2.CAP_PROP_FPS) or 1
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(fps_val * 4))  # jump to 4 seconds
            ret2, frame2 = cap2.read()
            cap2.release()
            if ret2:
                client = get_rf_client()
                detections, annotated2 = analyze_frame(client, frame2, model_id, conf_thresh, overlap_thresh)
                st.image(annotated2, channels="BGR", use_container_width=True)
                st.success(f"✅ Instant preview at 4s: {len(detections)} detections")
            else:
                st.error("❌ Could not retrieve frame at 4 seconds")
        
        # Smooth Detection trigger
        if st.button("🎬 Smooth Detection", use_container_width=True):
            uploaded_file.seek(0)
            # Activate live detection UI
            st.session_state.show_realtime = True
            run_smooth_detection_live(uploaded_file, model_id, conf_thresh, overlap_thresh)
            # Stop further rendering
            st.stop()
        
        # Full Analysis trigger
        if st.button("📊 Full Analysis", use_container_width=True):
            with st.spinner("🔄 Running comprehensive analysis..."):
                try:
                    client = get_rf_client()
                    results = process_video(uploaded_file, client, model_id, conf_thresh, overlap_thresh, frame_skip, save_annotated_video)
                    st.session_state.analysis_results = results
                    st.session_state.processed_video_path = results.get("annotated_video_path")
                    st.success("✅ Full analysis completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Analysis Dashboard")
        
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            metrics = results["metrics"]
            video_info = results.get("video_info", {})
            
            # Compliance overview
            compliance_rate = metrics["compliance_rate"]
            if compliance_rate >= 80:
                status_class = "status-safe"
                status_text = "🟢 COMPLIANT"
            elif compliance_rate >= 60:
                status_class = "status-warning"
                status_text = "🟡 NEEDS ATTENTION"
            else:
                status_class = "status-danger"
                status_text = "🔴 NON-COMPLIANT"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Safety Compliance</h4>
                <span class="status-badge {status_class}">{status_text}</span>
                <p><strong>Compliance Rate:</strong> {compliance_rate:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            st.markdown(f"""
            <div class="metric-card">
                <h4>Detection Summary</h4>
                <p><strong>Frames Analyzed:</strong> {metrics['total_frames_analyzed']}</p>
                <p><strong>Persons Detected:</strong> {metrics['total_persons_detected']}</p>
                <p><strong>Safety Violations:</strong> {metrics['total_violations']}</p>
                <p><strong>Video Duration:</strong> {video_info.get('duration', 0):.1f}s</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show live detection status
            if st.session_state.get('show_realtime', False):
                st.markdown("""
                <div class="metric-card">
                    <h4>🎬 Live Detection Active</h4>
                    <p>🔴 Real-time processing in progress</p>
                    <p>🎯 PPE detection running frame-by-frame</p>
                    <p>📺 Watch the live feed in the main area</p>
                </div>
                """, unsafe_allow_html=True)
            elif 'realtime_video_path' in st.session_state:
                st.markdown("""
                <div class="metric-card">
                    <h4>🎬 Live Detection</h4>
                    <p>✅ Video processed with real-time annotations</p>
                    <p>🎯 PPE equipment identified frame-by-frame</p>
                    <p>📺 Watch the annotated video above</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Download annotated video (for detailed analysis)
            processed_video_path = st.session_state.get('processed_video_path')
            if processed_video_path and os.path.exists(processed_video_path):
                try:
                    with open(processed_video_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Detailed Analysis Video",
                            data=file.read(),
                            file_name="ppe_detailed_analysis.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error accessing video file: {str(e)}")
        
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>Ready for Analysis</h4>
                <p>Upload a video and choose your analysis type:</p>
                <p>🎬 <strong>Live Detection:</strong> Watch PPE identification in real-time</p>
                <p>📊 <strong>Detailed Analysis:</strong> Get comprehensive reports</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4>🎯 What We Detect</h4>
                <p>🦺 Safety Vests</p>
                <p>⛑️ Hard Hats</p>
                <p>👷 Workers/Persons</p>
                <p>⚠️ Safety Violations</p>
            </div>
            """, unsafe_allow_html=True)

# Detailed analysis section
if st.session_state.analysis_results is not None and detailed_analysis:
    st.markdown("---")
    st.markdown("### 📈 Detailed Analysis Report")
    
    results = st.session_state.analysis_results
    detections = results["detections"]
    
    if detections:
        # Create detection timeline
        df = pd.DataFrame(detections)
        
        # Detection count by class
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Detection Counts by Class")
            class_counts = df['class'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
            # Convert pandas values to list to avoid type issues
            bars = ax.bar(class_counts.index, class_counts.values.tolist(), color=colors[:len(class_counts)])
            ax.set_title('PPE Detection Summary')
            ax.set_xlabel('Equipment Type')
            ax.set_ylabel('Detection Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### ⏱️ Detection Timeline")
            # Create timeline data
            timeline_data = df.groupby(['frame', 'class']).size().reset_index(name='count')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            for class_name in timeline_data['class'].unique():
                class_data = timeline_data[timeline_data['class'] == class_name]
                ax.plot(class_data['frame'], class_data['count'], 
                       marker='o', label=class_name, linewidth=2)
            
            ax.set_title('Detection Timeline')
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Detection Count')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        
        # Confidence distribution
        st.markdown("#### 🎯 Confidence Score Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for class_name in df['class'].unique():
            class_data = df[df['class'] == class_name]
            ax.hist(class_data['confidence'], alpha=0.7, label=class_name, bins=20)
        
        ax.set_title('Detection Confidence Distribution')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        # Raw data table
        with st.expander("📋 View Raw Detection Data"):
            st.dataframe(df, use_container_width=True)
            
            # Export functionality
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Detection Data (CSV)",
                data=csv,
                file_name="ppe_detection_data.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <p>🛡️ Smart Safety Monitor - Video PPE Analysis</p>
    <p>Powered by AI Computer Vision • Real-time Safety Compliance Monitoring</p>
</div>
""", unsafe_allow_html=True)