import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import sys
import base64
from typing import List, Dict, Tuple
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort

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

# Load audio file
@st.cache_data
def load_audio_bytes():
    """Load emergency alarm audio file using correct path resolution"""
    # Try multiple possible paths to find the audio file
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
    page_title="PPE Video Analysis – Smart Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables FIRST - before any widgets
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = False
if 'current_playback_speed' not in st.session_state:
    st.session_state.current_playback_speed = 1.0
if 'processing_active' not in st.session_state:
    st.session_state.processing_active = False
if 'processing_counter' not in st.session_state:
    st.session_state.processing_counter = 0
if 'video_file_uploaded' not in st.session_state:
    st.session_state.video_file_uploaded = None
if 'video_process_running' not in st.session_state:
    st.session_state.video_process_running = False
if 'video_processing_session' not in st.session_state:
    st.session_state.video_processing_session = 0

# CSS for consistent styling
st.markdown("""
<style>
    /* Global text color override for main content */
    .main .block-container {
        color: white !important;
    }
    
    /* Force all text elements to be white */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stText, .stWrite {
        color: white !important;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: #ffffff !important;
        color: #333333 !important;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        text-align: center;
    }
    
    .metric-card h4 {
        color: #333333 !important;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .metric-card p {
        color: #555555 !important;
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .status-safe {background: linear-gradient(45deg, #28a745, #20c997);}
    .status-warning {background: linear-gradient(45deg, #ffc107, #fd7e14);}
    .status-danger {background: linear-gradient(45deg, #dc3545, #e74c3c);}
    
    /* Progress container */
    .progress-container {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Enhanced compliance alert with animations */
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
    
    /* Violation Alert Animations */
    @keyframes flashRed {
        0%, 50% { background-color: #dc3545; }
        25%, 75% { background-color: #ff6b6b; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes sirenFlash {
        0%, 25% { color: #dc3545; text-shadow: 0 0 10px #dc3545; }
        50%, 75% { color: #ffc107; text-shadow: 0 0 10px #ffc107; }
    }
    
    .violation-alert {
        animation: flashRed 1s infinite, pulse 2s infinite;
        border: 3px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .siren-icon {
        animation: sirenFlash 0.5s infinite;
        font-size: 2rem;
        margin: 0 0.5rem;
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

def normalize_class_name(class_name: str) -> str:
    """Normalize class names for consistent display"""
    normalized = class_name.lower()
    
    # Rename Hardhat to Helmet
    if normalized == "hardhat":
        return "Helmet"
    elif normalized == "no-hardhat":
        return "No-Helmet"
    
    # Handle other normalizations
    replacements = {
        "safety-vest": "Safety Vest",
        "no-safety-vest": "No-Safety Vest",
        "safety cone": "Safety Cone",
        "mask": "Mask",
        "no-mask": "No-Mask",
        "person": "Person"
    }
    
    return replacements.get(normalized, class_name.title())

def should_include_in_display(class_name: str) -> bool:
    """Determine if a class should be displayed (exclude Machinery and Vehicle)"""
    excluded_classes = ["machinery", "vehicle"]
    return class_name.lower() not in excluded_classes

def should_include_in_metrics(class_name: str) -> bool:
    """Determine if a class should be included in compliance metrics"""
    ppe_classes = ["hardhat", "helmet", "no-hardhat", "no-helmet", "mask", "no-mask", 
                   "safety-vest", "no-safety-vest"]
    return class_name.lower() in ppe_classes

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def associate_ppe_with_person(person_box, ppe_detections, iou_threshold=0.1):
    """Associate PPE detections with a person based on spatial proximity"""
    person_ppe = set()
    
    for ppe_name, ppe_box in ppe_detections:
        # Calculate IoU or simple overlap
        iou = calculate_iou(person_box, ppe_box)
        
        # Also check if PPE is near the person (within expanded bounding box)
        expanded_person = [
            person_box[0] - 50,  # Expand left
            person_box[1] - 50,  # Expand top  
            person_box[2] + 50,  # Expand right
            person_box[3] + 50   # Expand bottom
        ]
        
        proximity = calculate_iou(expanded_person, ppe_box)
        
        if iou > iou_threshold or proximity > iou_threshold:
            person_ppe.add(ppe_name.lower())
    
    return person_ppe

def process_video_realtime(video_path: str, conf_thresh: float, model, frame_placeholder, progress_bar, status_text, metrics_placeholder, playback_speed: float = 1.0, audio_placeholder=None):
    """Process video with real-time display and DeepSORT tracking"""
    
    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30, n_init=3)
    
    # Keep track of unique people and their PPE status
    seen_people = {}  # track_id -> person_info with PPE status
    total_violations = 0
    total_detections = 0
    
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    start_time = time.time()
    
    # PPE class names we're looking for
    required_ppe = {"helmet", "mask", "safety vest"}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO inference
        results = model(frame, conf=conf_thresh, verbose=False)
        
        person_detections = []
        ppe_detections = []
        
        # Process YOLO results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id].lower()
                    
                    # Skip excluded classes
                    if not should_include_in_display(class_name):
                        continue
                    
                    total_detections += 1
                    
                    if class_name == "person":
                        # Format for DeepSORT: [x, y, w, h]
                        person_detections.append([[x1, y1, x2-x1, y2-y1], confidence, class_id])
                    elif class_name in ["hardhat", "helmet", "mask", "safety vest"]:
                        # Normalize class names and store for PPE association
                        normalized_name = normalize_class_name(class_name)
                        ppe_detections.append((normalized_name.lower(), [x1, y1, x2, y2]))
        
        # Calculate metrics based on tracked people (not raw PPE counts)
        current_helmets = sum(1 for p in seen_people.values() if p.get("has_helmet", False))
        current_masks = sum(1 for p in seen_people.values() if p.get("has_mask", False))
        current_vests = sum(1 for p in seen_people.values() if p.get("has_vest", False))
        current_violations = sum(1 for p in seen_people.values() if not p.get("is_compliant", True))
        
        # Calculate compliance (same logic as image analysis)
        helmet_compliance = 1 if current_helmets > 0 else 0
        mask_compliance = 1 if current_masks > 0 else 0
        vest_compliance = 1 if current_vests > 0 else 0
        overall_compliance = (helmet_compliance + mask_compliance + vest_compliance) / 3 * 100
        
        # Update tracker with person detections
        tracks = tracker.update_tracks(person_detections, frame=frame)
        
        # Process tracked people and associate PPE with person IDs
        current_frame_people = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            current_frame_people.add(track_id)
            ltrb = track.to_ltrb()
            person_box = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
            
            # Associate PPE with this person
            person_ppe = associate_ppe_with_person(person_box, ppe_detections)
            
            # Check compliance for this person
            has_helmet = any(ppe in person_ppe for ppe in ["helmet"])
            has_mask = any(ppe in person_ppe for ppe in ["mask"])
            has_vest = any(ppe in person_ppe for ppe in ["safety vest"])
            
            compliance_score = (int(has_helmet) + int(has_mask) + int(has_vest)) / 3 * 100
            is_compliant = compliance_score >= 90
            
            # Initialize or update person tracking info
            if track_id not in seen_people:
                seen_people[track_id] = {
                    "first_seen": frame_count,
                    "best_compliance": compliance_score,
                    "total_frames": 1,
                    "has_helmet": has_helmet,
                    "has_mask": has_mask,
                    "has_vest": has_vest,
                    "is_compliant": is_compliant
                }
            else:
                seen_people[track_id]["total_frames"] += 1
                seen_people[track_id]["best_compliance"] = max(
                    seen_people[track_id]["best_compliance"], 
                    compliance_score
                )
                # Update current PPE status
                seen_people[track_id]["has_helmet"] = has_helmet
                seen_people[track_id]["has_mask"] = has_mask
                seen_people[track_id]["has_vest"] = has_vest
                seen_people[track_id]["is_compliant"] = is_compliant
            
            # Draw person bounding box
            color = (0, 255, 0) if is_compliant else (0, 0, 255)
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), color, 2)
            
            # Draw person label
            status = "COMPLIANT" if is_compliant else "VIOLATION"
            label = f"ID:{track_id} | {status} ({compliance_score:.0f}%)"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1]) - label_size[1] - 8), 
                         (int(ltrb[0]) + label_size[0], int(ltrb[1])), color, -1)
            cv2.putText(frame, label, (int(ltrb[0]), int(ltrb[1]) - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw PPE detections
        for ppe_name, ppe_box in ppe_detections:
            color = get_color_for_class(ppe_name)
            cv2.rectangle(frame, (int(ppe_box[0]), int(ppe_box[1])), 
                         (int(ppe_box[2]), int(ppe_box[3])), color, 2)
            cv2.putText(frame, ppe_name, (int(ppe_box[0]), int(ppe_box[1]) - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Update display
        frame_placeholder.image(frame, channels="BGR", caption=f"Frame {frame_count}/{total_frames}")
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Update status
        elapsed_time = time.time() - start_time
        fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
        eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
        
        status_text.text(f"Processing frame {frame_count}/{total_frames} | "
                        f"FPS: {fps_current:.1f} | ETA: {eta:.1f}s")
        
        # Update live metrics (same structure as image analysis)
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🪖 Helmets", current_helmets)
            with col2:
                st.metric("😷 Masks", current_masks)
            with col3:
                st.metric("🦺 Safety Vests", current_vests)
            with col4:
                st.metric("✅ Compliance", f"{overall_compliance:.1f}%")
        
        # Adjust delay based on playback speed and reduce base FPS
        target_fps = 15  # Reduced from 30 for better performance
        frame_delay = (1.0 / target_fps) / playback_speed
        actual_elapsed = time.time() - start_time
        expected_time = frame_count / target_fps
        sleep_time = max(0, frame_delay - (actual_elapsed - expected_time))
        time.sleep(sleep_time)
    
    cap.release()
    processing_time = time.time() - start_time
    
    # Calculate final statistics based on tracked people
    final_helmets = sum(1 for p in seen_people.values() if p.get("has_helmet", False))
    final_masks = sum(1 for p in seen_people.values() if p.get("has_mask", False))
    final_vests = sum(1 for p in seen_people.values() if p.get("has_vest", False))
    final_violations = sum(1 for p in seen_people.values() if not p.get("is_compliant", True))
    
    final_helmet_compliance = 1 if final_helmets > 0 else 0
    final_mask_compliance = 1 if final_masks > 0 else 0  
    final_vest_compliance = 1 if final_vests > 0 else 0
    final_compliance = (final_helmet_compliance + final_mask_compliance + final_vest_compliance) / 3 * 100
    
    return {
        "success": True,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "total_violations": final_violations,
        "unique_people": len(seen_people),
        "compliant_people": sum(1 for p in seen_people.values() if p.get("is_compliant", True)),
        "processing_time": processing_time,
        "avg_fps": frame_count / processing_time if processing_time > 0 else 0,
        "compliance_percentage": final_compliance,
        "people_details": seen_people,
        "helmets": final_helmets,
        "masks": final_masks,
        "vests": final_vests,
        "no_helmets": len(seen_people) - final_helmets,
        "no_masks": len(seen_people) - final_masks,
        "no_vests": len(seen_people) - final_vests
    }

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
                    class_name = model.names[class_id]
                    
                    # Skip excluded classes (Machinery, Vehicle)
                    if not should_include_in_display(class_name):
                        continue
                    
                    # Normalize class name (Hardhat -> Helmet)
                    display_name = normalize_class_name(class_name)
                    
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
                        "class": display_name,
                        "original_class": class_name  # Keep original for metric calculations
                    }
                    detections.append(detection)
                    
                    # Draw bounding box on image
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    color = get_color_for_class(class_name)
                    
                    # Draw rectangle
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with normalized name
                    label = f"{display_name} ({confidence:.2f})"
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

def calculate_compliance_percentage(ppe_counts):
    """Calculate compliance percentage based on PPE counts (same logic as image analysis)"""
    helmet_count = ppe_counts.get("Hardhat", 0)
    mask_count = ppe_counts.get("Mask", 0)
    vest_count = ppe_counts.get("Safety Vest", 0)
    
    # Calculate compliance for each PPE item
    helmet_compliance = 1 if helmet_count > 0 else 0
    mask_compliance = 1 if mask_count > 0 else 0
    vest_compliance = 1 if vest_count > 0 else 0
    
    # Total compliance is sum of individual compliances divided by 3
    return (helmet_compliance + mask_compliance + vest_compliance) / 3 * 100

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
    
    # PPE category counts - using original class names for accurate counting
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
        
        # Update statistics and PPE counts using original class names
        total_detections += len(detections)
        for detection in detections:
            original_class = detection['original_class']
            if original_class in ppe_counts:
                ppe_counts[original_class] += 1
            if 'no-' in original_class.lower():
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
    st.header("⚙️ Detection Parameters")
    
    # Show model info
    if LOCAL_MODEL_AVAILABLE and LOCAL_MODEL_PATH:
        st.success("🎯 Local YOLO Model Ready")
        st.info(f"📁 Model: {os.path.basename(LOCAL_MODEL_PATH)}")
    else:
        st.error("❌ Local model not found")
        st.stop()
    
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, float(CONF_THRESH_DEFAULT), 0.05)
    
    st.markdown("---")
    st.header("🔊 Audio Alerts")
    
    # Audio toggle control using session state
    if st.toggle(
        "Enable Audio Alerts", 
        value=st.session_state.audio_enabled,
        help="Automatically play alarm sound when violations are detected",
        key="audio_toggle"
    ):
        if not st.session_state.audio_enabled:
            st.session_state.audio_enabled = True
    else:
        if st.session_state.audio_enabled:
            st.session_state.audio_enabled = False

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
    
    # Playback speed control for real-time processing
    st.markdown("---")
    st.subheader("🎛️ Real-time Playback Controls")
    
    # Speed selection with session state
    speed_options = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    speed_labels = ["1.0x", "2.0x", "4.0x", "6.0x", "8.0x", "10.0x"]
    
    current_speed_index = speed_options.index(st.session_state.current_playback_speed) if st.session_state.current_playback_speed in speed_options else 0
    
    # Use a form to prevent immediate reruns
    with st.form("speed_control_form"):
        selected_speed_index = st.selectbox(
            "Playback Speed",
            range(len(speed_options)),
            index=current_speed_index,
            format_func=lambda x: speed_labels[x],
            help="Select playback speed for real-time video processing"
        )
        speed_submit = st.form_submit_button("Update Speed")
        
        if speed_submit:
            st.session_state.current_playback_speed = speed_options[selected_speed_index]
            st.success(f"Speed updated to {speed_labels[selected_speed_index]}")
    
    # Process video button - now with real-time option
    col1, col2 = st.columns(2)
    
    with col1:
        realtime_mode = st.button("🎥 Play with Real-time Detection", type="primary", help="Play video with live PPE detection and tracking")
        if realtime_mode:
            # Increment processing session to ensure fresh keys
            st.session_state.video_processing_session += 1
    
    with col2:
        batch_mode = st.button("🚀 Process Complete Video", help="Process entire video and download result")
    
    if realtime_mode or batch_mode:
        if realtime_mode:
            st.markdown("### 🎥 Real-time PPE Detection & Tracking")
            
            # Create placeholders for real-time updates
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create columns for live metrics
            st.markdown("#### 📊 Live Detection Metrics")
            metrics_placeholder = st.empty()
            
            # Use the session state speed (already set by the main control)
            playback_speed = st.session_state.current_playback_speed
            
            # Audio alert placeholder
            audio_placeholder = st.empty()
            
            # Process video in real-time using session state speed
            with st.spinner("🎬 Starting real-time video analysis..."):
                result = process_video_realtime(
                    input_path, conf_thresh, model, 
                    frame_placeholder, progress_bar, status_text, metrics_placeholder,
                    st.session_state.current_playback_speed, audio_placeholder
                )
            
            if result.get("success"):
                st.success("✅ Real-time analysis completed!")
                
                # Check for violations and trigger audio alert at the end
                compliance_pct = float(result.get("compliance_percentage", 100))
                people_count = int(result.get("unique_people", 0))
                
                if compliance_pct < 50 and people_count > 0:
                    with audio_placeholder.container():
                        st.markdown(f"""
                        <div class="violation-alert">
                            <h3 style="color: white; margin: 0;">
                                <span class="siren-icon">🚨</span>
                                SAFETY VIOLATIONS DETECTED
                                <span class="siren-icon">🚨</span>
                            </h3>
                            <p style="color: white; margin: 0.5rem 0; font-size: 1.1rem;">
                                Overall Compliance: {compliance_pct:.1f}% - IMMEDIATE ACTION REQUIRED
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Audio alert if enabled (only at the end)
                        if st.session_state.get('audio_enabled', False):
                            audio_bytes, _ = load_audio_bytes()
                            if audio_bytes:
                                audio_base64 = base64.b64encode(audio_bytes).decode()
                                st.markdown(f"""
                                <audio autoplay loop style="display: none;">
                                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                                    Your browser does not support the audio element.
                                </audio>
                                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Real-time processing failed: {result.get('error', 'Unknown error')}")
        
        elif batch_mode:
            # Original batch processing mode
            st.markdown("### 🚀 Batch Video Processing")
            
            # Create output path
            with tempfile.NamedTemporaryFile(delete=False, suffix='_processed.mp4') as tmp_output:
                output_path = tmp_output.name
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔍 Processing complete video with local YOLO model..."):
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
                    st.markdown("### 📊 Detection Results")
                    
                    # Determine safety status
                    compliance = result["compliance_percentage"]
                    people = result["ppe_counts"].get("Person", 0)
                    
                    if compliance >= 90:
                        status = "FULLY COMPLIANT"
                        status_class = "status-safe"
                        status_icon = "✅"
                    elif compliance >= 50:
                        status = "PARTIALLY COMPLIANT"
                        status_class = "status-warning"
                        status_icon = "⚠️"
                    elif result["total_violations"] > 0:
                        status = "VIOLATIONS DETECTED"
                        status_class = "status-danger"
                        status_icon = "🚨"
                    else:
                        status = "NO PEOPLE DETECTED"
                        status_class = "status-warning"
                        status_icon = "👤"
                    
                    # Violation Alert System - trigger if compliance < 50%
                    if compliance < 50 and people > 0:
                        st.markdown(f"""
                        <div class="violation-alert">
                            <h2 style="color: white; margin: 0;">
                                <span class="siren-icon">🚨</span>
                                SAFETY VIOLATION ALERT
                                <span class="siren-icon">🚨</span>
                            </h2>
                            <p style="color: white; margin: 0.5rem 0; font-size: 1.2rem;">
                                Compliance Level: {compliance:.1f}% - IMMEDIATE ACTION REQUIRED
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Audio alert using hidden audio element if enabled
                        if st.session_state.get('audio_enabled', False):
                            audio_bytes, _ = load_audio_bytes()
                            if audio_bytes:
                                # Create hidden audio element with loop for continuous play
                                audio_base64 = base64.b64encode(audio_bytes).decode()
                                
                                st.markdown(f"""
                                <audio autoplay loop style="display: none;">
                                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                                    Your browser does not support the audio element.
                                </audio>
                                """, unsafe_allow_html=True)
                    
                    # Status badge
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="status-badge {status_class}">
                            {status_icon} {status}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
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
                os.unlink(output_path)
            except:
                pass

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6c757d;padding:1rem;'>🛡️ Smart Safety Monitor - Local YOLO Video Analysis</div>", unsafe_allow_html=True)
