import streamlit as st
import cv2
import numpy as np
from typing import List, Tuple
import tempfile
import os
import sys
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

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
    page_title="PPE Image Analysis – Smart Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Copy CSS for consistent styling
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
    
    /* Upload area enhancement */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Results section */
    .results-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    /* Detection details card with black text for visibility */
    .detection-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .detection-card p {
        color: #333333 !important;
        margin: 0;
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
    
    /* Compact image styling */
    .compact-image {
        max-height: 400px;
        object-fit: contain;
    }
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
        "hardhat": (0,255,0),
        "helmet": (0,255,0),
        "mask": (0,255,255),
        "safety-vest": (0,255,255),
        "person": (255,0,0),
        "safety cone": (255,165,0),
        "no-hardhat": (0,0,255),
        "no-helmet": (0,0,255),
        "no-mask": (0,0,255),
        "no-safety-vest": (0,165,255)
    }.get(name.lower(), (128,128,128))

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

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Detection Parameters")
    
    # Check model availability
    if not LOCAL_MODEL_AVAILABLE or not LOCAL_MODEL_PATH:
        st.error("❌ Local model not found")
        st.stop()
    
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    st.markdown("---")
    st.header("🔊 Audio Alerts")
    
    # Audio toggle control
    audio_enabled = st.toggle(
        "Enable Audio Alerts", 
        value=False,
        help="Automatically play alarm sound when violations are detected"
    )
    
    # Initialize session state
    if 'audio_enabled' not in st.session_state:
        st.session_state.audio_enabled = False
    
    # Update session state
    if audio_enabled != st.session_state.audio_enabled:
        st.session_state.audio_enabled = audio_enabled

# Load model
model = load_yolo_model()
if model is None:
    st.error("❌ Failed to load YOLO model. Please check the model file.")
    st.stop()

# Main content
st.markdown("### 📤 Upload Image for Local PPE Analysis")
st.markdown("""
<div class="upload-area">
    <p style="color: #667eea; font-size: 1.1rem; margin: 0;">
        🖼️ Drag and drop or click to upload an image showing workers with or without PPE
    </p>
    <p style="color: #6c757d; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
        Supported formats: JPG, JPEG, PNG, BMP
    </p>
</div>
""", unsafe_allow_html=True)

img_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp"], help="Upload an image showing workers with or without PPE", label_visibility="collapsed")
if img_file:
    # Read and process image
    data = img_file.read()
    img_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Show original with better layout (compact and centered)
    st.markdown("### 📷 Original Image")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, channels="BGR", caption=f"Uploaded: {img_file.name}", width=600)

    # Analyze with progress indicator
    with st.spinner("🔍 Analyzing image with local YOLO model..."):
        detections, annotated = run_yolo_inference(image, conf_thresh, model)

    # Results section
    st.markdown("---")
    if detections:
        st.markdown("### 🎯 Analysis Results")
        
        # Show annotated image (compact and centered)
        st.markdown("#### Annotated Image with Detections")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(annotated, channels="BGR", 
                    caption=f"✅ Analysis Complete - {len(detections)} objects detected", 
                    width=600)
    else:
        st.warning("🔍 No objects detected in the image. Try adjusting the confidence threshold or upload a different image.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, channels="BGR", caption="No detections found", width=600)

    # New compliance calculation system based on original classes
    def calculate_compliance(detections):
        """Calculate compliance based on presence of all 3 PPE items"""
        # Count based on original class names (before normalization)
        helmet_count = sum(1 for d in detections if d['original_class'].lower() == 'hardhat')
        no_helmet_count = sum(1 for d in detections if d['original_class'].lower() == 'no-hardhat')
        
        mask_count = sum(1 for d in detections if d['original_class'].lower() == 'mask')
        no_mask_count = sum(1 for d in detections if d['original_class'].lower() == 'no-mask')
        
        vest_count = sum(1 for d in detections if d['original_class'].lower() == 'safety vest')
        no_vest_count = sum(1 for d in detections if d['original_class'].lower() == 'no-safety vest')
        
        # Calculate compliance for each PPE item
        helmet_compliance = 1 if helmet_count > 0 else 0
        mask_compliance = 1 if mask_count > 0 else 0
        vest_compliance = 1 if vest_count > 0 else 0
        
        # Total compliance is sum of individual compliances divided by 3
        total_compliance = (helmet_compliance + mask_compliance + vest_compliance) / 3 * 100
        
        return total_compliance, helmet_count, mask_count, vest_count, no_helmet_count, no_mask_count, no_vest_count

    # Calculate metrics using the new system
    compliance, helmets, masks, vests, no_helmets, no_masks, no_vests = calculate_compliance(detections)
    total_violations = no_helmets + no_masks + no_vests
    people = sum(1 for d in detections if d['original_class'].lower() == 'person')
    safety_cones = sum(1 for d in detections if d['original_class'].lower() == 'safety cone')
    
    # Enhanced metrics display with visual indicators
    st.markdown("### 📊 Detection Results")
    
    # Determine safety status
    if compliance >= 90:
        status = "FULLY COMPLIANT"
        status_class = "status-safe"
        status_icon = "✅"
    elif compliance >= 50:
        status = "PARTIALLY COMPLIANT"
        status_class = "status-warning"
        status_icon = "⚠️"
    elif total_violations > 0:
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
                import base64
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
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("👥 People", people, help="Total number of people detected")
    with col2:
        st.metric("🪖 Helmets", helmets, help="Workers wearing helmets")
    with col3:
        st.metric("😷 Masks", masks, help="Workers wearing masks")
    with col4:
        st.metric("🦺 Safety Vests", vests, help="Workers wearing safety vests")
    with col5:
        st.metric("⚠️ Violations", total_violations, delta=f"-{total_violations}" if total_violations > 0 else None, delta_color="inverse", help="PPE violations detected")
    with col6:
        st.metric("✅ Compliance", f"{compliance:.1f}%", 
                 delta=f"{compliance:.1f}%" if compliance >= 80 else f"-{100-compliance:.1f}%", 
                 delta_color="normal" if compliance >= 80 else "inverse",
                 help="PPE compliance percentage (Helmet + Mask + Vest) / 3")

    # Download section with better styling
    st.markdown("### 💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Download annotated image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            cv2.imwrite(tmp.name, annotated)
            with open(tmp.name, 'rb') as f:
                btn = st.download_button(
                    label="📥 Download Annotated Image",
                    data=f.read(),
                    file_name=f"ppe_analysis_{img_file.name.split('.')[0]}.png",
                    mime="image/png",
                    help="Download the image with detection annotations",
                    use_container_width=True
                )
            os.unlink(tmp.name)
    
    with col2:
        # Generate and download report
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""PPE Analysis Report
{'=' * 50}
Image: {img_file.name}
Analysis Date: {current_time}

Detection Summary:
- Total Objects: {len(detections)}
- People Detected: {people}
- Helmets: {helmets}
- Masks: {masks}
- Safety Vests: {vests}
- Total Violations: {total_violations}
- Compliance Rate: {compliance:.1f}%

Safety Status: {status}

Individual PPE Compliance:
- Helmet Compliance: {'✅' if helmets > 0 else '❌'}
- Mask Compliance: {'✅' if masks > 0 else '❌'}
- Vest Compliance: {'✅' if vests > 0 else '❌'}

Detailed Detections:
"""
        for i, d in enumerate(detections, 1):
            report += f"{i}. {d['class']} (Confidence: {d['confidence']:.2f}) at position ({int(d['x'])}, {int(d['y'])})\n"
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=report,
            file_name=f"ppe_report_{img_file.name.split('.')[0]}.txt",
            mime="text/plain",
            help="Download a detailed text report of the analysis",
            use_container_width=True
        )

    # Enhanced detection details (remove individual detections, fix text colors)
    if detections:
        with st.expander("🔍 Detailed Detection Analysis", expanded=True):
            st.markdown("#### Detection Summary")
            
            # Group detections by type
            detection_groups = {}
            for d in detections:
                class_name = d['class']
                if class_name not in detection_groups:
                    detection_groups[class_name] = []
                detection_groups[class_name].append(d)
            
            # Display grouped detections with black text
            for class_name, group in detection_groups.items():
                count = len(group)
                avg_confidence = sum(d['confidence'] for d in group) / count
                
                # Determine status color
                if 'no-' in class_name.lower():
                    status_color = "#dc3545"  # Red for violations
                    icon = "⚠️"
                elif class_name.lower() in ['helmet', 'mask', 'safety vest']:
                    status_color = "#28a745"  # Green for PPE
                    icon = "✅"
                else:
                    status_color = "#6c757d"  # Gray for people/cones
                    icon = "👤" if class_name.lower() == 'person' else "🔶"
                
                st.markdown(f"""
                <div class="detection-card">
                    <h5 style="color: {status_color}; margin: 0 0 0.5rem 0;">
                        {icon} {class_name.replace('-', ' ').title()}
                    </h5>
                    <p style="margin: 0; color: #333333 !important;">
                        <strong style="color: #333333;">Count:</strong> {count} | 
                        <strong style="color: #333333;">Avg Confidence:</strong> {avg_confidence:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6c757d;padding:1rem;'>🛡️ Smart Safety Monitor - Local YOLO PPE Analysis</div>", unsafe_allow_html=True)