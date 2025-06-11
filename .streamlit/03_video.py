import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import subprocess
import json
import base64
import sys
from typing import List, Dict, Tuple
from inference_sdk import InferenceHTTPClient

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
    """Run local YOLO inference using subprocess with separate environment"""
    try:
        # Encode image to base64
        image_b64 = encode_image_to_base64(image)
        
        # Get paths
        script_path = os.path.join(os.path.dirname(__file__), "..", "local_yolo_inference.py")
        
        # Look for separate YOLO environment
        yolo_env_path = os.path.join(os.path.dirname(__file__), "..", "yolo_env")
        yolo_python = os.path.join(yolo_env_path, "bin", "python")
        
        # Choose Python executable
        if os.path.exists(yolo_python):
            python_executable = yolo_python
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

st.set_page_config(
    page_title="PPE Detection – Video Upload",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for styling ---
st.markdown("""
<style>
    .main-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);padding: 2rem;border-radius: 15px;margin-bottom: 2rem;text-align: center;box-shadow: 0 8px 32px rgba(0,0,0,0.1);}
    .metric-card {background: white;padding: 1.5rem;border-radius: 10px;box-shadow: 0 4px 16px rgba(0,0,0,0.1);border-left: 4px solid #667eea;margin: 1rem 0;color: #333333 !important;}
    .metric-card h4 {color: #333333 !important;margin-bottom: 1rem;}
    .metric-card p {color: #555555 !important;margin: 0.5rem 0;}
    .status-badge {display: inline-block;padding: 0.5rem 1rem;border-radius: 20px;color: white;font-weight: bold;margin: 0.5rem 0;}
    .status-safe {background: linear-gradient(45deg, #28a745, #20c997);} .status-warning {background: linear-gradient(45deg, #ffc107, #fd7e14);} .status-danger {background: linear-gradient(45deg, #dc3545, #e74c3c);}
    .upload-area {background: #f8f9fa;border: 2px dashed #dee2e6;border-radius: 15px;padding: 3rem;text-align: center;margin: 2rem 0;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1 style="color:white;margin:0;font-size:2.5rem;">🎥 Video PPE Detection</h1>
    <p style="color:rgba(255,255,255,0.9);margin:0.5rem 0 0 0;font-size:1.1rem;">Upload a high-res short video for PPE detection and compliance analysis</p>
</div>
""", unsafe_allow_html=True)

# --- API and Model Config ---
API_KEY = "mDauQAfDrFWieIsSqti6"
MODEL_ID_DEFAULT = "pbe-detection/4"

@st.cache_resource(show_spinner=False)
def get_rf_client() -> InferenceHTTPClient:
    return InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

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

def get_color_for_class(name: str) -> Tuple[int,int,int]:
    return {
        "helmet": (0,255,0),
        "safety-vest": (0,255,255),
        "person": (255,0,0),
        "no-helmet": (0,0,255),
        "no-safety-vest": (0,165,255)
    }.get(name.lower(), (128,128,128))

def analyze_frame(image: np.ndarray, client, model_id: str, conf_thresh: float, overlap_thresh: float):
    if model_id == "best.pt (Local)":
        # Use local model via subprocess
        return run_local_inference(image, conf_thresh)
    else:
        # Use Roboflow API
        resp = client.infer(image, model_id=model_id)
        preds = resp.get("predictions", [])
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
            cv2.rectangle(out, (x1,y1-sz[1]-8), (x1+sz[0], y1), col, -1) # type: ignore
            cv2.putText(out, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return filtered, out

def calculate_metrics(detections: List[Dict]) -> Dict:
    # Only count unique helmets/vests per frame, not cumulative
    frame_helmet = {}
    frame_vest = {}
    frame_no_helmet = {}
    frame_no_vest = {}
    for d in detections:
        f = d.get('frame', -1)
        cls = d['class'].lower()
        if cls == 'helmet':
            frame_helmet.setdefault(f, 0)
            frame_helmet[f] += 1
        elif cls == 'no-helmet':
            frame_no_helmet.setdefault(f, 0)
            frame_no_helmet[f] += 1
        elif 'vest' in cls and 'no-' not in cls:
            frame_vest.setdefault(f, 0)
            frame_vest[f] += 1
        elif cls == 'no-safety-vest':
            frame_no_vest.setdefault(f, 0)
            frame_no_vest[f] += 1
    # Count unique frames with at least one detection
    helmets = sum(1 for v in frame_helmet.values() if v > 0)
    vests = sum(1 for v in frame_vest.values() if v > 0)
    violations = sum(1 for v in frame_no_helmet.values() if v > 0) + sum(1 for v in frame_no_vest.values() if v > 0)
    total_frames = len(set(list(frame_helmet.keys()) + list(frame_vest.keys()) + list(frame_no_helmet.keys()) + list(frame_no_vest.keys())))
    compliance = ((total_frames - violations) / total_frames * 100) if total_frames else 0
    return {
        "helmets": helmets,
        "vests": vests,
        "violations": violations,
        "compliance": compliance
    }

# --- Sidebar controls ---
with st.sidebar:
    st.header("⚙️ Detection Parameters")
    # Model selection
    model_options = ["ppe-factory-bmdcj/2", "pbe-detection/4", "safety-pyazl/1"]
    if LOCAL_MODEL_AVAILABLE:
        model_options.append("best.pt (Local)")
        st.info(f"✅ Local model found at: {LOCAL_MODEL_PATH}")
    else:
        st.warning("⚠️ Local model not found - using API models only")
    
    model_id = st.selectbox(
        "🎯 Model Selection",
        model_options,
        index=1,
        help="Choose the PPE detection model"
    )
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    overlap_thresh = st.slider("IoU Threshold", 0.0, 1.0, 0.3, 0.05)

# --- Main content ---
st.markdown("### Upload Video for PPE Detection")
video_file = st.file_uploader("Choose a video", type=["mp4","avi","mov"], help="Upload a short, high-res video")

if video_file:
    t_start = time.time()
    # Save video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0
    st.info(f"Video: {frame_count} frames | {duration:.1f}s | {width}x{height} @ {fps} FPS")
    client = get_rf_client()

    # --- Live video playback with detection ---
    st.markdown("### ▶️ Live PPE Detection Feed")
    play = st.button("Play", key="play_btn")
    pause = st.button("Pause", key="pause_btn")
    stop = st.button("Stop/Reset", key="stop_btn")
    speed = st.selectbox("Playback Speed", ["0.1x", "0.25x", "0.5x", "1x", "2x", "4x"], index=3)
    speed_map = {"0.1x": 10.0, "0.25x": 4.0, "0.5x": 2.0, "1x": 1.0, "2x": 0.5, "4x": 0.25}
    frame_delay = (1.0 / fps) * speed_map[speed] if fps else 0.04
    detection_interval = st.slider("Detection Interval (frames)", 1, 30, 3, 1, help="Run detection every Nth frame for smooth playback")

    # Session state for playback
    if 'video_frame_idx' not in st.session_state or stop:
        st.session_state.video_frame_idx = 0
        st.session_state.video_playing = False
        st.session_state.detections_live = []
        st.session_state.last_detections = []
    if play:
        st.session_state.video_playing = True
    if pause:
        st.session_state.video_playing = False

    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_bar = st.progress(0.0)

    while st.session_state.video_playing and st.session_state.video_frame_idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.video_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Only run detection every Nth frame
        if st.session_state.video_frame_idx % detection_interval == 0:
            dets, annotated = analyze_frame(frame, client, model_id, conf_thresh, overlap_thresh)
            st.session_state.last_detections = dets
            # Draw detections on frame
            display_frame = annotated
        else:
            # For skipped frames, just draw last detections (approximate)
            display_frame = frame.copy()
            for p in st.session_state.last_detections:
                x1 = int(p["x"] - p["width"]/2)
                y1 = int(p["y"] - p["height"]/2)
                x2 = int(p["x"] + p["width"]/2)
                y2 = int(p["y"] + p["height"]/2)
                col = get_color_for_class(p["class"])
                cv2.rectangle(display_frame, (x1,y1), (x2,y2), col, 2)
                label = f"{p['class']} ({p['confidence']:.2f})"
                sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame, (x1,y1-sz[1]-8), (x1+sz[0], y1), col, -1)
                cv2.putText(display_frame, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # For metrics, only count detections on detection frames
        if st.session_state.video_frame_idx % detection_interval == 0:
            for d in st.session_state.last_detections:
                d['frame'] = st.session_state.video_frame_idx
                d['timestamp'] = st.session_state.video_frame_idx / fps if fps else 0
                st.session_state.detections_live.append(d)
        frame_placeholder.image(display_frame, channels="BGR", caption=f"Frame {st.session_state.video_frame_idx+1}/{frame_count}", use_container_width=True)
        # Live metrics (remove cumulative counting, just show frame info)
        if st.session_state.last_detections:
            frame_helmets = sum(1 for d in st.session_state.last_detections if d['class'].lower() == 'helmet')
            frame_no_helmets = sum(1 for d in st.session_state.last_detections if d['class'].lower() == 'no-helmet')
            frame_vests = sum(1 for d in st.session_state.last_detections if 'vest' in d['class'].lower() and 'no-' not in d['class'].lower())
            frame_no_vests = sum(1 for d in st.session_state.last_detections if d['class'].lower() == 'no-safety-vest')
            metrics_placeholder.markdown(f"**Frame {st.session_state.video_frame_idx+1}/{frame_count}**  🪖 Helmets: {frame_helmets}  ❌ No Helmet: {frame_no_helmets}  🦺 Vests: {frame_vests}  ❌ No Vest: {frame_no_vests}")
        else:
            metrics_placeholder.markdown(f"**Frame {st.session_state.video_frame_idx+1}/{frame_count}**  No detections.")
        progress_bar.progress((st.session_state.video_frame_idx+1)/frame_count)
        st.session_state.video_frame_idx += 1
        time.sleep(frame_delay)
    cap.release()
    os.unlink(tmp_path)
    if st.session_state.video_frame_idx >= frame_count:
        st.session_state.video_playing = False
        st.success(f"Detection complete in {time.time()-t_start:.1f}s")
# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6c757d;padding:1rem;'>🛡️ Smart Safety Monitor - Video PPE Detection</div>", unsafe_allow_html=True)
