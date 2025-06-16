import streamlit as st 
import os

# Set page configuration
st.set_page_config(
    page_title="Smart Safety Monitor - Local YOLO PPE Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix the logo path - use absolute path from current file location
logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "logo.webp")

# CSS for consistent styling with dark/light theme compatibility
st.markdown("""
<style>
    /* Global text color override */
    .main .block-container {
        color: white !important;
    }
    
    /* Force all text elements to be white */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stText, .stWrite {
        color: white !important;
    }
    
    /* Force all markdown content to be white */
    .main .block-container .stMarkdown {
        color: white !important;
    }
    
    /* Force paragraph text to be white */
    .main .block-container p {
        color: white !important;
    }
    
    /* Force div text to be white */
    .main .block-container div {
        color: white !important;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Feature cards with forced visibility */
    .feature-card {
        background: #ffffff !important;
        color: #000000 !important;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        border: 1px solid #e0e0e0;
    }
    
    /* Content cards with strong contrast */
    .content-card {
        background: #f8f9fa !important;
        color: #212529 !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 2px solid #dee2e6;
    }
    
    /* Force content card text to be black */
    .content-card h3, .content-card p, .content-card div {
        color: #212529 !important;
    }
    
    /* Override global white text for content cards */
    .content-card * {
        color: #212529 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: #ffffff !important;
        color: #000000 !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Highlighted text */
    .highlight {
        color: #667eea !important;
        font-weight: bold;
    }
    
    /* Success text */
    .success-text {
        color: #28a745 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Display logo and main header
if os.path.exists(logo_path):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_path, width=300)

st.markdown("""
<div class="main-header">
    <h1 style="color:white;margin:0;font-size:3rem;">🛡️ Smart Safety Monitor</h1>
    <h2 style="color:rgba(255,255,255,0.9);margin:1rem 0;font-size:1.5rem;">Local YOLO PPE Detection System</h2>
    <p style="color:rgba(255,255,255,0.8);margin:0;font-size:1.2rem;">Privacy-First Computer Vision for Safety Compliance</p>
</div>
""", unsafe_allow_html=True)

# Key features section with fallback
st.markdown('<h2 style="color: white !important;">🚀 Key Features</h2>', unsafe_allow_html=True)
st.markdown('<p style="color: white !important; font-weight: bold; font-size: 1.2rem;">Local YOLO PPE Detection System - Privacy-First Computer Vision</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="content-card">
        <h3 style="color: #212529 !important;">🖼️ Image Analysis</h3>
        <p style="color: #212529 !important;">Upload images for instant PPE compliance detection</p>
        <div style="color: #28a745 !important; font-weight: bold;">
            ✓ Fast inference<br>
            ✓ No data sent to cloud<br>
            ✓ High accuracy detection
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Fallback content
    st.info("**Image Analysis**: Upload and analyze images for PPE compliance")

with col2:
    st.markdown("""
    <div class="content-card">
        <h3 style="color: #212529 !important;">🎥 Video Processing</h3>
        <p style="color: #212529 !important;">Process video files frame-by-frame for safety analysis</p>
        <div style="color: #28a745 !important; font-weight: bold;">
            ✓ Batch processing<br>
            ✓ Detailed statistics<br>
            ✓ Downloadable results
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Fallback content
    st.info("**Video Processing**: Process videos frame-by-frame")

with col3:
    st.markdown("""
    <div class="content-card">
        <h3 style="color: #212529 !important;">📹 Live Detection</h3>
        <p style="color: #212529 !important;">Real-time webcam monitoring for safety compliance</p>
        <div style="color: #28a745 !important; font-weight: bold;">
            ✓ Real-time processing<br>
            ✓ Instant alerts<br>
            ✓ Continuous monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Fallback content
    st.info("**Live Detection**: Real-time webcam monitoring")

# Technical specifications
st.markdown("---")
st.markdown('<h2 style="color: white !important;">🔧 Technical Specifications</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p style="color: white !important; font-weight: bold; font-size: 1.1rem;">🤖 Detection Capabilities:</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: white !important;">• <strong>Hard Hats/Helmets</strong> - Safety headgear detection</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: white !important;">• <strong>Safety Vests</strong> - High-visibility clothing</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: white !important;">• <strong>Person Detection</strong> - Human identification</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: white !important;">• <strong>Violation Detection</strong> - Missing PPE alerts</p>', unsafe_allow_html=True)

with col2:
    st.markdown('<p style="color: white !important; font-weight: bold; font-size: 1.1rem;">⚡ Performance Features:</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: white !important;">• <strong>Local Processing</strong> - No internet required</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: white !important;">• <strong>YOLO v8 Model</strong> - State-of-the-art accuracy</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: white !important;">• <strong>Real-time Inference</strong> - < 100ms per frame</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: white !important;">• <strong>Privacy First</strong> - Data never leaves your device</p>', unsafe_allow_html=True)

# Getting started section
st.markdown("---")
st.markdown('<h2 style="color: white !important;">🎯 Getting Started</h2>', unsafe_allow_html=True)
st.markdown('<p style="color: white !important; font-weight: bold; font-size: 1.1rem;">Quick Start Guide:</p>', unsafe_allow_html=True)
st.markdown('<p style="color: white !important;">1. <strong>Navigate</strong> to any page using the sidebar menu</p>', unsafe_allow_html=True)
st.markdown('<p style="color: white !important;">2. <strong>Check model status</strong> - Ensure local YOLO model is loaded</p>', unsafe_allow_html=True)
st.markdown('<p style="color: white !important;">3. <strong>Upload content</strong> - Images, videos, or use live camera</p>', unsafe_allow_html=True)
st.markdown('<p style="color: white !important;">4. <strong>Adjust settings</strong> - Configure confidence thresholds</p>', unsafe_allow_html=True)
st.markdown('<p style="color: white !important;">5. <strong>View results</strong> - Get instant PPE compliance reports</p>', unsafe_allow_html=True)

# Model information
st.markdown("---")
st.markdown('<h2 style="color: white !important;">📊 Model Information</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Model Type", "YOLOv8", "Custom PPE Detection")
with col2:
    st.metric("⚡ Performance", "< 100ms", "Inference time")
with col3:
    st.metric("🔒 Privacy", "100%", "Local processing")

# Footer and navigation
st.markdown("---")
st.success("✅ **Application Ready!** Use the sidebar to navigate to Image Analysis, Video Processing, or Live Detection pages.")

st.markdown("""
<div style="text-align:center;padding:2rem;color:#6c757d;">
    <h3>🛡️ Smart Safety Monitor</h3>
    <p>Powered by Local YOLO v8 • Built with Streamlit • Privacy-First Design</p>
    <p><em>Ensuring workplace safety through intelligent computer vision</em></p>
</div>
""", unsafe_allow_html=True)