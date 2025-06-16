# Smart Safety Monitor 🛡️

**Local YOLO PPE Detection System** - Privacy-First Computer Vision

A comprehensive Personal Protective Equipment (PPE) detection system built with Streamlit and a locally trained YOLO model. Features real-time webcam monitoring, video processing, and image analysis with complete privacy protection through offline processing.

## ✨ Features

- **🖼️ Image Analysis** - Upload and analyze images for PPE compliance with detailed metrics
- **🎥 Video Processing** - Process video files with frame-by-frame detection and tracking
- **📹 Live Detection** - Real-time webcam monitoring with continuous compliance tracking
- **🤖 Local Trained Model** - Custom-trained YOLO model specifically for PPE detection
- **⚡ High Performance** - Optimized inference with dedicated virtual environment
- **🔒 Privacy First** - 100% local processing, no cloud dependencies or data transmission
- **📊 Compliance Metrics** - Real-time safety compliance reporting and audio alerts
- **🎯 Multi-PPE Detection** - Detects helmets, masks, safety vests, and violations

## 🚀 Quick Setup

### Prerequisites
- Python 3.8+ (3.12 recommended)
- Webcam (for live detection)
- ~4GB disk space for virtual environment and dependencies

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd security-checker

# Activate the dedicated YOLO virtual environment
source yolo_env/bin/activate  # On macOS/Linux
# OR
yolo_env\Scripts\activate     # On Windows

# Install additional dependencies if needed
pip install -r requirements.txt

# Run the application
streamlit run .streamlit/app.py
```

## 📁 Project Structure

```
├── .streamlit/              # Streamlit applications
│   ├── 01_home.py          # Landing page and navigation
│   ├── 02_image.py         # Image analysis interface
│   ├── 03_video.py         # Video processing with tracking
│   └── 04_live.py          # Live webcam detection
├── utils/                  # Core utility modules
│   ├── __init__.py         # Package initialization
│   └── config.py           # Environment configuration
├── model/                  # Local trained YOLO model
│   └── best.pt             # Custom PPE detection model
├── assets/                 # Static assets
│   ├── logo.webp           # Application logo
│   └── emergency-alarm.mp3 # Audio alert sound
├── yolo_env/              # Dedicated virtual environment
│   ├── bin/               # Environment executables
│   ├── lib/               # Python packages
│   └── ...                # Virtual environment files
├── notebooks/             # Development notebooks
│   ├── ppe_v1.ipynb       # Model development
│   └── ppe_v2.ipynb       # Model training iterations
├── requirements.txt       # Python dependencies
├── docker-compose.yaml    # Docker deployment (optional)
└── README.md             # This file
```

## 🔧 Configuration

The system uses environment-based configuration with sensible defaults:

### Model Configuration
- **Local Model**: `model/best.pt` - Custom-trained PPE detection model
- **Detection Classes**: Helmet, Mask, Safety Vest, Person, and violation classes
- **Confidence Threshold**: Adjustable per-page (default varies by use case)

### Virtual Environment
The app uses a dedicated virtual environment (`yolo_env/`) with optimized dependencies:
- **PyTorch**: GPU-accelerated inference (when available)
- **Ultralytics YOLO**: Latest YOLO implementation
- **OpenCV**: Computer vision processing
- **Streamlit**: Web interface framework

## 🎯 Detection Classes

The custom-trained model detects the following PPE items and violations:

### ✅ Positive PPE Detection
- **🪖 Helmet/Hard Hat** - Safety headgear compliance
- **😷 Mask** - Face covering detection  
- **🦺 Safety Vest** - High-visibility clothing detection
- **👤 Person** - Human detection for context

### ❌ Violation Detection
- **NO-Helmet** - Missing headgear violations
- **NO-Mask** - Missing face covering violations
- **NO-Safety Vest** - Missing vest violations

### 🚧 Environment Detection
- **Safety Cone** - Work zone markers
- **Machinery** - Industrial equipment
- **Vehicle** - Construction vehicles

## ⚡ Performance & Architecture

### Direct YOLO Integration
The system uses direct integration with a custom-trained YOLO model:

```bash
# Custom model trained specifically for PPE detection
# Optimized for construction and industrial environments
# Real-time processing with dedicated virtual environment
```

**Architecture Benefits:**
- **Custom Training** - Model specifically trained on PPE scenarios
- **Fast Inference** - Direct model access with optimized environment
- **No Network Dependencies** - Completely offline processing
- **Memory Efficient** - Model loads once and stays in memory
- **Privacy Focused** - No data transmission to external services

### Key Features
1. **Real-time Tracking** - DeepSORT integration for person tracking in videos
2. **Compliance Metrics** - Continuous calculation based on all 3 PPE items
3. **Audio Alerts** - Customizable sound notifications for violations
4. **Multi-format Support** - Images, videos, and live webcam streams

## 📊 Usage Guide

### 🖼️ Image Analysis
1. Navigate to the **Image Analysis** page
2. Upload an image file (JPG, PNG, etc.)
3. Adjust confidence threshold as needed
4. View detection results and compliance metrics
5. Download annotated image with bounding boxes

### 🎥 Video Processing  
1. Go to the **Video Processing** page
2. Upload a video file (MP4, AVI, MOV, etc.)
3. Configure detection parameters and playback speed
4. Process video with frame-by-frame analysis
5. Monitor real-time compliance metrics during processing
6. Download processed video with annotations

### 📹 Live Detection
1. Go to the **Live Detection** page
2. Allow camera access when prompted by browser
3. Adjust detection settings and persistence
4. Monitor real-time PPE compliance
5. Enable audio alerts for immediate violation notifications

## 🛠️ Development & Customization

### Model Management
- **Current Model**: Custom-trained on PPE datasets
- **Model Location**: `model/best.pt`
- **Training Notebooks**: Available in `notebooks/` directory
- **Model Updates**: Replace `best.pt` with new trained models

### Adding New Detection Classes
1. Retrain model with new classes
2. Update color mapping in utility functions  
3. Modify compliance calculation logic
4. Update UI class displays and icons

## 📝 Recent Updates & Features

- ✅ **Custom Model Training** - Dedicated PPE model trained on construction datasets
- ✅ **Dedicated Virtual Environment** - Optimized dependency management with `yolo_env/`
- ✅ **Enhanced Live Detection** - Continuous compliance tracking with audio alerts
- ✅ **Video Processing Improvements** - DeepSORT tracking and real-time metrics
- ✅ **Privacy-First Architecture** - Complete offline processing, no external dependencies
- ✅ **Streamlined Interface** - Clean, focused UI without unnecessary model status messages
- ✅ **Real-time Compliance** - Dynamic metrics that update as PPE is added/removed
- ✅ **Audio Alert System** - Customizable sound notifications for safety violations

## 🔍 Troubleshooting

### Environment Issues
- **Activate Environment**: Ensure `yolo_env` is activated before running
- **Dependencies**: Check `requirements.txt` for any missing packages
- **Python Version**: Use Python 3.8+ (3.12 recommended)

### Model Issues
- **Model Location**: Verify `model/best.pt` exists and is accessible
- **Model Format**: Ensure model is compatible with Ultralytics YOLO
- **Memory**: Ensure sufficient RAM (4GB+ recommended)

### Performance Issues
- **GPU Acceleration**: PyTorch will use GPU if available (CUDA/MPS)
- **Detection Interval**: Increase interval for slower hardware
- **Confidence Threshold**: Adjust for optimal detection vs. performance balance

### Camera/Browser Issues
- **Permissions**: Allow camera access in browser settings
- **Browser Support**: Use Chrome, Firefox, Safari, or Edge
- **WebRTC**: Ensure browser supports WebRTC for live detection
- **Refresh**: Try refreshing the page or restarting Streamlit

## 🐳 Docker Deployment (Optional)

For containerized deployment, use the included Docker configuration:

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application at http://localhost:8501
```

## 🤝 Contributing

1. **Model Improvements**: Train new models with additional PPE classes
2. **Performance Optimization**: Enhance inference speed and accuracy
3. **UI/UX Enhancements**: Improve user interface and experience
4. **Documentation**: Update documentation and examples

## 📄 License

This project is designed for educational and safety monitoring purposes. Please ensure compliance with local privacy and surveillance regulations when deploying in workplace environments.