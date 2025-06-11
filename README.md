# Smart Safety Monitor 🛡️

**Local YOLO PPE Detection System** - Privacy-First Computer Vision

A streamlined Personal Protective Equipment (PPE) detection system built with Streamlit and local YOLO models for enhanced privacy, performance, and offline capability.

## ✨ Features

- **🖼️ Image Analysis** - Upload and analyze images for PPE compliance
- **🎥 Video Processing** - Process video files with frame-by-frame detection  
- **📹 Live Detection** - Real-time webcam monitoring with optimized performance
- **🤖 Local YOLO Model** - No internet required, data never leaves your device
- **⚡ High Performance** - Sub-100ms inference times for smooth real-time detection
- **🔒 Privacy First** - 100% local processing, no cloud dependencies
- **📊 Compliance Metrics** - Detailed safety compliance reporting

## 🚀 Quick Setup

### Prerequisites
- Python 3.8+
- Webcam (for live detection)
- ~2GB disk space for dependencies

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd security-checker

# Install dependencies
pip install -r requirements.txt

# Test your setup
python test_setup.py

# Run the application
streamlit run .streamlit/01_home.py
```

### YOLO Environment (Optional - Better Performance)
For optimal performance, you can set up a separate YOLO environment:
```bash
# Set up isolated YOLO environment
./scripts/setup_yolo_env_separate.sh

# This creates yolo_env/ with optimized dependencies
# The app will automatically detect and use it
```

## 📁 Project Structure

```
├── .streamlit/              # Streamlit applications
│   ├── 01_home.py          # Landing page
│   ├── 02_image.py         # Image analysis
│   ├── 03_video.py         # Video processing
│   └── 04_live.py          # Live detection
├── utils/                  # Utility modules
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Environment configuration
│   ├── yolo_model_manager.py # Optimized model management
│   ├── yolo_server.py      # Persistent YOLO inference server
│   └── local_yolo_inference.py # Legacy subprocess inference
├── scripts/                # Setup and utility scripts
│   ├── setup_yolo_env.sh   # YOLO environment setup
│   └── README.md           # Scripts documentation
├── model/                  # Local YOLO model files
│   └── best.pt             # Trained PPE detection model
├── yolo_env/              # Isolated YOLO environment
├── test_setup.py          # Setup validation script
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

The system uses environment variables for configuration:

```bash
# Local YOLO Model Configuration
LOCAL_MODEL_PATH=model/best.pt

# Default Detection Parameters
CONF_THRESH_DEFAULT=0.5
DETECTION_INTERVAL_DEFAULT=2.0

# YOLO Server Configuration (optional)
YOLO_SERVER_HOST=127.0.0.1
YOLO_SERVER_PORT=8888
```

## 🎯 Detection Classes

The system detects the following PPE items:
- **✅ Hard Hat / Helmet** - Safety headgear compliance
- **✅ Safety Vest** - High-visibility clothing detection
- **❌ No Hard Hat** - Missing headgear violations
- **❌ No Safety Vest** - Missing vest violations
- **👤 Person** - Human detection for context

## ⚡ Performance Optimization

### Direct YOLO Integration
The system uses direct YOLO model integration for optimal performance:

```bash
# Model loads once in memory for fast inference
# Real-time processing with minimal latency
# No external API calls or network dependencies
```

**Performance Benefits:**
- **Fast Inference** - Direct model access (50-100ms per frame)
- **No Network Latency** - Completely offline processing
- **Memory Efficient** - Model loads once and stays in memory
- **Privacy Focused** - No data transmission to external services

### Inference Methods
1. **Direct YOLO** - Fastest (50-100ms per frame)
2. **YOLO Server** - Alternative for distributed processing
3. **Subprocess Fallback** - Legacy compatibility mode

## 📊 Usage Examples

### Image Analysis
1. Navigate to the Image page
2. Upload an image file
3. Adjust confidence threshold
4. View results and download annotated image

### Video Processing
1. Go to the Video Processing page
2. Upload a video file
3. Configure detection parameters
4. Process video and download results

### Live Detection
1. Go to the Live Detection page
2. Allow camera access when prompted
3. Adjust detection settings
4. Monitor real-time PPE compliance

## 🛠️ Development

### Adding New Models
1. Place model file in `model/` directory
2. Update LOCAL_MODEL_PATH in `.env`
3. Test with `python test_setup.py`

### Extending Detection Classes
1. Update color mapping in utility functions
2. Modify class detection logic
3. Update UI class information displays

## 📝 Recent Improvements

- ✅ **Simplified Architecture** - Local-only processing, removed API dependencies
- ✅ **Performance Optimization** - Direct YOLO integration for real-time detection
- ✅ **Enhanced Privacy** - 100% local processing, no data transmission
- ✅ **Streamlined Setup** - Simplified installation and configuration
- ✅ **Better UX** - Clean, focused interface for local model usage
- ✅ **Robust Testing** - Comprehensive setup validation script

## 🔍 Troubleshooting

### Model Issues
- Ensure model file exists: `ls -la model/best.pt`
- Run setup test: `python test_setup.py`
- Check YOLO environment: `./scripts/setup_yolo_env_separate.sh`

### Performance Issues
- Use YOLO environment for better performance
- Increase detection interval for slower hardware
- Ensure adequate system memory (4GB+ recommended)

### Camera Issues
- Allow camera permissions in browser
- Try different browsers (Chrome/Firefox recommended)
- Restart the Streamlit application

## 📄 License

This project is for educational and demonstration purposes. 