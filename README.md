# Smart Safety Monitor 🛡️

AI-powered PPE Detection System with Real-time Computer Vision

A comprehensive Personal Protective Equipment (PPE) detection system built with Streamlit, offering both cloud-based API inference and local YOLO model support for enhanced privacy and performance.

## ✨ Features

- **🖼️ Image Analysis** - Upload and analyze images for PPE compliance
- **🎥 Video Processing** - Process video files with frame-by-frame detection
- **📹 Live Detection** - Real-time webcam monitoring with optimized performance
- **🔄 Dual Model Support** - Choose between Roboflow API or local YOLO models
- **⚡ High Performance** - Persistent YOLO server for smooth real-time inference
- **🔒 Security First** - Environment-based configuration management
- **📊 Compliance Metrics** - Detailed safety compliance reporting

## 🚀 Quick Setup

### Option 1: Standard Setup (API only)
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your Roboflow API key

# Run the application
streamlit run .streamlit/01_home.py
```

### Option 2: Full Setup (API + Local Models)
```bash
# Set up local YOLO environment
./scripts/setup_yolo_env_separate.sh

# Install main dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start YOLO server for optimal performance (optional)
yolo_env/bin/python utils/yolo_server.py model/best.pt

# Run the application
streamlit run .streamlit/01_home.py
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
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

The system uses environment variables for secure configuration:

```bash
# Roboflow API Configuration
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_API_URL=https://detect.roboflow.com

# Available Models
API_MODELS=ppe-factory-bmdcj/2,pbe-detection/4,safety-pyazl/1

# Default Parameters
MODEL_ID_DEFAULT=ppe-factory-bmdcj/2
CONF_THRESH_DEFAULT=0.5
OVERLAP_THRESH_DEFAULT=0.3
DETECTION_INTERVAL_DEFAULT=8.0
```

## 🎯 Detection Classes

The system detects the following PPE items:
- **✅ Hard Hat / Helmet** - Safety headgear compliance
- **✅ Safety Vest** - High-visibility clothing detection
- **❌ No Hard Hat** - Missing headgear violations
- **❌ No Safety Vest** - Missing vest violations
- **👤 Person** - Human detection for context

## ⚡ Performance Optimization

### YOLO Server Architecture
For optimal real-time performance, the system includes a persistent YOLO server:

```bash
# Start the YOLO server
yolo_env/bin/python utils/yolo_server.py model/best.pt
```

**Benefits:**
- Model loads once and stays in memory
- Sub-100ms inference times for live detection
- No subprocess overhead
- Smooth real-time video processing

### Inference Methods (in order of performance)
1. **HTTP Server** - Fastest (50-100ms per frame)
2. **Optimized In-Process** - Fast (100-200ms per frame)  
3. **Subprocess Fallback** - Slower (2-5s per frame, legacy)

## 📊 Usage Examples

### Image Analysis
1. Navigate to the Image page
2. Upload an image file
3. Adjust confidence and IoU thresholds
4. Select detection model
5. View results and download annotated image

### Live Detection
1. Go to the Live Detection page
2. Start the YOLO server (for best performance)
3. Select "best.pt (Local)" model
4. Enable live detection
5. Allow camera access
6. Monitor real-time PPE compliance

## 🛠️ Development

### Adding New Models
1. Place model file in `model/` directory
2. Update model detection logic in each page
3. Add model configuration to `.env`

### Extending Detection Classes
1. Update color mapping in `yolo_server.py`
2. Modify class detection logic in inference functions
3. Update UI class information displays

## 📝 Recent Improvements

- ✅ **Security Enhancement** - Moved all hardcoded credentials to environment variables
- ✅ **Performance Optimization** - Implemented persistent YOLO server for real-time detection
- ✅ **UI Cleanup** - Removed verbose notification messages
- ✅ **Configuration Management** - Centralized config system
- ✅ **Error Handling** - Improved error handling and fallback mechanisms
- ✅ **Code Organization** - Better project structure and documentation

## 🔍 Troubleshooting

### Local Model Issues
- Ensure YOLO environment is set up: `./scripts/setup_yolo_env_separate.sh`
- Check model file exists: `ls -la model/best.pt`
- Start YOLO server for best performance

### Performance Issues
- Use HTTP server method for live detection
- Increase detection interval for slower hardware
- Consider using API models for very low-end devices

### API Issues
- Verify Roboflow API key in `.env`
- Check internet connectivity
- Monitor API usage quotas

## 📄 License

This project is for educational and demonstration purposes. 