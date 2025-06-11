# Security-checker Project 🦺

PPE Detection System with both API and Local Model Support

## Quick Setup

```bash
# Set up local YOLO model environment (optional)
./scripts/setup_yolo_env_separate.sh

# Run the app
streamlit run .streamlit/02_image.py
```

## Project Structure

- `.streamlit/` - Streamlit applications (Image, Video, Live detection)
- `scripts/` - Setup and utility scripts
- `model/` - Local YOLO model files
- `yolo_env/` - Separate Python environment for local models (auto-created)

## 