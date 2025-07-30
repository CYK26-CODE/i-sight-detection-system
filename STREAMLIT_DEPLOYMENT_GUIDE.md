# 🚀 i-sight Streamlit Deployment Guide

## 📋 Overview

This guide will help you deploy the i-sight detection system on Streamlit Cloud, making it accessible via web browser with a modern, responsive interface.

## 🎯 Features of Streamlit Version

- **Real-time Video Feed**: Live camera stream with detection overlays
- **Interactive Controls**: Start/stop detection, voice testing, force speak
- **Live Metrics**: Real-time detection counts, FPS, processing time
- **Detection Zones**: Visual representation of active detection zones
- **Voice System**: Full voice feedback integration with Windows SAPI
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 📦 Prerequisites

### 1. Local Development Setup
```bash
# Create virtual environment
python -m venv streamlit_env
streamlit_env\Scripts\activate  # Windows
# OR
source streamlit_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements_streamlit.txt
```

### 2. Required Files Structure
```
your-project/
├── streamlit_app.py              # Main Streamlit application
├── i_sight_flask_integrated.py   # Detection system (shared)
├── requirements_streamlit.txt    # Dependencies
├── yolov5s.pt                    # YOLO model (if using)
├── models/                       # YOLO models directory
├── utils/                        # YOLO utilities
└── Traffic-Sign-Detection/       # Traffic sign detection (if using)
```

## 🌐 Streamlit Cloud Deployment

### Method 1: Streamlit Cloud (Recommended)

#### Step 1: Prepare Your Repository
1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial i-sight Streamlit app"
   git branch -M main
   git remote add origin https://github.com/yourusername/i-sight-streamlit.git
   git push -u origin main
   ```

2. **Repository Structure**
   ```
   i-sight-streamlit/
   ├── streamlit_app.py
   ├── i_sight_flask_integrated.py
   ├── requirements_streamlit.txt
   ├── .gitignore
   └── README.md
   ```

#### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and branch
5. Set main file path: `streamlit_app.py`
6. Click "Deploy"

### Method 2: Local Streamlit Server

#### Step 1: Run Locally
```bash
# Activate virtual environment
streamlit_env\Scripts\activate

# Run Streamlit app
streamlit run streamlit_app.py
```

#### Step 2: Access Application
- Open browser: http://localhost:8501
- The app will automatically reload on code changes

## 🔧 Configuration Options

### 1. Environment Variables
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "localhost"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### 2. Camera Configuration
Modify camera settings in `streamlit_app.py`:
```python
# In initialize_detector() function
def initialize_detector():
    if st.session_state.detector is None:
        try:
            # Custom camera settings
            detector = ISightDetector()
            detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            detector.cap.set(cv2.CAP_PROP_FPS, 30)
            st.session_state.detector = detector
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize detector: {e}")
            return False
    return True
```

## 🎮 Usage Instructions

### 1. Starting the Application
1. Open the Streamlit app URL
2. Wait for the interface to load
3. Check system status in the sidebar

### 2. Detection Controls
- **🚀 Start**: Begin real-time detection
- **🛑 Stop**: Stop detection and release camera
- **📸 Screenshot**: Capture current frame

### 3. Voice System
- **🔊 Test Voice**: Test the voice system
- **🎤 Force Speak**: Send custom voice messages
- **📝 Voice Log**: View recent voice activity

### 4. Monitoring
- **Live Video**: Real-time camera feed with detections
- **Metrics**: Face count, vehicle count, FPS, processing time
- **Detection Zones**: Visual zone indicators
- **System Status**: Camera, voice, and model status

## 🔒 Security Considerations

### 1. Camera Access
- Streamlit Cloud doesn't support camera access
- Use local deployment for camera functionality
- Consider using uploaded images for cloud deployment

### 2. Voice System
- Windows SAPI only works on Windows
- Use pyttsx3 fallback for cross-platform support
- Disable voice for cloud deployment

### 3. Model Files
- Large model files (>100MB) may cause deployment issues
- Use model hosting services (Hugging Face, etc.)
- Consider using lighter models for cloud deployment

## 🚀 Advanced Deployment Options

### 1. Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Heroku Deployment
Create `Procfile`:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

### 3. AWS/GCP Deployment
- Use EC2/GCE instances
- Set up reverse proxy with nginx
- Configure SSL certificates

## 🐛 Troubleshooting

### Common Issues

#### 1. Camera Not Working
```python
# Try different camera indices
detector.cap = cv2.VideoCapture(0)  # Try 0, 1, 2...
```

#### 2. Voice System Issues
```python
# Check voice availability
if not VOICE_AVAILABLE:
    st.warning("Voice system not available")
```

#### 3. Model Loading Errors
```python
# Check model file path
model_path = 'yolov5s.pt'
if not os.path.exists(model_path):
    st.error("Model file not found")
```

#### 4. Performance Issues
```python
# Reduce frame rate for better performance
time.sleep(1/15)  # 15 FPS instead of 30
```

### Debug Mode
```bash
# Run with debug information
streamlit run streamlit_app.py --logger.level=debug
```

## 📊 Performance Optimization

### 1. Frame Rate Control
```python
# Adjust frame rate based on performance
target_fps = 15  # Lower for better performance
time.sleep(1/target_fps)
```

### 2. Image Resolution
```python
# Reduce resolution for better performance
detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

### 3. Detection Frequency
```python
# Process every nth frame
if frame_count % 3 == 0:  # Process every 3rd frame
    # Run detection
```

## 🎨 Customization

### 1. UI Themes
Modify CSS in `streamlit_app.py`:
```python
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #your-color1 0%, #your-color2 100%);
    }
</style>
""", unsafe_allow_html=True)
```

### 2. Detection Zones
Modify zone names in `streamlit_app.py`:
```python
zone_names = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"]
```

### 3. Voice Messages
Customize voice announcements in `i_sight_flask_integrated.py`:
```python
def announce_detections(self, person_count, vehicle_count, traffic_sign_count, detected_zones):
    # Custom voice messages
    if person_count == 2:
        message = f"Two individuals detected in zones {detected_zones}"
```

## 📈 Monitoring and Analytics

### 1. Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log detection events
logger.info(f"Detection: {person_count} people, {vehicle_count} vehicles")
```

### 2. Metrics Collection
```python
# Track usage metrics
st.metric("Total Detections", total_detections)
st.metric("Average FPS", avg_fps)
st.metric("Detection Accuracy", accuracy)
```

## 🔄 Updates and Maintenance

### 1. Regular Updates
```bash
# Update dependencies
pip install --upgrade -r requirements_streamlit.txt

# Update Streamlit
pip install --upgrade streamlit
```

### 2. Backup Strategy
- Regular backups of configuration files
- Version control for all code changes
- Backup detection models and data

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Streamlit documentation
3. Check system logs for error messages
4. Test with minimal configuration first

## 🎉 Success Checklist

- [ ] Streamlit app runs locally
- [ ] Camera access works
- [ ] Voice system functions
- [ ] Detection system operates correctly
- [ ] UI is responsive and user-friendly
- [ ] Deployment to cloud platform successful
- [ ] Performance meets requirements
- [ ] Security measures implemented
- [ ] Documentation complete
- [ ] Testing completed

---

**Happy Deploying! 🚀** 