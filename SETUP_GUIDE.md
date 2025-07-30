# 🚀 Setup Guide: Computer Vision Detection System

This guide will help you set up and run the complete computer vision detection system with backend server and frontend application.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Camera**: Webcam or USB camera
- **OS**: Windows, macOS, or Linux

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)
- **CPU**: Multi-core processor (4+ cores recommended)
- **Camera**: HD webcam (720p or higher)

## 🛠️ Installation Steps

### Step 1: Clone/Download Project
```bash
# If using git
git clone <your-repository-url>
cd computer-vision-detection-system

# Or download and extract the project files
```

### Step 2: Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

#### Option A: Install All Dependencies (Recommended)
```bash
# Install backend dependencies
pip install -r requirements_backend.txt

# Install frontend dependencies
pip install -r requirements_frontend.txt
```

#### Option B: Install Core Dependencies Only
```bash
# Core packages
pip install fastapi uvicorn streamlit streamlit-webrtc
pip install opencv-python torch torchvision
pip install numpy pandas plotly requests websockets
pip install scikit-learn scikit-image pillow
```

### Step 4: Verify Installation
```bash
# Check if all packages are installed
python -c "import fastapi, streamlit, cv2, torch; print('✅ All packages installed successfully')"
```

## 🚀 Running the System

### Option 1: Automatic Startup (Recommended)
```bash
# Run the complete system with one command
python start_system.py
```

This will:
- Start the backend server on port 8000
- Start the frontend application on port 8501
- Open the web interface automatically
- Monitor both services

### Option 2: Manual Startup

#### Start Backend Server
```bash
# Terminal 1: Start backend
python backend_server.py
```

#### Start Frontend Application
```bash
# Terminal 2: Start frontend
streamlit run frontend_app.py --server.port 8501
```

## 🌐 Accessing the Application

### Web Interface
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Default Credentials
- No authentication required for local development
- Configure authentication for production deployment

## 🎛️ Using the Application

### 1. System Status
- Check the status indicators in the header
- Verify camera connection and model loading
- Monitor FPS and detection performance

### 2. Detection Controls
- **Face Detection**: Toggle on/off in sidebar
- **Vehicle Detection**: Toggle on/off in sidebar
- **Traffic Sign Detection**: Toggle on/off in sidebar

### 3. Configuration
- **Confidence Thresholds**: Adjust detection sensitivity
- **Frame Processing**: Control processing frequency
- **Performance Settings**: Optimize for your hardware

### 4. Analytics
- **Real-time Metrics**: View current detection counts
- **Performance Charts**: Monitor FPS and processing time
- **Detection Timeline**: Track detections over time

## 🔧 Configuration

### Backend Configuration
Edit `backend_server.py` to modify:
```python
# Server settings
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8000       # Server port

# Detection settings
DEFAULT_CONFIG = DetectionConfig(
    enable_faces=True,
    enable_vehicles=True,
    enable_traffic_signs=True,
    confidence_threshold=0.3,
    vehicle_confidence_threshold=0.25,
    process_every_n_frames=2
)
```

### Frontend Configuration
Edit `frontend_app.py` to modify:
```python
# Backend connection
BACKEND_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000/ws"

# Streamlit settings
st.set_page_config(
    page_title="Computer Vision Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Camera Not Detected
```bash
# Check camera permissions
# Windows: Allow camera access in settings
# macOS: Grant camera permissions to terminal/IDE
# Linux: Check /dev/video* devices

# Test camera with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened())"
```

#### 2. Model Loading Errors
```bash
# Check model files exist
ls runs/train/exp12/weights/best.pt
ls Traffic-Sign-Detection/Traffic-Sign-Detection/data_svm.dat

# Verify PyTorch installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

#### 3. Port Already in Use
```bash
# Check what's using the ports
# Windows:
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# macOS/Linux:
lsof -i :8000
lsof -i :8501

# Kill processes if needed
# Windows:
taskkill /PID <process_id> /F
# macOS/Linux:
kill -9 <process_id>
```

#### 4. WebRTC Issues
```bash
# Install additional dependencies
pip install aiortc av

# Check browser compatibility
# Use Chrome/Edge for best WebRTC support
```

#### 5. Performance Issues
```python
# Reduce processing frequency
process_every_n_frames = 5  # Process every 5th frame

# Lower resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Use CPU only
device = 'cpu'
```

### Debug Mode
Enable debug output by setting environment variables:
```bash
# Windows:
set DEBUG_MODE=1
set LOG_LEVEL=DEBUG

# macOS/Linux:
export DEBUG_MODE=1
export LOG_LEVEL=DEBUG
```

## 📊 Performance Optimization

### For Better Performance
1. **Use GPU**: Install CUDA-enabled PyTorch
2. **Reduce Resolution**: Lower camera resolution
3. **Increase Frame Skip**: Process fewer frames
4. **Close Other Applications**: Free up system resources

### For Better Accuracy
1. **Adjust Confidence Thresholds**: Fine-tune detection sensitivity
2. **Improve Lighting**: Ensure good camera lighting
3. **Stable Camera**: Use tripod or stable mount
4. **Clean Lens**: Keep camera lens clean

## 🔒 Security Considerations

### For Production Deployment
1. **Enable Authentication**: Add user login system
2. **Use HTTPS**: Configure SSL certificates
3. **Restrict Access**: Limit to specific IP addresses
4. **Log Monitoring**: Implement security logging
5. **Regular Updates**: Keep dependencies updated

### Environment Variables
```bash
# Security settings
export SECRET_KEY="your-secret-key"
export ALLOWED_HOSTS="localhost,127.0.0.1"
export DEBUG=False
```

## 📈 Monitoring and Logging

### System Monitoring
- **CPU Usage**: Monitor processing load
- **Memory Usage**: Check RAM consumption
- **GPU Usage**: Monitor if using GPU acceleration
- **Network**: Monitor API request volume

### Log Files
```bash
# Backend logs
tail -f backend.log

# Frontend logs
tail -f streamlit.log

# System logs
# Windows: Event Viewer
# macOS/Linux: /var/log/
```

## 🚀 Deployment

### Local Network Deployment
```bash
# Allow external access
# Backend:
uvicorn backend_server:app --host 0.0.0.0 --port 8000

# Frontend:
streamlit run frontend_app.py --server.address 0.0.0.0 --server.port 8501
```

### Cloud Deployment
1. **Docker**: Create Docker containers
2. **Cloud Platforms**: Deploy to AWS, Azure, or GCP
3. **Load Balancing**: Use multiple instances
4. **Auto-scaling**: Configure based on demand

## 📞 Support

### Getting Help
1. **Check Logs**: Review error messages
2. **Documentation**: Read the README.md
3. **Issues**: Report bugs with detailed information
4. **Community**: Join discussion forums

### Useful Commands
```bash
# Check system status
curl http://localhost:8000/status

# Test API endpoints
curl http://localhost:8000/latest-detection

# Monitor processes
ps aux | grep python

# Check dependencies
pip list | grep -E "(fastapi|streamlit|opencv|torch)"
```

---

**Happy detecting! 🚦**

For more information, refer to the main README.md file. 