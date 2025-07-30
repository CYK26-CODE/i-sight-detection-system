# i-sight Flask Integrated System

A unified computer vision detection system that combines the powerful i-sight detector with a modern Flask web interface for real-time monitoring and control.

## 🚀 Features

- **Real-time Detection**: Face, vehicle, and traffic sign detection using AI/ML
- **Voice Feedback**: Multi-method voice announcements (Windows SAPI, pyttsx3, PowerShell)
- **Web Dashboard**: Modern, responsive web interface for monitoring and control
- **Live Video Stream**: Real-time video feed with detection overlays
- **Detection Zones**: 5-zone detection system for spatial awareness
- **API Endpoints**: RESTful API for integration with other systems
- **Screenshot Capture**: Save detection results as images
- **System Monitoring**: Real-time statistics and status monitoring

## 📋 Requirements

- Python 3.8 or higher
- Windows 10/11 (for voice features)
- Webcam or camera device
- 4GB+ RAM recommended
- Internet connection (for initial model downloads)

## 🛠️ Installation

### Quick Start (Windows)

1. **Download and Extract**
   ```bash
   # Clone or download the project files
   ```

2. **Run the Startup Script**
   ```bash
   # Double-click or run:
   start_i_sight_flask.bat
   ```
   
   This script will:
   - Check Python installation
   - Create virtual environment
   - Install dependencies
   - Start the system

### Manual Installation

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements_i_sight_flask.txt
   ```

3. **Run the System**
   ```bash
   python i_sight_flask_integrated.py
   ```

## 🌐 Web Interface

Once running, access the system at:

- **Dashboard**: http://localhost:5000
- **Video Stream**: http://localhost:5000/video-feed
- **API Documentation**: http://localhost:5000/api/

### Dashboard Features

- **Live Video Feed**: Real-time camera stream with detection overlays
- **System Status**: Detection, camera, and voice system status
- **Detection Zones**: Visual representation of active detection zones
- **Real-time Statistics**: Face count, vehicle count, traffic signs, FPS
- **Control Panel**: Start/stop detection, voice test, screenshot capture
- **Activity Log**: Real-time system events and detection logs

## 🎮 Controls

### Web Interface Controls

- **🚀 Start Detection**: Begin real-time detection
- **🛑 Stop Detection**: Stop detection system
- **🔊 Test Voice**: Test voice feedback system
- **📸 Screenshot**: Capture current frame with detections

### Keyboard Shortcuts (if running in console mode)

- `q` - Quit system
- `s` - Save screenshot
- `v` - Toggle voice feedback
- `t` - Test voice system
- `a` - Check audio settings

## 🔧 API Endpoints

### System Control

- `GET /api/status` - Get system status
- `POST /api/start-detection` - Start detection
- `POST /api/stop-detection` - Stop detection
- `GET /api/voice-test` - Test voice system

### Data Access

- `GET /api/latest-detection` - Get latest detection data
- `GET /api/stats` - Get detection statistics
- `GET /api/screenshot` - Get current screenshot
- `GET /video-feed` - Live video stream

### Example API Usage

```bash
# Get system status
curl http://localhost:5000/api/status

# Start detection
curl -X POST http://localhost:5000/api/start-detection

# Get latest detection data
curl http://localhost:5000/api/latest-detection
```

## 🎯 Detection Features

### Face Detection
- Uses Haar cascades for fast, reliable detection
- 5-zone spatial detection system
- Real-time confidence scoring
- Voice announcements for detected faces

### Vehicle Detection
- YOLOv5-based detection (when available)
- Multiple vehicle class support
- Configurable confidence thresholds
- Bounding box visualization

### Traffic Sign Detection
- Custom traffic sign classification
- Support for common traffic signs
- Real-time sign type identification
- Confidence-based filtering

### Voice Feedback
- **Windows SAPI COM**: Primary method (most reliable)
- **pyttsx3**: Fallback method
- **PowerShell TTS**: System fallback
- Rate-limited announcements to prevent spam
- Emergency audio feedback when TTS fails

## 📊 Detection Zones

The system divides the camera view into 5 zones:

1. **Far Left** - Leftmost detection area
2. **Slight Left** - Left-center detection area
3. **Center** - Central detection area
4. **Slight Right** - Right-center detection area
5. **Far Right** - Rightmost detection area

Each zone is monitored independently and provides spatial awareness for detected objects.

## 🔧 Configuration

### Detection Parameters

```python
# Face detection
confidence_threshold = 0.3
lightweight_mode = True

# Vehicle detection
vehicle_conf_threshold = 0.25
vehicle_iou_threshold = 0.45

# Voice system
voice_cooldown = 2.0  # seconds between announcements
```

### Camera Settings

```python
# Camera configuration
frame_width = 640
frame_height = 480
fps = 30
buffer_size = 1
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Check camera permissions
   - Ensure camera is not in use by other applications
   - Try different camera index (0, 1, 2...)

2. **Voice Not Working**
   - Check Windows Volume Mixer for python.exe
   - Ensure speakers/headphones are connected
   - Try voice test button in web interface

3. **Detection Not Starting**
   - Check camera connection
   - Verify model files are present
   - Check console for error messages

4. **Web Interface Not Loading**
   - Ensure port 5000 is not in use
   - Check firewall settings
   - Verify Flask installation

### Performance Optimization

1. **Reduce FPS**: Lower frame rate for better performance
2. **Lightweight Mode**: Enable for faster processing
3. **Model Selection**: Use smaller models for speed
4. **Zone Reduction**: Reduce detection zones if needed

## 📁 File Structure

```
i-sight-flask-integrated/
├── i_sight_flask_integrated.py    # Main integrated system
├── requirements_i_sight_flask.txt # Dependencies
├── start_i_sight_flask.bat       # Windows startup script
├── templates/
│   └── index.html                # Web dashboard
├── models/                       # AI models
├── yolov5s.pt                   # YOLOv5 model (download separately)
└── README.md                    # This file
```

## 🔄 Updates and Maintenance

### Updating Dependencies
```bash
pip install --upgrade -r requirements_i_sight_flask.txt
```

### Model Updates
- Download latest YOLOv5 models from official repository
- Update traffic sign models as needed
- Retrain custom models for specific use cases

### System Updates
- Regular security updates for Flask and dependencies
- Performance optimizations
- New detection features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review console output for error messages
3. Check system requirements
4. Create an issue with detailed information

## 🎉 Acknowledgments

- OpenCV for computer vision capabilities
- YOLOv5 for object detection
- Flask for web framework
- Windows SAPI for voice synthesis
- Open source community for various libraries

---

**i-sight Flask Integrated System** - Bringing advanced computer vision to the web with voice feedback and real-time monitoring. 