# 👁️ i-sight Detection System

A comprehensive real-time computer vision detection system with voice feedback, designed for accessibility and multi-modal detection capabilities.

## 🌟 Features

### 🎯 Detection Capabilities
- **Face Detection**: Real-time face detection with zone-based positioning
- **Vehicle Detection**: YOLOv5-based vehicle detection (cars, trucks, buses, motorcycles)
- **Traffic Sign Detection**: SVM-based traffic sign classification
- **Multi-zone Analysis**: 5-zone detection system (Far Left, Slight Left, Center, Slight Right, Far Right)

### 🔊 Voice System
- **Multi-platform TTS**: Windows SAPI, pyttsx3, PowerShell fallback
- **Intelligent Announcements**: Zone-specific detection announcements
- **Multi-person Detection**: Detailed descriptions for multiple people
- **Accessibility Focus**: Designed for visually disabled users

### 🌐 Web Interfaces
- **Flask Dashboard**: Full-featured web interface with real-time video feed
- **Streamlit App**: Simplified interface with accessibility mode
- **RESTful API**: Complete API for system control and data access

### ♿ Accessibility Features
- **Large Dynamic Toggle**: On/off button with visual feedback
- **Voice Feedback**: Comprehensive audio announcements
- **Simplified Controls**: Easy-to-use interface for disabled users
- **Status Indicators**: Clear visual and audio status feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam/Camera
- Windows (for optimal voice support)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/i-sight-detection-system.git
   cd i-sight-detection-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # For Flask version
   pip install -r requirements_i_sight_flask.txt
   
   # For Streamlit version
   pip install -r requirements_streamlit.txt
   ```

4. **Run the application**

   **Option 1: Flask Application**
   ```bash
   python i_sight_flask_integrated.py
   ```
   Open: http://localhost:5000

   **Option 2: Streamlit Application**
   ```bash
   streamlit run streamlit_app_simple.py
   ```
   Open: http://localhost:8501

## 📁 Project Structure

```
i-sight-detection-system/
├── i_sight_flask_integrated.py      # Main Flask application
├── streamlit_app_simple.py          # Streamlit application
├── i_sight_detector.py              # Core detection system
├── templates/
│   └── index.html                   # Flask web interface
├── Traffic-Sign-Detection/          # Traffic sign detection module
├── requirements_i_sight_flask.txt   # Flask dependencies
├── requirements_streamlit.txt       # Streamlit dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

## 🎮 Usage Guide

### Flask Application

1. **Start the system**
   - Click "Start Detection" in the sidebar
   - View real-time video feed
   - Monitor detection metrics

2. **Voice Controls**
   - Test voice system
   - Force speak custom messages
   - View voice logs

3. **Accessibility Mode**
   - Switch to "Accessibility Mode" tab
   - Use the large toggle button
   - Automatic detection control

### Streamlit Application

1. **Main Dashboard**
   - Real-time video feed
   - Detection metrics
   - System status

2. **Accessibility Mode**
   - Large dynamic toggle button
   - Voice feedback
   - Simplified controls

## 🔧 Configuration

### Camera Settings
The system automatically detects and configures your camera. For optimal performance:

```python
# Camera settings in i_sight_flask_integrated.py
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cap.set(cv2.CAP_PROP_CONTRAST, 100)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
```

### Voice Settings
Configure voice system in the VoiceManager class:

```python
# Voice settings
self.voice_enabled = True
self.voice_rate = 150
self.voice_volume = 0.9
```

## 🌐 Deployment

### Local Deployment
1. Follow the installation steps above
2. Run the desired application
3. Access via localhost

### Streamlit Cloud Deployment
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_i_sight_flask.txt .
RUN pip install -r requirements_i_sight_flask.txt
COPY . .
EXPOSE 5000
CMD ["python", "i_sight_flask_integrated.py"]
```

## 🔌 API Endpoints

### Flask API
- `GET /` - Main dashboard
- `GET /api/status` - System status
- `POST /api/start` - Start detection
- `POST /api/stop` - Stop detection
- `GET /api/voice-log` - Voice log
- `POST /api/voice-force-speak` - Force speak
- `GET /api/system-health` - System health
- `GET /api/debug` - Debug information

## 🛠️ Troubleshooting

### Common Issues

1. **Camera not working**
   ```bash
   python camera_diagnostic.py
   ```

2. **Voice system issues**
   - Check Windows SAPI installation
   - Verify microphone permissions
   - Test with PowerShell TTS

3. **Import errors**
   ```bash
   pip install --upgrade -r requirements_i_sight_flask.txt
   ```

4. **Performance issues**
   - Reduce video resolution
   - Disable unused detection modules
   - Check system resources

### System Requirements
- **Minimum**: 4GB RAM, Intel i3/AMD Ryzen 3
- **Recommended**: 8GB RAM, Intel i5/AMD Ryzen 5
- **Camera**: USB webcam or built-in camera
- **OS**: Windows 10/11 (optimal), Linux, macOS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/yourusername/i-sight-detection-system.git
cd i-sight-detection-system
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements_i_sight_flask.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- YOLOv5 for object detection
- Streamlit for web application framework
- Flask for web framework
- Windows SAPI for voice synthesis

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/i-sight-detection-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/i-sight-detection-system/discussions)
- **Email**: your.email@example.com

## 🔄 Version History

- **v1.0.0** - Initial release with Flask integration
- **v1.1.0** - Added Streamlit support
- **v1.2.0** - Enhanced accessibility features
- **v1.3.0** - Dynamic toggle button and voice improvements

---

**Made with ❤️ for accessibility and computer vision enthusiasts**
