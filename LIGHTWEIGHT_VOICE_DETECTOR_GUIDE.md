# 🎯 Lightweight Voice Detection System

A complete, optimized detection system with voice output for each action. Built for performance and smooth operation.

## ✨ Features

### 🎤 Voice Output
- **Real-time voice announcements** for each detection
- **Rate-limited announcements** to prevent spam
- **Background voice processing** for smooth operation
- **Configurable voice settings** (speed, volume)

### 🔍 Detection Capabilities
- **Face Detection**: Haar Cascade classifier
- **Vehicle Detection**: YOLOv5 with vehicle classes (car, truck, bus, motorcycle, bicycle)
- **Traffic Sign Detection**: Color-based detection (red for stop signs, blue for other signs)
- **AI Analysis**: Gemini API integration for scene understanding

### ⚡ Performance Optimizations
- **Low resolution**: 320x240 for fast processing
- **Optimized camera settings**: 30 FPS with minimal buffer
- **Frame skipping**: Intelligent processing to maintain performance
- **Lightweight models**: Efficient detection algorithms

### 📊 Output & Monitoring
- **Real-time FPS display**
- **Detection counters** on screen
- **JSON results saving** with timestamps
- **Auto-save functionality** every 30 seconds

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install all required packages
python -m pip install -r requirements_lightweight_voice.txt

# Or install individually
python -m pip install opencv-python numpy pyttsx3 google-generativeai torch torchvision ultralytics Pillow requests
```

### 2. Set Up Gemini API (Optional)
```bash
# Set your Gemini API key for AI analysis
export GEMINI_API_KEY="your_api_key_here"

# On Windows
set GEMINI_API_KEY=your_api_key_here
```

### 3. Run the System
```bash
# Use the launcher (recommended)
python run_lightweight_voice.py

# Or run directly
python lightweight_voice_detector.py
```

## 🎮 Controls

- **`q`**: Quit the application
- **`s`**: Save results manually
- **`Ctrl+C`**: Force quit

## 📁 Files Created

### Core Files
- `lightweight_voice_detector.py` - Main detection system
- `run_lightweight_voice.py` - Smart launcher with dependency checking
- `requirements_lightweight_voice.txt` - Package dependencies
- `test_voice_output.py` - Test script for verification

### Output Files
- `lightweight_detection_results.json` - Detection results with timestamps

## 🔧 Configuration

### Camera Settings
```python
self.frame_width = 320    # Resolution width
self.frame_height = 240   # Resolution height  
self.fps = 30            # Target FPS
```

### Detection Settings
```python
self.confidence_threshold = 0.5  # Minimum confidence for detections
self.gemini_cooldown = 5.0       # Seconds between AI analysis calls
```

### Voice Settings
```python
self.engine.setProperty('rate', 150)    # Speech speed
self.engine.setProperty('volume', 0.8)  # Volume level
self.cooldown = 3.0                     # Seconds between same announcements
```

## 🎯 Voice Announcements

The system provides voice feedback for:

- **Face Detection**: "Detected X face(s)"
- **Vehicle Detection**: "Detected X vehicle(s): car, truck, etc."
- **Traffic Signs**: "Detected X traffic sign(s): stop_sign, etc."
- **AI Analysis**: "AI Analysis: [description]"

## 🔍 Detection Details

### Face Detection
- Uses OpenCV Haar Cascade classifier
- Detects frontal faces in real-time
- Confidence threshold: 0.5

### Vehicle Detection
- YOLOv5 model with COCO dataset
- Vehicle classes: car, truck, bus, motorcycle, bicycle
- GPU acceleration if available

### Traffic Sign Detection
- Color-based detection using HSV color space
- Red detection for stop signs
- Blue detection for other traffic signs
- Shape analysis for stop sign validation

### AI Analysis (Gemini)
- Scene understanding and description
- Safety concern identification
- Traffic sign and signal detection
- Rate-limited to prevent API spam

## 📊 Performance Metrics

- **Target FPS**: 30
- **Resolution**: 320x240 (optimized for speed)
- **Memory Usage**: Minimal (lightweight models)
- **CPU Usage**: Optimized for real-time processing

## 🛠️ Troubleshooting

### Voice Issues
```bash
# Reinstall voice package
python -m pip uninstall pyttsx3
python -m pip install pyttsx3
```

### Camera Issues
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Failed')"
```

### Model Loading Issues
```bash
# Clear torch cache
rm -rf ~/.cache/torch/hub/
# Reinstall ultralytics
python -m pip install --force-reinstall ultralytics
```

### Performance Issues
- Reduce resolution in `lightweight_voice_detector.py`
- Disable vehicle detection by setting `YOLO_AVAILABLE = False`
- Increase `gemini_cooldown` for fewer AI calls

## 🎉 Success Indicators

✅ **System Ready**: All dependencies installed  
✅ **Camera Working**: Camera access confirmed  
✅ **Voice Active**: Voice announcements working  
✅ **Detection Running**: Real-time detection active  
✅ **FPS Stable**: Consistent 25-30 FPS  

## 📈 Advanced Usage

### Custom Voice Messages
```python
# In lightweight_voice_detector.py
self.voice.announce('custom', 'Your custom message here')
```

### Custom Detection Classes
```python
# Add new vehicle classes
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'your_class']
```

### Integration with Other Systems
```python
# Import and use in other scripts
from lightweight_voice_detector import LightweightVoiceDetector
detector = LightweightVoiceDetector()
# Use detector methods
```

## 🔄 Updates & Maintenance

- **Regular Updates**: Keep packages updated
- **Model Updates**: YOLOv5 models auto-update
- **API Key Rotation**: Rotate Gemini API keys regularly
- **Performance Monitoring**: Monitor FPS and memory usage

---

**🎯 Ready to use!** The lightweight voice detector provides a complete, optimized detection system with voice feedback for each action. Perfect for real-time monitoring and accessibility applications. 