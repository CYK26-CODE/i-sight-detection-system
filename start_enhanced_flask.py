#!/usr/bin/env python3
"""
Enhanced Flask i-sight System Startup Script
Starts the Flask integrated system with all new distance-aware and vehicle detection features
"""

import os
import sys
import time
import subprocess
import threading
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies for enhanced i-sight system...")
    
    dependencies = [
        ('cv2', 'OpenCV'),
        ('flask', 'Flask'),
        ('numpy', 'NumPy'),
        ('threading', 'Threading (built-in)'),
        ('queue', 'Queue (built-in)'),
        ('datetime', 'DateTime (built-in)')
    ]
    
    missing_deps = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Missing")
            missing_deps.append(name)
    
    # Check optional dependencies
    optional_deps = [
        ('torch', 'PyTorch (for YOLOv5 vehicle detection)'),
        ('win32com.client', 'Windows SAPI COM (for voice)'),
        ('pyttsx3', 'pyttsx3 (for voice fallback)'),
        ('ultralytics', 'YOLO11n (for enhanced object detection)')
    ]
    
    print("\n🔍 Checking optional dependencies...")
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - Optional, will use fallback")
    
    if missing_deps:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install opencv-python flask numpy")
        return False
    
    print("\n✅ All required dependencies available!")
    return True

def display_enhanced_features():
    """Display information about enhanced features"""
    print("\n🚀 Enhanced i-sight Features")
    print("=" * 50)
    print("📍 Distance-aware Face Detection:")
    print("   - Estimates distance to faces using focal length formula")
    print("   - Categories: very close (<50cm), close (<100cm), medium (<200cm), far (>200cm)")
    print("   - Voice: 'Person detected in Center zone very close at 45 centimeters'")
    print()
    print("🚗 Vehicle Detection with Direction:")
    print("   - Detects cars, bicycles, motorcycles, buses, trucks")
    print("   - Direction analysis: left, right, front zones")
    print("   - Distance estimation for each vehicle type")
    print("   - Voice: 'car approaching from left close at 2.5 meters'")
    print()
    print("🔊 Enhanced Voice Announcements:")
    print("   - Multi-vehicle summaries: '3 vehicles detected, 1 from left, 2 from right'")
    print("   - Distance categories in appropriate units (cm/meters)")
    print("   - Windows SAPI COM, pyttsx3, and PowerShell fallbacks")
    print()
    print("🌐 Flask Web Interface:")
    print("   - Real-time detection data API with distance/direction info")
    print("   - Enhanced JSON output with vehicle directions and face distances")
    print("   - Live screenshot updates with annotated detections")
    print("=" * 50)

def start_flask_server():
    """Start the enhanced Flask server"""
    print("🌐 Starting Enhanced Flask i-sight Server...")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        sys.path.append('.')
        from i_sight_flask_integrated import app, ISightDetector
        
        print("✅ Enhanced Flask app imported successfully")
        print("📡 Server will be available at: http://localhost:5000")
        print("🎮 Controls:")
        print("   - Open http://localhost:5000 in browser for web interface")
        print("   - API endpoints: /api/detection-data, /api/screenshot, /api/voice-test")
        print("   - Press Ctrl+C to stop the server")
        print()
        print("🔊 Voice System Status:")
        print("   - Enhanced voice announcements will include distance and direction")
        print("   - Test with: curl -X POST http://localhost:5000/api/voice-test")
        print()
        print("📊 Enhanced Detection Features Active:")
        print("   - Distance-aware face detection in 5 zones")
        print("   - Vehicle detection with direction (left/right/front)")
        print("   - Enhanced voice announcements with detailed information")
        print("=" * 50)
        
        # Start the Flask development server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Disable debug mode for production-like experience
            threaded=True,
            use_reloader=False
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all files are in the correct location")
        return False
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        return False

def main():
    """Main startup function"""
    print("🎯 Enhanced Flask i-sight System Startup")
    print("=" * 70)
    print("🔮 AI-Powered Computer Vision with Distance Awareness")
    print("👁️  Real-time face detection with precise distance measurement")
    print("🚗 Vehicle detection with approach direction and distance")
    print("🔊 Intelligent voice announcements for accessibility")
    print("🌐 Flask web interface with enhanced API")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Display enhanced features
    display_enhanced_features()
    
    # Check if camera is available
    print("📷 Checking camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera available")
            cap.release()
        else:
            print("⚠️  Camera not available - system will run but may not detect")
    except Exception as e:
        print(f"⚠️  Camera check failed: {e}")
    
    print("\n🚀 Starting enhanced system...")
    
    # Start Flask server
    try:
        start_flask_server()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("✅ Enhanced Flask i-sight system shutdown complete")
        return True
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("👋 Goodbye!")
        else:
            print("❌ Startup failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Startup interrupted by user")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
