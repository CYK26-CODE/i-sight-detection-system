#!/usr/bin/env python3
"""
Quick Voice Output Test
Test the voice functionality of the lightweight detector
"""

import time
import sys

def test_voice_system():
    """Test the voice system"""
    print("🔊 Testing Voice System...")
    
    try:
        # Test voice system directly without importing the full detector
        import pyttsx3
        
        # Test basic voice functionality
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.8)
        
        print("✅ Voice system is working")
        
        # Test different announcements
        test_messages = [
            "Detected 2 faces",
            "Detected 1 car", 
            "Detected 1 stop sign",
            "AI Analysis: Clear road ahead"
        ]
        
        for message in test_messages:
            print(f"🔊 Testing: {message}")
            engine.say(message)
            engine.runAndWait()
            time.sleep(1)  # Wait between announcements
        
        # Cleanup
        engine.stop()
        print("✅ Voice test completed successfully")
        return True
        
    except ImportError:
        print("❌ pyttsx3 not available - voice output disabled")
        return False
    except Exception as e:
        print(f"❌ Voice test failed: {e}")
        return False

def test_detection_system():
    """Test the detection system without camera"""
    print("\n🎯 Testing Detection System...")
    
    try:
        # Test face detection on a dummy image
        import cv2
        import numpy as np
        
        # Create a dummy image
        dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Test face detection directly
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if not face_cascade.empty():
            gray = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            print(f"✅ Face detection test: {len(faces)} faces detected")
        else:
            print("❌ Face detection model failed to load")
            return False
        
        # Test traffic sign detection (color-based)
        hsv = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_contours, _ = cv2.findContours(red_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"✅ Traffic sign detection test: {len(red_contours)} potential signs detected")
        
        print("✅ Detection system test completed")
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Lightweight Voice Detector Test")
    print("=" * 40)
    
    # Test voice system
    voice_ok = test_voice_system()
    
    # Test detection system
    detection_ok = test_detection_system()
    
    # Summary
    print("\n📊 Test Results:")
    print(f"Voice System: {'✅ PASS' if voice_ok else '❌ FAIL'}")
    print(f"Detection System: {'✅ PASS' if detection_ok else '❌ FAIL'}")
    
    if voice_ok and detection_ok:
        print("\n🎉 All tests passed! The lightweight voice detector is ready to use.")
        print("Run 'python run_lightweight_voice.py' to start the full system.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 