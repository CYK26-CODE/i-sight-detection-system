#!/usr/bin/env python3
"""
Headless Distance Estimation Test
Tests distance estimation without GUI windows
"""

import cv2
import numpy as np
import time

def test_distance_estimation_headless():
    """Test distance estimation without GUI"""
    
    # Constants
    KNOWN_DISTANCE = 50  # cm
    KNOWN_FACE_WIDTH = 15  # cm
    
    # Load face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("❌ Face detection model failed to load")
        return
    
    print("✅ Face detection model loaded")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("✅ Camera opened successfully")
    print("📷 Starting distance estimation test...")
    print("   Move your face around to see distance measurements")
    print("   Press Ctrl+C to stop")
    
    focal_length = None
    calibrated = False
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                continue
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Auto-calibrate on first face detection
            if not calibrated and len(faces) > 0:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                focal_length = (w * KNOWN_DISTANCE) / KNOWN_FACE_WIDTH
                calibrated = True
                print(f"✅ Auto-calibrated! Focal length: {focal_length:.2f}")
                print(f"   - Face width: {w} pixels")
                print(f"   - Known distance: {KNOWN_DISTANCE} cm")
                print(f"   - Known face width: {KNOWN_FACE_WIDTH} cm")
            
            # Process detections
            if calibrated and focal_length and len(faces) > 0:
                for i, (x, y, w, h) in enumerate(faces):
                    # Calculate distance
                    distance = (KNOWN_FACE_WIDTH * focal_length) / w
                    
                    # Categorize distance
                    if distance < 100:
                        category = "Very Close"
                    elif distance < 200:
                        category = "Close"
                    elif distance < 400:
                        category = "Medium"
                    elif distance < 600:
                        category = "Far"
                    else:
                        category = "Very Far"
                    
                    # Print results every 30 frames (about 1 second)
                    if frame_count % 30 == 0:
                        print(f"📏 Face {i+1}: {distance:.1f}cm ({category}) - Size: {w}x{h}")
            
            # Print status every 60 frames
            if frame_count % 60 == 0:
                if calibrated:
                    print(f"🔄 Status: {len(faces)} face(s) detected, Focal length: {focal_length:.1f}")
                else:
                    print("🔄 Status: Waiting for face detection to calibrate...")
            
            # Brief pause
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cap.release()
        print("✅ Camera released")
        print(f"📊 Test summary:")
        print(f"   - Frames processed: {frame_count}")
        print(f"   - Calibrated: {calibrated}")
        if calibrated:
            print(f"   - Focal length: {focal_length:.2f}")

def test_camera_only():
    """Simple camera test without face detection"""
    print("📷 Testing camera access...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    
    print("✅ Camera opened successfully")
    
    # Test frame capture
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured successfully")
        print(f"   - Frame size: {frame.shape}")
        print(f"   - Frame type: {frame.dtype}")
        
        # Calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        print(f"   - Mean brightness: {mean_brightness:.2f}")
        
        if mean_brightness > 10:
            print("✅ Camera appears to be working properly")
            cap.release()
            return True
        else:
            print("⚠️ Camera frame is very dark - check camera settings")
            cap.release()
            return False
    else:
        print("❌ Failed to capture frame")
        cap.release()
        return False

if __name__ == "__main__":
    print("🚀 Starting Headless Distance Estimation Test")
    print("=" * 50)
    
    # First test camera access
    if test_camera_only():
        print("\n" + "=" * 50)
        print("📏 Starting distance estimation test...")
        test_distance_estimation_headless()
    else:
        print("❌ Camera test failed - cannot proceed with distance estimation") 