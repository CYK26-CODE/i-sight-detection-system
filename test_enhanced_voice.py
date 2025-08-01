#!/usr/bin/env python3
"""
Enhanced Voice Announcement Test
Tests the voice announcements with distance information
"""

import cv2
import numpy as np
import time
import threading
import queue

def test_enhanced_voice_announcements():
    """Test enhanced voice announcements with distance information"""
    
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
    print("📷 Starting enhanced voice announcement test...")
    print("   Move your face around to hear distance announcements")
    print("   Press Ctrl+C to stop")
    
    focal_length = None
    calibrated = False
    frame_count = 0
    last_announcement_time = 0
    announcement_interval = 3  # Announce every 3 seconds
    
    def create_voice_announcement(message):
        """Simulate voice announcement"""
        print(f"🔊 VOICE: {message}")
    
    def announce_detection_with_distance(person_count, detected_zones, distance_info):
        """Create enhanced announcement with distance"""
        if person_count == 0:
            return
        
        zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        
        if person_count == 1:
            if detected_zones:
                zone_name = zone_names[detected_zones[0]]
                message = f"Person detected in {zone_name} zone{distance_info}"
            else:
                message = f"Person detected{distance_info}"
        elif person_count == 2:
            if len(detected_zones) == 2:
                zone1_name = zone_names[detected_zones[0]]
                zone2_name = zone_names[detected_zones[1]]
                message = f"Two people detected: one in {zone1_name} zone and one in {zone2_name} zone{distance_info}"
            elif len(detected_zones) == 1:
                zone_name = zone_names[detected_zones[0]]
                message = f"Two people detected in {zone_name} zone{distance_info}"
            else:
                message = f"Two people detected{distance_info}"
        else:
            if detected_zones:
                zone_names_list = [zone_names[zone] for zone in detected_zones]
                if len(zone_names_list) == 1:
                    message = f"{person_count} people detected in {zone_names_list[0]} zone{distance_info}"
                else:
                    zone_list = ", ".join(zone_names_list[:-1]) + f" and {zone_names_list[-1]}"
                    message = f"{person_count} people detected in {zone_list} zones{distance_info}"
            else:
                message = f"{person_count} people detected{distance_info}"
        
        create_voice_announcement(message)
    
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
            
            # Process detections and create announcements
            if calibrated and focal_length and len(faces) > 0:
                # Calculate distance for closest face
                closest_face = min(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = closest_face
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
                
                distance_info = f" at {distance:.1f} centimeters, {category} range"
                
                # Determine zones (simplified - just use face center)
                detected_zones = []
                for (fx, fy, fw, fh) in faces:
                    center_x = fx + fw // 2
                    # Simple zone calculation (5 zones)
                    zone_width = frame.shape[1] // 5
                    zone_idx = min(center_x // zone_width, 4)
                    if zone_idx not in detected_zones:
                        detected_zones.append(zone_idx)
                
                # Announce every few seconds
                current_time = time.time()
                if current_time - last_announcement_time > announcement_interval:
                    announce_detection_with_distance(len(faces), detected_zones, distance_info)
                    last_announcement_time = current_time
                
                # Print status every 30 frames
                if frame_count % 30 == 0:
                    print(f"📏 {len(faces)} face(s) detected at {distance:.1f}cm ({category})")
            
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

def test_voice_message_examples():
    """Test various voice message examples"""
    print("🔊 Testing voice message examples:")
    print("=" * 50)
    
    examples = [
        "Person detected in Center zone at 45.2 centimeters, Very Close range",
        "Two people detected: one in Slight Left zone and one in Slight Right zone at 67.8 centimeters, Close range",
        "Three people detected in Far Left, Center and Far Right zones at 123.4 centimeters, Medium range",
        "Car detected in Center zone at 89.1 centimeters, Close range",
        "Two objects detected: chair and table in Slight Right zone at 156.7 centimeters, Medium range"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
        time.sleep(1)  # Pause between examples
    
    print("=" * 50)
    print("✅ Voice message examples completed")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Voice Announcement Test")
    print("=" * 50)
    
    # First show voice message examples
    test_voice_message_examples()
    
    print("\n" + "=" * 50)
    print("📏 Starting real-time distance announcement test...")
    test_enhanced_voice_announcements() 