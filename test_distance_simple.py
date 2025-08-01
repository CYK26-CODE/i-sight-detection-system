#!/usr/bin/env python3
"""
Simple Distance Estimation Test
Tests the distance estimation module without complex integration
"""

import cv2
import numpy as np
import time

def test_basic_distance_estimation():
    """Test basic distance estimation using face detection"""
    
    # Constants for distance calculation
    KNOWN_DISTANCE = 50  # cm - distance used for calibration
    KNOWN_FACE_WIDTH = 15  # cm - average human face width
    
    # Load face detection cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("❌ Face detection model failed to load")
        return
    
    print("✅ Face detection model loaded")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("✅ Camera opened successfully")
    
    # Calibration phase
    print("\n📷 CALIBRATION PHASE")
    print("Please position your face at exactly 50cm from the camera")
    print("Press 'C' to calibrate when ready, 'Q' to quit")
    
    focal_length = None
    calibrated = False
    
    try:
        while not calibrated:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Draw detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {w}x{h}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw instructions
            cv2.putText(frame, "Position face at 50cm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'C' to calibrate, 'Q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and len(faces) > 0:
                # Use the largest face for calibration
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                focal_length = (w * KNOWN_DISTANCE) / KNOWN_FACE_WIDTH
                calibrated = True
                
                print(f"✅ Calibration successful!")
                print(f"   - Face width: {w} pixels")
                print(f"   - Known distance: {KNOWN_DISTANCE} cm")
                print(f"   - Known face width: {KNOWN_FACE_WIDTH} cm")
                print(f"   - Calculated focal length: {focal_length:.2f}")
                break
        
        if not calibrated:
            print("❌ Calibration failed or cancelled")
            return
        
        # Distance estimation phase
        print("\n📏 DISTANCE ESTIMATION PHASE")
        print("Move around to test distance estimation")
        print("Press 'Q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Calculate distance
                distance = (KNOWN_FACE_WIDTH * focal_length) / w
                
                # Categorize distance
                if distance < 100:
                    category = "Very Close"
                    color = (0, 0, 255)  # Red
                elif distance < 200:
                    category = "Close"
                    color = (0, 165, 255)  # Orange
                elif distance < 400:
                    category = "Medium"
                    color = (0, 255, 255)  # Yellow
                elif distance < 600:
                    category = "Far"
                    color = (0, 255, 0)  # Green
                else:
                    category = "Very Far"
                    color = (255, 255, 255)  # White
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw distance information
                distance_text = f"Face: {distance:.1f}cm"
                
                # Text background
                text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                             (x + text_size[0], y), color, -1)
                
                # Text
                cv2.putText(frame, distance_text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Category
                cv2.putText(frame, category, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display info
            cv2.putText(frame, f"Focal Length: {focal_length:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Faces Detected: {len(faces)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Distance Estimation", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test completed")

if __name__ == "__main__":
    print("🚀 Starting Distance Estimation Test")
    test_basic_distance_estimation() 