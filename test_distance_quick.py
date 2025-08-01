#!/usr/bin/env python3
"""
Distance Estimation Test Script
Tests the distance estimation functionality
"""

import cv2
import numpy as np
import time

def test_distance_estimation():
    """Test distance estimation with camera"""
    
    # Constants
    KNOWN_DISTANCE = 50  # cm
    KNOWN_FACE_WIDTH = 15  # cm
    
    # Load face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("✅ Camera opened")
    print("📷 Position your face at 50cm and press 'C' to calibrate")
    
    focal_length = None
    calibrated = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if calibrated and focal_length:
                    # Calculate distance
                    distance = (KNOWN_FACE_WIDTH * focal_length) / w
                    cv2.putText(frame, f"{distance:.1f}cm", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"{w}x{h}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instructions
            if not calibrated:
                cv2.putText(frame, "Press 'C' to calibrate at 50cm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"Focal Length: {focal_length:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press 'Q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Distance Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and len(faces) > 0 and not calibrated:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                focal_length = (w * KNOWN_DISTANCE) / KNOWN_FACE_WIDTH
                calibrated = True
                print(f"✅ Calibrated! Focal length: {focal_length:.2f}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_distance_estimation()
