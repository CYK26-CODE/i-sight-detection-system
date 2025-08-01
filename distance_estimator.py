#!/usr/bin/env python3
"""
Distance Estimation Module for i-sight System
Uses face detection and known measurements to estimate distance
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional

class DistanceEstimator:
    """Distance estimation using face detection and known measurements"""
    
    def __init__(self):
        # Constants for distance calculation
        self.KNOWN_DISTANCE = 50  # cm - distance used for calibration
        self.KNOWN_FACE_WIDTH = 15  # cm - average human face width
        self.KNOWN_PERSON_HEIGHT = 170  # cm - average person height
        self.KNOWN_VEHICLE_WIDTH = 180  # cm - average vehicle width
        
        # Focal length (will be calculated during calibration)
        self.focal_length_face = None
        self.focal_length_person = None
        self.focal_length_vehicle = None
        
        # Face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        
        # Body detection (full body)
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # Calibration status
        self.calibrated = False
        self.calibration_image_path = "calibration.jpg"
        
        print("✅ Distance estimator initialized")
    
    def calibrate_from_image(self, image_path: str = None) -> bool:
        """Calibrate focal length from a calibration image"""
        if image_path is None:
            image_path = self.calibration_image_path
        
        try:
            # Load calibration image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load calibration image: {image_path}")
                return False
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces for calibration
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Use the largest face for calibration
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Calculate focal length
                self.focal_length_face = (w * self.KNOWN_DISTANCE) / self.KNOWN_FACE_WIDTH
                
                print(f"✅ Face calibration successful:")
                print(f"   - Face width in pixels: {w}")
                print(f"   - Known distance: {self.KNOWN_DISTANCE} cm")
                print(f"   - Known face width: {self.KNOWN_FACE_WIDTH} cm")
                print(f"   - Calculated focal length: {self.focal_length_face:.2f}")
                
                # Also detect full body for person height calibration
                bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3)
                if len(bodies) > 0:
                    largest_body = max(bodies, key=lambda x: x[2] * x[3])
                    _, _, _, body_height = largest_body
                    self.focal_length_person = (body_height * self.KNOWN_DISTANCE) / self.KNOWN_PERSON_HEIGHT
                    print(f"   - Person height focal length: {self.focal_length_person:.2f}")
                
                self.calibrated = True
                return True
            else:
                print("❌ No faces detected in calibration image")
                return False
                
        except Exception as e:
            print(f"❌ Calibration failed: {e}")
            return False
    
    def calibrate_from_camera(self, target_distance: float = 50.0) -> bool:
        """Calibrate using live camera feed"""
        print(f"📷 Starting camera calibration at {target_distance} cm distance...")
        print("   Please position a person's face at exactly this distance from the camera")
        print("   Press 'C' to capture calibration, 'Q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera for calibration")
            return False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # Draw detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face: {w}x{h}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw calibration instructions
                cv2.putText(frame, f"Position face at {target_distance}cm", (10, 30),
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
                    
                    self.focal_length_face = (w * target_distance) / self.KNOWN_FACE_WIDTH
                    self.calibrated = True
                    
                    print(f"✅ Camera calibration successful:")
                    print(f"   - Face width: {w} pixels")
                    print(f"   - Target distance: {target_distance} cm")
                    print(f"   - Focal length: {self.focal_length_face:.2f}")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return self.calibrated
    
    def estimate_face_distance(self, face_bbox: List[int]) -> Optional[float]:
        """Estimate distance to a detected face"""
        if not self.calibrated or self.focal_length_face is None:
            return None
        
        try:
            x, y, w, h = face_bbox
            
            # Calculate distance using focal length formula
            # distance = (known_width * focal_length) / pixel_width
            distance = (self.KNOWN_FACE_WIDTH * self.focal_length_face) / w
            
            return max(0, distance)  # Ensure non-negative distance
            
        except Exception as e:
            print(f"❌ Face distance estimation error: {e}")
            return None
    
    def estimate_person_distance(self, person_bbox: List[int]) -> Optional[float]:
        """Estimate distance to a detected person (full body)"""
        if not self.calibrated or self.focal_length_person is None:
            return None
        
        try:
            x, y, w, h = person_bbox
            
            # Use height for more accurate person distance estimation
            distance = (self.KNOWN_PERSON_HEIGHT * self.focal_length_person) / h
            
            return max(0, distance)
            
        except Exception as e:
            print(f"❌ Person distance estimation error: {e}")
            return None
    
    def estimate_vehicle_distance(self, vehicle_bbox: List[int]) -> Optional[float]:
        """Estimate distance to a detected vehicle"""
        if not self.calibrated or self.focal_length_vehicle is None:
            return None
        
        try:
            x, y, w, h = vehicle_bbox
            
            # Use width for vehicle distance estimation
            distance = (self.KNOWN_VEHICLE_WIDTH * self.focal_length_vehicle) / w
            
            return max(0, distance)
            
        except Exception as e:
            print(f"❌ Vehicle distance estimation error: {e}")
            return None
    
    def get_distance_category(self, distance: float) -> str:
        """Categorize distance into zones"""
        if distance < 100:
            return "Very Close"
        elif distance < 200:
            return "Close"
        elif distance < 400:
            return "Medium"
        elif distance < 600:
            return "Far"
        else:
            return "Very Far"
    
    def process_frame_with_distance(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Process frame and return annotated frame with distance information"""
        if not self.calibrated:
            return frame, []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            distance = self.estimate_face_distance([x, y, w, h])
            if distance is not None:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw distance information
                distance_text = f"Face: {distance:.1f}cm"
                category = self.get_distance_category(distance)
                
                # Text background
                text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                             (x + text_size[0], y), (0, 255, 0), -1)
                
                # Text
                cv2.putText(frame, distance_text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Category
                cv2.putText(frame, category, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                detections.append({
                    'type': 'face',
                    'bbox': [x, y, w, h],
                    'distance': distance,
                    'category': category,
                    'confidence': 0.8
                })
        
        # Detect full bodies
        bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in bodies:
            distance = self.estimate_person_distance([x, y, w, h])
            if distance is not None:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Draw distance information
                distance_text = f"Person: {distance:.1f}cm"
                category = self.get_distance_category(distance)
                
                # Text background
                text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                             (x + text_size[0], y), (255, 0, 0), -1)
                
                # Text
                cv2.putText(frame, distance_text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Category
                cv2.putText(frame, category, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                detections.append({
                    'type': 'person',
                    'bbox': [x, y, w, h],
                    'distance': distance,
                    'category': category,
                    'confidence': 0.7
                })
        
        return frame, detections
    
    def get_calibration_status(self) -> dict:
        """Get calibration status and focal lengths"""
        return {
            'calibrated': self.calibrated,
            'focal_length_face': self.focal_length_face,
            'focal_length_person': self.focal_length_person,
            'focal_length_vehicle': self.focal_length_vehicle,
            'known_distance': self.KNOWN_DISTANCE,
            'known_face_width': self.KNOWN_FACE_WIDTH,
            'known_person_height': self.KNOWN_PERSON_HEIGHT
        }

# Test function
def test_distance_estimator():
    """Test the distance estimator"""
    estimator = DistanceEstimator()
    
    # Try to calibrate from image first
    if not estimator.calibrate_from_image():
        print("📷 Image calibration failed, trying camera calibration...")
        if not estimator.calibrate_from_camera():
            print("❌ Both calibration methods failed")
            return
    
    print("✅ Calibration successful! Starting distance estimation...")
    
    # Start real-time distance estimation
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame with distance estimation
            processed_frame, detections = estimator.process_frame_with_distance(frame)
            
            # Display calibration info
            cv2.putText(processed_frame, f"Focal Length: {estimator.focal_length_face:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Detections: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Distance Estimator", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_distance_estimator() 