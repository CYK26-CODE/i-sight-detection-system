import cv2
import numpy as np
import time
import os
from pathlib import Path

class LightweightFaceDetector:
    def __init__(self):
        # Initialize camera with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for speed
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 30 FPS is sufficient
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Load lightweight detection models
        self.load_lightweight_models()
        
        # Detection zones
        self.zones = []
        self.zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        
        # Colors
        self.colors = {
            'zone': (0, 255, 0),
            'detection': (0, 0, 255),
            'text': (255, 255, 255),
            'curve': (255, 255, 0),
            'face': (255, 0, 255),
            'background': (0, 0, 0)
        }
        
        # Optimized parameters
        self.confidence_threshold = 0.3
        self.curve_amplitude = 80  # Smaller for performance
        
        # Performance settings
        self.process_every_n_frames = 2  # Process every 2nd frame
        self.frame_count = 0
        
    def load_lightweight_models(self):
        """Load lightweight, GPU-efficient detection models"""
        print("Loading lightweight detection models...")
        
        # Use lightweight Haar cascades (CPU-based, no GPU)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        
        # Alternative lightweight cascades
        self.face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Check if models loaded
        if self.face_cascade.empty():
            print("Warning: Primary face cascade not loaded")
        if self.face_cascade2.empty():
            print("Warning: Secondary face cascade not loaded")
        
        print("Lightweight models loaded successfully!")
    
    def setup_detection_zones(self, frame_width, frame_height):
        """Setup detection zones with optimized curved boundaries"""
        zone_width = frame_width // 5
        self.zones = []
        
        for i in range(5):
            x1 = i * zone_width
            x2 = (i + 1) * zone_width
            margin_y = int(frame_height * 0.2)  # Larger margins for smaller resolution
            y1 = margin_y
            y2 = frame_height - margin_y
            self.zones.append((x1, y1, x2, y2))
    
    def draw_optimized_curves(self, frame):
        """Draw optimized curved boundaries for performance"""
        frame_height, frame_width = frame.shape[:2]
        
        # Simplified curve calculation for speed
        top_curve_points = []
        bottom_curve_points = []
        
        # Use fewer points for better performance
        for x in range(0, frame_width, 10):  # Every 10 pixels
            curve_offset = int(self.curve_amplitude * np.sin(np.pi * x / frame_width))
            
            top_y = int(frame_height * 0.2) + curve_offset
            bottom_y = int(frame_height * 0.8) - curve_offset
            
            top_curve_points.append([x, top_y])
            bottom_curve_points.append([x, bottom_y])
        
        # Draw curves efficiently
        top_curve_points = np.array(top_curve_points, np.int32)
        bottom_curve_points = np.array(bottom_curve_points, np.int32)
        
        cv2.polylines(frame, [top_curve_points], False, self.colors['curve'], 3)
        cv2.polylines(frame, [bottom_curve_points], False, self.colors['curve'], 3)
    
    def draw_detection_zones(self, frame):
        """Draw optimized detection zone overlay"""
        frame_height, frame_width = frame.shape[:2]
        
        # Draw curves
        self.draw_optimized_curves(frame)
        
        # Draw vertical lines efficiently
        for i in range(1, 5):
            x = i * (frame_width // 5)
            start_y = int(frame_height * 0.2) + int(self.curve_amplitude * np.sin(np.pi * x / frame_width))
            end_y = int(frame_height * 0.8) - int(self.curve_amplitude * np.sin(np.pi * x / frame_width))
            cv2.line(frame, (x, start_y), (x, end_y), self.colors['zone'], 2)
        
        # Draw zone labels efficiently
        for i, (x1, y1, x2, y2) in enumerate(self.zones):
            center_x = (x1 + x2) // 2
            label_y = y1 - 20
            
            cv2.putText(frame, self.zone_names[i], (center_x - 40, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
    
    def detect_faces_lightweight(self, gray):
        """Lightweight face detection optimized for performance"""
        detections = []
        
        # Use optimized parameters for speed
        face_params = [
            (self.face_cascade, 1.1, 3),   # Primary cascade
            (self.face_cascade2, 1.15, 2), # Secondary cascade
        ]
        
        for cascade, scale_factor, min_neighbors in face_params:
            if not cascade.empty():
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(30, 30),
                    maxSize=(150, 150)
                )
                
                for (x, y, w, h) in faces:
                    # Estimate person size from face
                    person_height = int(h * 6)  # Face is ~1/6 of body
                    person_width = int(w * 2.5)  # Approximate body width
                    
                    # Center the person on the face
                    person_x = max(0, x - (person_width - w) // 2)
                    person_y = max(0, y - (person_height - h) // 2)
                    
                    # High confidence for face detection
                    confidence = 0.95
                    
                    detections.append((person_x, person_y, person_width, person_height, 'face', confidence))
        
        return detections
    
    def detect_people_in_zones(self, frame):
        """Optimized detection function"""
        # Convert to grayscale efficiently
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Lightweight preprocessing
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        face_detections = self.detect_faces_lightweight(gray)
        
        detected_zones = set()
        person_count = 0
        detection_info = []
        
        # Process detections
        for (x, y, w, h, detection_type, confidence) in face_detections:
            if confidence > self.confidence_threshold:
                person_center_x = x + w // 2
                person_center_y = y + h // 2
                
                # Determine zone
                for zone_idx, (zone_x1, zone_y1, zone_x2, zone_y2) in enumerate(self.zones):
                    if (zone_x1 <= person_center_x <= zone_x2 and 
                        zone_y1 <= person_center_y <= zone_y2):
                        
                        # Draw detection box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face'], 2)
                        
                        # Add detection info
                        cv2.putText(frame, f"FACE: {confidence:.2f}", (x, y-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['face'], 1)
                        
                        detected_zones.add(zone_idx)
                        person_count += 1
                        detection_info.append((zone_idx, detection_type, confidence))
                        break
        
        return detected_zones, person_count, detection_info
    
    def display_info(self, frame, detected_zones, person_count, detection_info):
        """Display optimized information overlay"""
        # Simple background
        cv2.rectangle(frame, (5, 5), (300, 100), self.colors['background'], -1)
        cv2.rectangle(frame, (5, 5), (300, 100), self.colors['text'], 1)
        
        if not detected_zones:
            cv2.putText(frame, "No person detected", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        else:
            cv2.putText(frame, f"People detected: {person_count}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            
            zone_texts = [self.zone_names[i] for i in sorted(detected_zones)]
            cv2.putText(frame, f"Zones: {', '.join(zone_texts)}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
            y_offset = 75
            for zone_idx, detection_type, confidence in detection_info:
                zone_text = f"Zone {zone_idx + 1}: {self.zone_names[zone_idx]}"
                cv2.putText(frame, zone_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset += 20
    
    def add_zone_highlights(self, frame, detected_zones):
        """Add lightweight zone highlighting"""
        for zone_idx in detected_zones:
            if zone_idx < len(self.zones):
                x1, y1, x2, y2 = self.zones[zone_idx]
                
                # Simple highlight
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['face'], 2)
    
    def run(self):
        """Optimized main detection loop"""
        print("Starting Lightweight Face Detection...")
        print("Press 'q' to quit")
        print("Press 's' to save frame")
        print("Press 'f' to toggle frame processing")
        
        frame_count = 0
        frame_skip_toggle = False
        last_detection_results = (set(), 0, [])  # Cache last results
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Setup zones on first frame
            if not self.zones:
                frame_height, frame_width = frame.shape[:2]
                self.setup_detection_zones(frame_width, frame_height)
            
            # Toggle frame processing
            if frame_skip_toggle:
                self.process_every_n_frames = 3  # Process every 3rd frame
            else:
                self.process_every_n_frames = 2  # Process every 2nd frame
            
            # Draw zones
            self.draw_detection_zones(frame)
            
            # Detect people (only when needed)
            if frame_count % self.process_every_n_frames == 0:
                detected_zones, person_count, detection_info = self.detect_people_in_zones(frame)
                last_detection_results = (detected_zones, person_count, detection_info)
            else:
                # Use cached results for skipped frames
                detected_zones, person_count, detection_info = last_detection_results
            
            # Highlight zones
            self.add_zone_highlights(frame, detected_zones)
            
            # Display info
            self.display_info(frame, detected_zones, person_count, detection_info)
            
            # Performance info
            cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1] - 120, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            cv2.putText(frame, f"Process: 1/{self.process_every_n_frames}", (frame.shape[1] - 120, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # Show frame
            cv2.imshow('Lightweight Face Detection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"lightweight_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('f'):
                frame_skip_toggle = not frame_skip_toggle
                print(f"Frame processing: every {self.process_every_n_frames} frame(s)")
            
            frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = LightweightFaceDetector()
    detector.run() 