import cv2
import torch
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add current directory to path for YOLOv5 imports
sys.path.append(str(Path(__file__).parent))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

class UnifiedDetector:
    def __init__(self):
        # Initialize camera with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Detection zones for face detection
        self.zones = []
        self.zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        
        # Colors
        self.colors = {
            'zone': (0, 255, 0),
            'detection': (0, 0, 255),
            'text': (255, 255, 255),
            'curve': (255, 255, 0),
            'face': (255, 0, 255),
            'vehicle': (0, 255, 0),
            'background': (0, 0, 0)
        }
        
        # Face detection parameters
        self.confidence_threshold = 0.3
        self.curve_amplitude = 80
        self.process_every_n_frames = 2
        self.frame_count = 0
        
        # Vehicle detection parameters
        self.vehicle_conf_threshold = 0.25
        self.vehicle_iou_threshold = 0.45
        self.img_size = 320
        
        # Load models
        self.load_models()
        
        # Performance tracking
        self.fps = 0
        self.start_time = time.time()
        self.fps_frame_count = 0
        
    def load_models(self):
        """Load both face and vehicle detection models"""
        print("Loading detection models...")
        
        # Load face detection models (lightweight Haar cascades)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Check face models
        if self.face_cascade.empty():
            print("Warning: Primary face cascade not loaded")
        if self.face_cascade2.empty():
            print("Warning: Secondary face cascade not loaded")
        
        # Load vehicle detection model (YOLOv5)
        try:
            weights_path = "runs/train/exp12/weights/best.pt"
            if os.path.exists(weights_path):
                self.device = select_device('')
                self.vehicle_model = attempt_load(weights_path, map_location=self.device)
                self.vehicle_model.eval()
                print(f"Vehicle model loaded successfully on {self.device}")
                self.vehicle_model_loaded = True
            else:
                print(f"Warning: Vehicle model not found at {weights_path}")
                self.vehicle_model_loaded = False
        except Exception as e:
            print(f"Error loading vehicle model: {e}")
            self.vehicle_model_loaded = False
        
        print("Models loaded successfully!")
    
    def setup_detection_zones(self, frame_width, frame_height):
        """Setup detection zones with optimized curved boundaries"""
        zone_width = frame_width // 5
        self.zones = []
        
        for i in range(5):
            x1 = i * zone_width
            x2 = (i + 1) * zone_width
            margin_y = int(frame_height * 0.2)
            y1 = margin_y
            y2 = frame_height - margin_y
            self.zones.append((x1, y1, x2, y2))
    
    def draw_optimized_curves(self, frame):
        """Draw optimized curved boundaries for performance"""
        frame_height, frame_width = frame.shape[:2]
        
        top_curve_points = []
        bottom_curve_points = []
        
        for x in range(0, frame_width, 10):
            curve_offset = int(self.curve_amplitude * np.sin(np.pi * x / frame_width))
            
            top_y = int(frame_height * 0.2) + curve_offset
            bottom_y = int(frame_height * 0.8) - curve_offset
            
            top_curve_points.append([x, top_y])
            bottom_curve_points.append([x, bottom_y])
        
        top_curve_points = np.array(top_curve_points, np.int32)
        bottom_curve_points = np.array(bottom_curve_points, np.int32)
        
        cv2.polylines(frame, [top_curve_points], False, self.colors['curve'], 3)
        cv2.polylines(frame, [bottom_curve_points], False, self.colors['curve'], 3)
    
    def detect_faces_lightweight(self, gray):
        """Detect faces using lightweight Haar cascades"""
        faces = []
        
        # Primary cascade
        faces1 = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Secondary cascade
        faces2 = self.face_cascade2.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Combine detections
        all_faces = list(faces1) + list(faces2)
        
        # Remove duplicates and add confidence
        for (x, y, w, h) in all_faces:
            faces.append([x, y, w, h, 0.8])  # Default confidence for Haar
        
        return faces
    
    def preprocess_frame_for_vehicles(self, frame):
        """Preprocess frame for YOLOv5 vehicle inference"""
        frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float()
        frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)
        return frame_tensor
    
    def detect_vehicles(self, frame):
        """Detect vehicles using YOLOv5"""
        if not self.vehicle_model_loaded:
            return frame, []
        
        # Preprocess frame
        frame_tensor = self.preprocess_frame_for_vehicles(frame)
        frame_tensor = frame_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.vehicle_model(frame_tensor)[0]
        
        # Apply NMS
        predictions = non_max_suppression(predictions, self.vehicle_conf_threshold, self.vehicle_iou_threshold)
        
        detections = []
        frame_with_boxes = frame.copy()
        
        if predictions[0] is not None and len(predictions[0]) > 0:
            # Scale coordinates back to original frame size
            predictions[0][:, :4] = scale_coords(frame_tensor.shape[2:], predictions[0][:, :4], frame.shape).round()
            
            # Process each detection
            for *xyxy, conf, cls in predictions[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(conf)
                class_id = int(cls)
                
                # Draw bounding box (green for vehicles)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), self.colors['vehicle'], 2)
                
                # Draw label
                label = f'Vehicle: {confidence:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame_with_boxes, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), self.colors['vehicle'], -1)
                
                # Draw label text
                cv2.putText(frame_with_boxes, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return frame_with_boxes, detections
    
    def detect_people_in_zones(self, frame):
        """Detect people/faces in different zones"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces_lightweight(gray)
        
        detected_zones = []
        person_count = 0
        
        # Check each zone
        for i, (x1, y1, x2, y2) in enumerate(self.zones):
            zone_faces = []
            
            for (fx, fy, fw, fh, conf) in faces:
                face_center_x = fx + fw // 2
                face_center_y = fy + fh // 2
                
                # Check if face is in this zone
                if x1 <= face_center_x <= x2 and y1 <= face_center_y <= y2:
                    zone_faces.append((fx, fy, fw, fh, conf))
                    person_count += 1
            
            if zone_faces:
                detected_zones.append(i)
                
                # Draw face boxes in zone
                for (fx, fy, fw, fh, conf) in zone_faces:
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), self.colors['face'], 2)
                    cv2.putText(frame, f'Face: {conf:.2f}', (fx, fy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['face'], 1)
        
        return detected_zones, person_count, faces
    
    def display_info(self, frame, detected_zones, person_count, vehicle_count, detection_info):
        """Display comprehensive detection information"""
        frame_height, frame_width = frame.shape[:2]
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "UNIFIED DETECTOR", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # FPS
        cv2.putText(frame, f'FPS: {self.fps:.1f}', (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detection counts
        cv2.putText(frame, f'Faces: {person_count}', (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f'Vehicles: {vehicle_count}', (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Resolution
        cv2.putText(frame, f'Resolution: {frame_width}x{frame_height}', (20, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Active zones
        if detected_zones:
            zone_text = f'Active Zones: {", ".join([self.zone_names[i] for i in detected_zones])}'
            cv2.putText(frame, zone_text, (20, 185), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def add_zone_highlights(self, frame, detected_zones):
        """Highlight active detection zones"""
        for i, (x1, y1, x2, y2) in enumerate(self.zones):
            color = self.colors['zone'] if i in detected_zones else (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Zone label
            label = self.zone_names[i]
            cv2.putText(frame, label, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_frame_count += 1
        if self.fps_frame_count % 30 == 0:
            elapsed_time = time.time() - self.start_time
            self.fps = self.fps_frame_count / elapsed_time
            self.start_time = time.time()
            self.fps_frame_count = 0
    
    def run(self):
        """Main detection loop"""
        print("Unified Face & Vehicle Detection Started!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save screenshot")
        print("- Press 'r' to reset FPS counter")
        print("- Press 'f' to toggle face detection zones")
        print("- Press 'v' to toggle vehicle detection")
        
        show_zones = True
        show_vehicles = True
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Setup zones on first frame
            if not self.zones:
                self.setup_detection_zones(frame.shape[1], frame.shape[0])
            
            # Process face detection every N frames for performance
            if self.frame_count % self.process_every_n_frames == 0:
                detected_zones, person_count, faces = self.detect_people_in_zones(frame)
            else:
                detected_zones, person_count = [], 0
            
            # Process vehicle detection
            if show_vehicles:
                frame, vehicle_detections = self.detect_vehicles(frame)
                vehicle_count = len(vehicle_detections)
            else:
                vehicle_count = 0
            
            # Draw zones and curves
            if show_zones:
                self.draw_optimized_curves(frame)
                self.add_zone_highlights(frame, detected_zones)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Display information
            self.display_info(frame, detected_zones, person_count, vehicle_count, {})
            
            # Show frame
            cv2.imshow('Unified Face & Vehicle Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"unified_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
            elif key == ord('r'):
                self.fps_frame_count = 0
                self.start_time = time.time()
                print("FPS counter reset")
            elif key == ord('f'):
                show_zones = not show_zones
                print(f"Face detection zones: {'ON' if show_zones else 'OFF'}")
            elif key == ord('v'):
                show_vehicles = not show_vehicles
                print(f"Vehicle detection: {'ON' if show_vehicles else 'OFF'}")
            
            self.frame_count += 1
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")

def main():
    detector = UnifiedDetector()
    detector.run()

if __name__ == "__main__":
    main() 