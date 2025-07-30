#!/usr/bin/env python3
"""
Lightweight Voice Detection System
Complete detection with voice output for each action
Optimized for performance and smooth operation
"""

import cv2
import numpy as np
import time
import json
import os
import sys
import threading
from datetime import datetime
import requests
from typing import Optional, Dict, List, Tuple
import queue
import logging

# Voice output setup
try:
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠️  pyttsx3 not available - voice output disabled")

# Gemini API setup
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Gemini API not available - AI analysis disabled")

# YOLOv5 setup (optional)
try:
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv5 not available - vehicle detection disabled")

class VoiceManager:
    """Manages voice output with rate limiting and queue system"""
    
    def __init__(self):
        self.engine = None
        self.voice_queue = queue.Queue()
        self.last_announcement = {}
        self.cooldown = 3.0  # seconds between same announcements
        self.voice_thread = None
        self.running = False
        
        if VOICE_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)  # Speed
                self.engine.setProperty('volume', 0.8)  # Volume
                self.running = True
                self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
                self.voice_thread.start()
                print("✅ Voice system initialized")
            except Exception as e:
                print(f"❌ Voice system failed: {e}")
                self.running = False
    
    def _voice_worker(self):
        """Background thread for voice output"""
        while self.running:
            try:
                message = self.voice_queue.get(timeout=1)
                if self.engine:
                    self.engine.say(message)
                    self.engine.runAndWait()
                self.voice_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice error: {e}")
    
    def announce(self, category: str, message: str):
        """Announce with rate limiting"""
        current_time = time.time()
        if category not in self.last_announcement or \
           current_time - self.last_announcement[category] > self.cooldown:
            self.last_announcement[category] = current_time
            if self.running:
                self.voice_queue.put(message)
                print(f"🔊 Voice: {message}")
    
    def stop(self):
        """Stop voice system"""
        self.running = False
        if self.voice_thread:
            self.voice_thread.join(timeout=1)

class LightweightVoiceDetector:
    """Complete lightweight detection system with voice output"""
    
    def __init__(self):
        # Camera settings (optimized for performance)
        self.camera_index = 0
        self.frame_width = 320
        self.frame_height = 240
        self.fps = 30
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.face_cascade = None
        self.vehicle_model = None
        self.device = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.start_time = time.time()
        self.current_fps = 0
        
        # Voice system
        self.voice = VoiceManager()
        
        # Gemini API
        self.gemini_model = None
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.last_gemini_call = 0
        self.gemini_cooldown = 5.0  # seconds between calls
        
        # Detection tracking
        self.detection_history = {
            'faces': [],
            'vehicles': [],
            'traffic_signs': [],
            'objects': []
        }
        
        # Output settings
        self.json_output_path = "lightweight_detection_results.json"
        self.save_interval = 30  # seconds
        
        # Initialize components
        self._initialize_camera()
        self._initialize_models()
        self._initialize_gemini()
        
        print("✅ Lightweight Voice Detector initialized")
    
    def _initialize_camera(self):
        """Initialize camera with optimized settings"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("❌ Cannot open camera")
            sys.exit(1)
        
        # Set camera properties for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"✅ Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
    
    def _initialize_models(self):
        """Initialize detection models"""
        # Face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                print("❌ Failed to load face cascade")
            else:
                print("✅ Face detection model loaded")
        except Exception as e:
            print(f"❌ Face detection error: {e}")
        
        # Vehicle detection (YOLOv5)
        if YOLO_AVAILABLE:
            try:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.vehicle_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.vehicle_model.to(self.device)
                self.vehicle_model.eval()
                print(f"✅ Vehicle detection model loaded on {self.device}")
            except Exception as e:
                print(f"❌ Vehicle detection error: {e}")
    
    def _initialize_gemini(self):
        """Initialize Gemini API"""
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
                print("✅ Gemini API initialized")
            except Exception as e:
                print(f"❌ Gemini API error: {e}")
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1.0:
            self.current_fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        face_detections = []
        for (x, y, w, h) in faces:
            confidence = 0.8  # Cascade doesn't provide confidence
            if confidence > self.confidence_threshold:
                face_detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'class': 'face'
                })
        
        return face_detections
    
    def detect_vehicles(self, frame):
        """Detect vehicles using YOLOv5"""
        if not YOLO_AVAILABLE or self.vehicle_model is None:
            return []
        
        try:
            # Vehicle classes in COCO dataset
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
            
            # Run inference
            results = self.vehicle_model(frame)
            
            vehicle_detections = []
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                
                if conf > self.confidence_threshold:
                    class_name = results.names[int(cls)]
                    if class_name in vehicle_classes:
                        vehicle_detections.append({
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'confidence': float(conf),
                            'class': class_name
                        })
            
            return vehicle_detections
        except Exception as e:
            print(f"Vehicle detection error: {e}")
            return []
    
    def detect_traffic_signs(self, frame):
        """Simple traffic sign detection using color and shape"""
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red color range for stop signs
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Blue color range for other signs
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sign_detections = []
        
        # Process red contours (potential stop signs)
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if it's roughly octagonal (stop sign shape)
                if 0.8 < aspect_ratio < 1.2:
                    sign_detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': 0.7,
                        'class': 'stop_sign'
                    })
        
        # Process blue contours (potential other signs)
        for contour in blue_contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                sign_detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': 0.6,
                    'class': 'traffic_sign'
                })
        
        return sign_detections
    
    def analyze_with_gemini(self, frame):
        """Analyze frame with Gemini API"""
        if not GEMINI_AVAILABLE or self.gemini_model is None:
            return None
        
        current_time = time.time()
        if current_time - self.last_gemini_call < self.gemini_cooldown:
            return None
        
        try:
            # Resize frame for API efficiency
            small_frame = cv2.resize(frame, (320, 240))
            
            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
            
            # Prepare prompt
            prompt = """
            Analyze this image and provide a brief description of what you see.
            Focus on:
            1. Objects and their locations
            2. Any safety concerns
            3. Traffic signs or signals
            4. People or vehicles
            Keep response under 50 words.
            """
            
            # Call Gemini API
            response = self.gemini_model.generate_content([prompt, pil_image])
            self.last_gemini_call = current_time
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
    
    def draw_detections(self, frame, detections, color):
        """Draw detection boxes on frame"""
        for det in detections:
            x, y, w, h = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def save_results(self):
        """Save detection results to JSON"""
        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'fps': convert_numpy(self.current_fps),
                'detections': self.detection_history,
                'total_frames': convert_numpy(self.frame_count)
            }
            
            with open(self.json_output_path, 'w') as f:
                json.dump(results, f, indent=2, default=convert_numpy)
            
            print(f"✅ Results saved to {self.json_output_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def process_frame(self, frame):
        """Process single frame with all detections"""
        # Detect faces
        faces = self.detect_faces(frame)
        if faces:
            self.voice.announce('faces', f"Detected {len(faces)} face{'s' if len(faces) > 1 else ''}")
        
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        if vehicles:
            vehicle_types = [v['class'] for v in vehicles]
            self.voice.announce('vehicles', f"Detected {len(vehicles)} vehicle{'s' if len(vehicles) > 1 else ''}: {', '.join(vehicle_types)}")
        
        # Detect traffic signs
        signs = self.detect_traffic_signs(frame)
        if signs:
            sign_types = [s['class'] for s in signs]
            self.voice.announce('signs', f"Detected {len(signs)} traffic sign{'s' if len(signs) > 1 else ''}: {', '.join(sign_types)}")
        
        # Update detection history
        self.detection_history['faces'] = faces
        self.detection_history['vehicles'] = vehicles
        self.detection_history['traffic_signs'] = signs
        
        # Analyze with Gemini (occasionally)
        gemini_analysis = self.analyze_with_gemini(frame)
        if gemini_analysis:
            self.voice.announce('analysis', f"AI Analysis: {gemini_analysis}")
        
        # Draw detections
        self.draw_detections(frame, faces, (0, 255, 0))      # Green for faces
        self.draw_detections(frame, vehicles, (255, 0, 0))   # Blue for vehicles
        self.draw_detections(frame, signs, (0, 0, 255))      # Red for signs
        
        return frame
    
    def run(self):
        """Main detection loop"""
        print("🚀 Starting Lightweight Voice Detection System")
        print("Press 'q' to quit, 's' to save results")
        
        last_save_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Add FPS display
                cv2.putText(processed_frame, f"FPS: {self.current_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add status display
                status_text = f"Faces: {len(self.detection_history['faces'])} | "
                status_text += f"Vehicles: {len(self.detection_history['vehicles'])} | "
                status_text += f"Signs: {len(self.detection_history['traffic_signs'])}"
                
                cv2.putText(processed_frame, status_text, 
                           (10, processed_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Lightweight Voice Detector', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_results()
                
                # Auto-save every save_interval seconds
                current_time = time.time()
                if current_time - last_save_time > self.save_interval:
                    self.save_results()
                    last_save_time = current_time
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        # Stop voice system
        self.voice.stop()
        
        # Release camera
        if hasattr(self, 'cap'):
            self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Save final results
        self.save_results()
        
        print("✅ Cleanup complete")

def main():
    """Main function"""
    print("🎯 Lightweight Voice Detection System")
    print("=" * 50)
    
    # Check dependencies
    print("📋 Checking dependencies...")
    if not VOICE_AVAILABLE:
        print("⚠️  Install pyttsx3 for voice output: pip install pyttsx3")
    if not GEMINI_AVAILABLE:
        print("⚠️  Install Gemini API for AI analysis: pip install google-generativeai")
    if not YOLO_AVAILABLE:
        print("⚠️  Install PyTorch for vehicle detection: pip install torch torchvision")
    
    # Create and run detector
    detector = LightweightVoiceDetector()
    detector.run()

if __name__ == "__main__":
    main() 