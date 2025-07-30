import cv2
import numpy as np
import time
import sys
import os
import json
import threading
import requests
import base64
import subprocess
import queue
from pathlib import Path
from datetime import datetime

# Try to import torch and YOLOv5 dependencies (optional)
try:
    import torch
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device
    YOLO_AVAILABLE = True
    print("✅ YOLOv5 dependencies available")
except ImportError as e:
    print(f"⚠️ YOLOv5 dependencies not available: {e}")
    YOLO_AVAILABLE = False

# Voice output setup
try:
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠️  pyttsx3 not available - voice output disabled")

# Add current directory to path for YOLOv5 imports
sys.path.append(str(Path(__file__).parent))

# Add traffic sign detection path
traffic_sign_path = Path("Traffic-Sign-Detection/Traffic-Sign-Detection")
sys.path.append(str(traffic_sign_path))

# Import traffic sign detection modules
try:
    from classification import training, getLabel
    from improved_classification import improved_training, improved_getLabel
    from main import localization
    TRAFFIC_SIGN_AVAILABLE = True
    print("✅ Traffic sign detection available")
except ImportError as e:
    print(f"⚠️ Traffic sign detection not available: {e}")
    TRAFFIC_SIGN_AVAILABLE = False

class VoiceManager:
    """Manages voice output with rate limiting and queue system"""
    
    def __init__(self):
        self.engine = None
        self.voice_queue = queue.Queue()
        self.last_announcement = {}
        self.cooldown = 3.0  # seconds between same announcements (increased to prevent overlap)
        self.voice_thread = None
        self.running = False
        
        if VOICE_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 120)  # Slower speed for clarity
                self.engine.setProperty('volume', 1.0)  # Maximum volume
                self.running = True
                self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
                self.voice_thread.start()
                print("✅ Voice system initialized")
            except Exception as e:
                print(f"❌ Voice system failed: {e}")
                self.running = False
    
    def _voice_worker(self):
        """Background thread for voice output with proper delays"""
        while self.running:
            try:
                message = self.voice_queue.get(timeout=1)
                if self.engine and message and self.running:
                    print(f"🔊 Speaking: {message}")
                    try:
                        self.engine.say(message)
                        self.engine.runAndWait()
                        print(f"✅ Voice completed: {message}")
                    except Exception as voice_error:
                        print(f"❌ Voice playback failed: {voice_error}")
                        # Try to reinitialize engine if it failed
                        try:
                            self.engine = pyttsx3.init()
                            self.engine.setProperty('rate', 120)
                            self.engine.setProperty('volume', 1.0)
                            self.engine.say(message)
                            self.engine.runAndWait()
                            print(f"✅ Voice recovered and completed: {message}")
                        except Exception as recovery_error:
                            print(f"❌ Voice recovery failed: {recovery_error}")
                    
                    # Wait between messages to prevent overlap
                    time.sleep(1.5)
                self.voice_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice worker error: {e}")
                time.sleep(0.5)
    
    def announce(self, category: str, message: str):
        """Announce with rate limiting and proper threading"""
        if not self.running or not self.engine:
            print(f"❌ Voice system not running. Message: {message}")
            return
            
        current_time = time.time()
        if category not in self.last_announcement or \
           current_time - self.last_announcement[category] > self.cooldown:
            self.last_announcement[category] = current_time
            
            # Force voice output regardless of queue status
            try:
                self.voice_queue.put(message, timeout=1)
                print(f"🔊 Voice queued: {message}")
            except queue.Full:
                print(f"⚠️ Voice queue full, speaking immediately: {message}")
                # If queue is full, speak immediately in current thread
                try:
                    self.engine.say(message)
                    self.engine.runAndWait()
                    print(f"✅ Voice immediate: {message}")
                except Exception as e:
                    print(f"❌ Immediate voice failed: {e}")
        else:
            remaining_cooldown = self.cooldown - (current_time - self.last_announcement[category])
            print(f"🔇 Voice cooldown active for {category}. {remaining_cooldown:.1f}s remaining")
    
    def stop(self):
        """Stop voice system"""
        self.running = False
        if self.voice_thread:
            self.voice_thread.join(timeout=1)

class EnhancedUnifiedDetector:
    def __init__(self):
        # Initialize HTTPS camera with authentication
        self.cap = None
        self.camera_url = "http://192.168.1.100:8080/shot.jpg"
        self.camera_username = "admin"
        self.camera_password = "admin123"
        
        # Detection models
        self.face_cascade = None
        self.vehicle_model = None
        self.traffic_sign_model = None
        
        # Detection zones
        self.zones = []
        self.zone_colors = ['red', 'green', 'blue', 'yellow', 'magenta']
        
        # Performance tracking
        self.frame_count = 0
        self.fps_frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0
        self.target_fps = 30.0
        self.process_every_n_frames = 2
        
        # Detection counters
        self.face_count = 0
        self.vehicle_count = 0
        self.traffic_sign_count = 0
        
        # Configuration flags
        self.lightweight_mode = True
        self.traffic_sign_enabled = False
        self.json_output_enabled = True
        self.voice_enabled = True  # Enable voice by default
        
        # Voice system
        self.voice = VoiceManager()
        
        # Vehicle detection parameters
        self.vehicle_conf_threshold = 0.25
        self.vehicle_iou_threshold = 0.45
        self.img_size = 320
        
        # Face detection parameters
        self.confidence_threshold = 0.3
        self.curve_amplitude = 80
        
        # Colors for visualization
        self.colors = {
            'zone': (0, 255, 0),
            'detection': (0, 0, 255),
            'text': (255, 255, 255),
            'curve': (255, 255, 0),
            'face': (255, 0, 255),
            'vehicle': (0, 255, 0),
            'traffic_sign': (0, 255, 255),  # Yellow for traffic signs
            'background': (0, 0, 0)
        }
        
        # Zone names
        self.zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        
        # Performance optimization settings
        self.detection_interval = 3   # Process detection every 3 frames for 30 FPS
        self.frame_times = []  # Track frame processing times
        self.max_latency_ms = 33  # Target max 33ms latency (30 FPS)
        
        # Voice feedback settings
        self.last_voice_announcement = 0
        self.voice_cooldown = 15.0  # 15 seconds between voice announcements
        
        # JSON output settings
        self.json_output_path = "detection_results.json"
        self.json_update_interval = 1.0  # Update every 1 second
        self.last_json_update = time.time()
        
        # Gemini API configuration
        self.gemini_enabled = True
        self.gemini_api_key = "AIzaSyDf-FK-jZfcYsQBVJh4GSwvn5Uokhb1Wlw"  # Direct API key
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
        self.gemini_cooldown = 3.0  # 3 seconds between requests
        self.max_requests_per_minute = 15  # Rate limit
        self.last_gemini_query = 0
        self.gemini_request_count = 0
        self.gemini_request_times = []  # Track request times for rate limiting
        
        # Voice engine
        self.voice_engine = None
        
        # Load models
        self.load_models()
        
        # Initialize voice if enabled
        if self.voice_enabled:
        self.initialize_voice_engine()
        
        # Enable Gemini with direct API key
        if self.gemini_api_key:
            self.gemini_enabled = True
            print("✅ Gemini API configured with advanced AI analysis")
        else:
            print("⚠️ Gemini API key not found.")
        
        # Initialize camera
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize camera with optimized settings"""
        try:
            # Try IP camera first
            username = "admin"
            password = "admin"
            ip_address = "192.168.0.108"
            port = "8080"
            
            url = f"http://{username}:{password}@{ip_address}:{port}/video"
            print(f"🔗 Connecting to IP camera: {url}")
            
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Set optimized parameters
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            
            if not self.cap.isOpened():
                print("❌ IP camera failed, trying webcam...")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise RuntimeError("No camera available")
                else:
                    print("✅ Connected to webcam")
            else:
                print("✅ Connected to IP camera")
                
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            # Fallback to webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("No camera available")
            print("✅ Connected to webcam as fallback")
        
    def load_models(self):
        """Load all detection models"""
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
        if YOLO_AVAILABLE:
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
        else:
            print("⚠️ YOLOv5 dependencies not available, vehicle detection disabled.")
            self.vehicle_model_loaded = False
        
        # Load traffic sign detection model
        if self.traffic_sign_enabled:
            try:
                self.traffic_sign_model = improved_training() if 'improved_training' in globals() else training()
                print("Traffic sign model loaded successfully")
            except Exception as e:
                print(f"Error loading traffic sign model: {e}")
                self.traffic_sign_enabled = False
        
        print("Models loaded successfully!")
    
    def generate_detection_json(self, detected_zones, person_count, vehicle_count, traffic_sign_count, faces, vehicles, traffic_signs):
        """Generate JSON output with detection results"""
        current_time = datetime.now().isoformat()
        
        # Zone mapping
        zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        
        # Process face detections with zone information
        face_detections = []
        for (x, y, w, h, conf) in faces:
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Determine zone
            zone_index = self.get_zone_for_point(face_center_x, face_center_y)
            zone_name = zone_names[zone_index] if zone_index < len(zone_names) else "Unknown"
            
            face_detections.append({
                "type": "face",
                "confidence": float(conf),
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "center": {"x": int(face_center_x), "y": int(face_center_y)},
                "zone": zone_name,
                "zone_index": zone_index
            })
        
        # Process vehicle detections
        vehicle_detections = []
        if YOLO_AVAILABLE:
        for vehicle in vehicles:
            if len(vehicle) >= 6:
                x1, y1, x2, y2, conf, class_id = vehicle
                vehicle_detections.append({
                    "type": "vehicle",
                    "confidence": float(conf),
                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    "class_id": int(class_id)
                })
        else:
            print("⚠️ YOLOv5 dependencies not available, vehicle detection disabled.")
            vehicle_detections = []
        
        # Process traffic sign detections
        traffic_sign_detections = []
        if TRAFFIC_SIGN_AVAILABLE:
        for sign in traffic_signs:
            if isinstance(sign, list) and len(sign) > 0:
                traffic_sign_detections.append({
                    "type": "traffic_sign",
                    "sign_type": str(sign[0]) if len(sign) > 0 else "unknown"
                })
        else:
            print("⚠️ Traffic sign detection not available, traffic sign detection disabled.")
            traffic_sign_detections = []
        
        # Create JSON structure
        detection_data = {
            "timestamp": current_time,
            "fps": float(self.current_fps),
            "frame_count": self.frame_count,
            "summary": {
                "total_faces": person_count,
                "total_vehicles": vehicle_count,
                "total_traffic_signs": traffic_sign_count,
                "active_zones": [zone_names[i] for i in detected_zones]
            },
            "detections": {
                "faces": face_detections,
                "vehicles": vehicle_detections,
                "traffic_signs": traffic_sign_detections
            },
            "performance": {
                "target_fps": self.target_fps,
                "current_fps": float(self.current_fps),
                "lightweight_mode": self.lightweight_mode
            }
        }
        
        return detection_data
    
    def get_zone_for_point(self, x, y):
        """Get zone index for a given point"""
        if not self.zones:
            return 0
        
        for i, (x1, y1, x2, y2) in enumerate(self.zones):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return 2  # Default to center zone
    
    def save_json_output(self, detection_data):
        """Save detection data to JSON file"""
        try:
            with open(self.json_output_path, 'w') as f:
                json.dump(detection_data, f, indent=2)
        except Exception as e:
            print(f"Error saving JSON: {e}")
    
    def encode_image_for_gemini(self, frame):
        """Encode image to base64 for Gemini API"""
        try:
            # Use smaller image size to reduce API payload size and improve responsiveness
            resized_frame = cv2.resize(frame, (320, 240))  # Match camera resolution
            _, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Lower quality for speed
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            return encoded_image
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def initialize_voice_engine(self):
        """Initialize text-to-speech engine"""
        # Voice is now handled by VoiceManager class
        if self.voice.running:
            print("✅ Voice feedback initialized (pyttsx3)")
            else:
            print("❌ Voice initialization failed")
            self.voice_enabled = False
    
    def speak_text(self, text):
        """Speak text using lightweight voice system"""
        if self.voice_enabled and self.voice.running:
            self.voice.announce('general', text)
    
    def test_voice(self):
        """Test voice feedback functionality"""
        if self.voice_enabled and self.voice.running:
            test_messages = [
                "Voice feedback is working correctly",
                "Person detected to the left of camera view",
                "Person detected in the center of camera view",
                "Person detected to the right of camera view",
                "Multiple people detected in camera view"
            ]
            for message in test_messages:
                self.voice.announce('test', message)
                time.sleep(2)  # Longer pause between messages
            print("🔊 Voice test completed")
        else:
            print("❌ Voice feedback is disabled")
    
    def speak_detection_summary(self, detection_data):
        """Generate and speak detection summary"""
        if not self.voice_enabled or not self.voice.running:
            return
        
        try:
            # Create voice announcements for each detection type
            if detection_data['summary']['total_faces'] > 0:
                self.voice.announce('faces', f"Detected {detection_data['summary']['total_faces']} face{'s' if detection_data['summary']['total_faces'] > 1 else ''}")
            
            if detection_data['summary']['total_vehicles'] > 0:
                self.voice.announce('vehicles', f"Detected {detection_data['summary']['total_vehicles']} vehicle{'s' if detection_data['summary']['total_vehicles'] > 1 else ''}")
            
            if detection_data['summary']['total_traffic_signs'] > 0:
                self.voice.announce('signs', f"Detected {detection_data['summary']['total_traffic_signs']} traffic sign{'s' if detection_data['summary']['total_traffic_signs'] > 1 else ''}")
            
            # System status announcement
            if detection_data['fps'] < 20:
                self.voice.announce('status', f"System performance: {detection_data['fps']:.1f} FPS")
            
        except Exception as e:
            print(f"Voice summary error: {e}")
    
    def generate_intelligent_voice_feedback(self, detection_type, count, details=None, frame=None):
        """Generate context-aware voice feedback using Gemini AI with local fallback"""
        if not self.voice_enabled or not self.voice.running:
            return
        
        try:
            # Create context-aware prompts for different detection types
            if detection_type == 'faces':
                prompt = f"""
                Context: A computer vision system has detected {count} face(s) in a real-time camera feed.
                
                Generate a natural, context-aware voice announcement that:
                1. Acknowledges the detection
                2. Provides situational awareness
                3. Suggests appropriate actions if needed
                4. Uses conversational, human-like language
                
                Examples of good responses:
                - "A person has been detected in the camera view"
                - "Multiple people are present in the area"
                - "Someone is approaching the camera"
                
                Keep the response under 2 sentences and make it sound natural.
                """
            
            elif detection_type == 'vehicles':
                vehicle_info = details if details else "vehicle(s)"
                prompt = f"""
                Context: A computer vision system has detected {count} {vehicle_info} in a real-time camera feed.
                
                Generate a natural, context-aware voice announcement that:
                1. Acknowledges the vehicle detection
                2. Provides situational awareness about traffic
                3. Suggests safety actions if needed
                4. Uses conversational, human-like language
                
                Examples of good responses:
                - "A car is approaching from the left, please step back"
                - "Multiple vehicles detected in the traffic area"
                - "Vehicle movement detected, maintain safe distance"
                
                Keep the response under 2 sentences and make it sound natural.
                """
            
            elif detection_type == 'traffic_signs':
                sign_info = details if details else "traffic sign(s)"
                prompt = f"""
                Context: A computer vision system has detected {count} {sign_info} in a real-time camera feed.
                
                Generate a natural, context-aware voice announcement that:
                1. Acknowledges the traffic sign detection
                2. Provides safety information
                3. Suggests appropriate actions
                4. Uses conversational, human-like language
                
                Examples of good responses:
                - "Stop sign detected, please halt movement"
                - "Traffic signal identified, proceed with caution"
                - "Warning sign visible, maintain alertness"
                
                Keep the response under 2 sentences and make it sound natural.
                """
            
            else:
                return
            
            # Try Gemini first if available
            if self.gemini_enabled:
                response = self.query_gemini_vision_text_only(prompt)
            
            if response and "Error" not in response and "Rate limit" not in response:
                    # Clean up the response
                    response = response.strip().replace('"', '').replace("'", "")
                    if response.endswith('.'):
                        response = response[:-1]
                    
                    # Announce the intelligent response
                    self.voice.announce(f'intelligent_{detection_type}', response)
                    print(f"🤖 AI Voice: {response}")
                    return
            
            # Fallback to local intelligent responses
            self.generate_local_intelligent_feedback(detection_type, count, details, frame)
            
        except Exception as e:
            print(f"Intelligent voice feedback error: {e}")
            # Fallback to local responses
            self.generate_local_intelligent_feedback(detection_type, count, details, frame)
    
    def generate_local_intelligent_feedback(self, detection_type, count, details=None, frame=None):
        """Generate local intelligent feedback with position detection when Gemini fails"""
        if not self.voice_enabled or not self.voice.running:
            return
        
        try:
            if detection_type == 'faces':
                # Get face positions for spatial awareness
                face_positions = self.get_face_positions(frame)
                if face_positions:
                    position_info = self.analyze_face_positions(face_positions, frame)
                    self.voice.announce('local_faces', position_info)
                    print(f"📍 Local Voice: {position_info}")
                else:
                    # Basic fallback
                    self.voice.announce('local_faces', f"ONE PERSON DETECTED")
                    print(f"📍 Local Voice: ONE PERSON DETECTED")
            
            elif detection_type == 'vehicles':
                # Get vehicle positions for spatial awareness
                vehicle_positions = self.get_vehicle_positions(frame)
                if vehicle_positions:
                    position_info = self.analyze_vehicle_positions(vehicle_positions, frame)
                    self.voice.announce('local_vehicles', position_info)
                    print(f"📍 Local Voice: {position_info}")
                else:
                    # Basic fallback
                    vehicle_types = details if details else "vehicle(s)"
                    self.voice.announce('local_vehicles', f"Vehicle detected: {vehicle_types}")
                    print(f"📍 Local Voice: Vehicle detected: {vehicle_types}")
            
            elif detection_type == 'traffic_signs':
                # Get traffic sign positions for spatial awareness
                sign_positions = self.get_traffic_sign_positions(frame)
                if sign_positions:
                    position_info = self.analyze_traffic_sign_positions(sign_positions, frame)
                    self.voice.announce('local_signs', position_info)
                    print(f"📍 Local Voice: {position_info}")
                else:
                    # Basic fallback
                    sign_types = details if details else "traffic sign(s)"
                    self.voice.announce('local_signs', f"Traffic sign detected: {sign_types}")
                    print(f"📍 Local Voice: Traffic sign detected: {sign_types}")
            
        except Exception as e:
            print(f"Local intelligent feedback error: {e}")
            # Final fallback to basic announcement
            if detection_type == 'faces':
                self.voice.announce(detection_type, f"{count} PERSON DETECTED")
            else:
                self.voice.announce(detection_type, f"Detected {count} {detection_type}")
    
    def get_face_positions(self, frame):
        """Get positions of detected faces"""
        if frame is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            positions = []
            for (x, y, w, h) in faces:
                center_x = x + w // 2
                center_y = y + h // 2
                positions.append({
                    'x': center_x,
                    'y': center_y,
                    'width': w,
                    'height': h,
                    'area': w * h
                })
            
            return positions
        except Exception as e:
            print(f"Error getting face positions: {e}")
            return []
    
    def get_vehicle_positions(self, frame):
        """Get positions of detected vehicles"""
        if frame is None or not YOLO_AVAILABLE:
            return []
        
        try:
            # This would need to be implemented based on your vehicle detection method
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            print(f"Error getting vehicle positions: {e}")
            return []
    
    def get_traffic_sign_positions(self, frame):
        """Get positions of detected traffic signs"""
        if frame is None:
            return []
        
        try:
            # This would need to be implemented based on your traffic sign detection method
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            print(f"Error getting traffic sign positions: {e}")
            return []
    
    def analyze_face_positions(self, face_positions, frame):
        """Analyze face positions and generate intelligent feedback"""
        if not face_positions:
            return "Person detected in camera view"
        
        try:
            frame_width = frame.shape[1] if frame is not None else 640
            frame_height = frame.shape[0] if frame is not None else 480
            
            # Analyze the largest face (closest to camera)
            largest_face = max(face_positions, key=lambda x: x['area'])
            center_x = largest_face['x']
            
            # Determine position relative to frame
            left_threshold = frame_width * 0.4
            right_threshold = frame_width * 0.6
            
            if center_x < left_threshold:
                position = "slightly left"
            elif center_x > right_threshold:
                position = "slightly right"
            else:
                position = "center"
            
            # Generate context-aware response
            if len(face_positions) == 1:
                if position == "center":
                    return f"ONE PERSON DETECTED IN CENTER"
                else:
                    return f"ONE PERSON DETECTED AT {position.upper()}"
            else:
                if position == "center":
                    return f"MULTIPLE PEOPLE DETECTED, MAIN PERSON IN CENTER"
                else:
                    return f"MULTIPLE PEOPLE DETECTED, MAIN PERSON AT {position.upper()}"
                    
        except Exception as e:
            print(f"Error analyzing face positions: {e}")
            return "Person detected in camera view"
    
    def analyze_vehicle_positions(self, vehicle_positions, frame):
        """Analyze vehicle positions and generate intelligent feedback"""
        if not vehicle_positions:
            return "Vehicle detected in camera view"
        
        try:
            frame_width = frame.shape[1] if frame is not None else 640
            
            # Analyze the largest vehicle (closest to camera)
            largest_vehicle = max(vehicle_positions, key=lambda x: x.get('area', 0))
            center_x = largest_vehicle.get('x', frame_width // 2)
            
            # Determine position relative to frame
            left_threshold = frame_width * 0.4
            right_threshold = frame_width * 0.6
            
            if center_x < left_threshold:
                position = "left side"
            elif center_x > right_threshold:
                position = "right side"
            else:
                position = "center"
            
            return f"Vehicle detected on the {position} of the camera view"
            
        except Exception as e:
            print(f"Error analyzing vehicle positions: {e}")
            return "Vehicle detected in camera view"
    
    def analyze_traffic_sign_positions(self, sign_positions, frame):
        """Analyze traffic sign positions and generate intelligent feedback"""
        if not sign_positions:
            return "Traffic sign detected in camera view"
        
        try:
            frame_width = frame.shape[1] if frame is not None else 640
            
            # Analyze the largest sign (closest to camera)
            largest_sign = max(sign_positions, key=lambda x: x.get('area', 0))
            center_x = largest_sign.get('x', frame_width // 2)
            
            # Determine position relative to frame
            left_threshold = frame_width * 0.4
            right_threshold = frame_width * 0.6
            
            if center_x < left_threshold:
                position = "left side"
            elif center_x > right_threshold:
                position = "right side"
            else:
                position = "center"
            
            return f"Traffic sign detected on the {position} of the camera view"
            
        except Exception as e:
            print(f"Error analyzing traffic sign positions: {e}")
            return "Traffic sign detected in camera view"
    
    def query_gemini_vision_text_only(self, prompt):
        """Query Gemini API for text-only responses (no image)"""
        if not self.gemini_enabled:
            return "Gemini API not configured."
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_gemini_query < self.gemini_cooldown:
            remaining_time = self.gemini_cooldown - (current_time - self.last_gemini_query)
            return f"Please wait {remaining_time:.1f} seconds before making another query."
        
        # Check rate limit
        if self.gemini_request_count >= self.max_requests_per_minute:
            return "Rate limit reached. Please wait 1 minute before making more requests."
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [
                        {"text": prompt}
                    ]
                }]
            }
            
            # Use text-only model for faster response
            text_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            url = f"{text_url}?key={self.gemini_api_key}"
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    self.last_gemini_query = current_time
                    self.gemini_request_count += 1
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "No response from Gemini."
            elif response.status_code == 429:
                return "Rate limit exceeded. Please wait longer before trying again."
            else:
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            return f"Error querying Gemini: {str(e)}"
    
    def reset_gemini_rate_limit(self):
        """Reset Gemini rate limit counters"""
        self.gemini_request_count = 0
        self.last_gemini_query = 0
        print("✅ Gemini rate limits reset")
    
    def get_gemini_status(self):
        """Get current Gemini API status"""
        current_time = time.time()
        time_since_last = current_time - self.last_gemini_query
        remaining_cooldown = max(0, self.gemini_cooldown - time_since_last)
        
        return {
            "enabled": self.gemini_enabled,
            "requests_this_minute": self.gemini_request_count,
            "max_requests_per_minute": self.max_requests_per_minute,
            "cooldown_remaining": remaining_cooldown,
            "can_make_request": remaining_cooldown <= 0 and self.gemini_request_count < self.max_requests_per_minute
        }
    
    def save_gemini_response(self, question, response, frame_count):
        """Save Gemini response to JSON file"""
        try:
            gemini_data = {
                "timestamp": datetime.now().isoformat(),
                "frame_count": frame_count,
                "question": question,
                "response": response
            }
            
            with open("gemini_responses.json", 'w') as f:
                json.dump(gemini_data, f, indent=2)
            print(f"✅ Gemini response saved to gemini_responses.json")
        except Exception as e:
            print(f"Error saving Gemini response: {e}")
    
    def query_gemini_vision(self, frame, question="What can you see in this image?"):
        """Query Gemini Vision API with current frame - optimized for performance"""
        if not self.gemini_enabled:
            return "Gemini API not configured. Set GEMINI_API_KEY environment variable."
        
        current_time = time.time()
        
        # Clean up old request times (older than 1 minute)
        self.gemini_request_times = [t for t in self.gemini_request_times if current_time - t < 60]
        
        # Check rate limit
        if len(self.gemini_request_times) >= self.max_requests_per_minute:
            return "Rate limit reached. Please wait before making more requests."
        
        # Check cooldown period
        if current_time - self.last_gemini_query < self.gemini_cooldown:
            remaining_time = self.gemini_cooldown - (current_time - self.last_gemini_query)
            return f"Please wait {remaining_time:.1f} seconds before making another query."
        
        try:
            encoded_image = self.encode_image_for_gemini(frame)
            if not encoded_image:
                return "Failed to encode image."
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [
                        {"text": question},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded_image
                            }
                        }
                    ]
                }]
            }
            
            url = f"{self.gemini_url}?key={self.gemini_api_key}"
            
            # Use shorter timeout to prevent blocking
            response = requests.post(url, headers=headers, json=data, timeout=8)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    self.last_gemini_query = current_time
                    self.gemini_request_times.append(current_time)
                    
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "No response from Gemini."
            elif response.status_code == 429:
                return "Rate limit exceeded. Please wait longer before trying again."
            else:
                return f"API Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Request timeout - Gemini API is slow to respond."
        except Exception as e:
            return f"Error querying Gemini: {str(e)}"
    
    def force_camera_refresh(self):
        """Force camera refresh to fix frame capture issues"""
        try:
            # Release and reinitialize camera
            self.cap.release()
            time.sleep(0.1)  # Brief pause
            
            # Reinitialize with optimized settings
            url = f"http://{self.ip_camera_url}"
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Reapply all settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced width
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # Reduced height
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            print("✅ Camera refreshed successfully")
            return True
        except Exception as e:
            print(f"❌ Camera refresh failed: {e}")
            return False
    
    def optimize_for_high_fps(self):
        """Optimize camera settings for high FPS (30+ FPS)"""
        if self.cap.isOpened():
            # Set minimal buffer size for low latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Use hardware acceleration if available
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            
            # Set optimal codec for high performance
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Disable autofocus and auto-exposure for faster processing
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            # Set high FPS target
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            
            print("✅ High FPS optimization applied (target: 30+ FPS)")
    
    def test_camera_connection(self):
        """Test camera connection and return status"""
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                print(f"✅ Camera test successful - Frame size: {frame.shape}")
                return True
            else:
                print("❌ Camera test failed - Cannot read frame")
                return False
        else:
            print("❌ Camera test failed - Camera not opened")
            return False
    
    def reconnect_camera(self):
        """Reconnect to HTTPS camera"""
        print(f"🔄 Attempting to reconnect to HTTPS camera at {self.ip_camera_url}")
        self.cap.release()
        self.cap = cv2.VideoCapture(self.ip_camera_url)
        if self.cap.isOpened():
            print("✅ Successfully reconnected to HTTPS camera")
            return True
        else:
            print("❌ Failed to reconnect to HTTPS camera")
            return False
    
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
        
        # Use smaller detection parameters for faster processing
        if self.lightweight_mode:
            scale_factor = 1.2  # Faster but less accurate
            min_neighbors = 2   # Lower threshold for speed
            min_size = (15, 15) # Smaller minimum size for 320x240
        else:
            scale_factor = 1.1  # Standard accuracy
            min_neighbors = 4   # Standard threshold
            min_size = (20, 20) # Standard minimum size for 320x240
        
        # Primary cascade
        faces1 = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        
        # Secondary cascade (only if not in lightweight mode)
        if not self.lightweight_mode:
            faces2 = self.face_cascade2.detectMultiScale(
                gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
            )
            # Combine detections
            all_faces = list(faces1) + list(faces2)
        else:
            all_faces = list(faces1)
        
        # Remove duplicates and add confidence
        for (x, y, w, h) in all_faces:
            faces.append([x, y, w, h, 0.8])  # Default confidence for Haar
        
        return faces
    
    def preprocess_frame_for_vehicles(self, frame):
        """Preprocess frame for YOLOv5 vehicle inference"""
        if not YOLO_AVAILABLE:
            return None
        
        # Use smaller input size for faster processing in lightweight mode
        if self.lightweight_mode:
            img_size = 256  # Smaller size for faster processing (320x240 frames)
        else:
            img_size = 320  # Standard size for 320x240 frames
            
        frame_resized = cv2.resize(frame, (img_size, img_size))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float()
        frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)
        return frame_tensor
    
    def detect_vehicles(self, frame):
        """Detect vehicles using YOLOv5"""
        if not YOLO_AVAILABLE:
            print("⚠️ YOLOv5 not available, vehicle detection disabled")
            return frame, []
        
        if not self.vehicle_model_loaded:
            return frame, []
        
        try:
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
            
        except Exception as e:
            print(f"Vehicle detection error: {e}")
            return frame, []
    
    def detect_traffic_signs(self, frame):
        """Detect traffic signs"""
        if not self.traffic_sign_enabled or self.traffic_sign_model is None:
            return frame, []
        
        try:
            # Use the localization function from traffic sign detection
            detected_signs = localization(
                frame, 
                min_size_components=300, 
                similitary_contour_with_circle=0.65, 
                model=self.traffic_sign_model, 
                count=0, 
                current_sign_type=""
            )
            
            # Draw traffic sign detections
            for sign_info in detected_signs:
                if len(sign_info) >= 2:
                    sign_type = sign_info[0]
                    # Draw a simple indicator for traffic signs
                    cv2.putText(frame, f"Traffic Sign: {sign_type}", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['traffic_sign'], 2)
            
            return frame, detected_signs
            
        except Exception as e:
            print(f"Traffic sign detection error: {e}")
            return frame, []
    
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
    
    def display_info(self, frame, detected_zones, person_count, vehicle_count, traffic_sign_count):
        """Display comprehensive detection information"""
        frame_height, frame_width = frame.shape[:2]
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (450, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 220), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ENHANCED UNIFIED DETECTOR", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # FPS and performance metrics
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show target FPS and performance status
        if self.current_fps >= self.target_fps:
            fps_color = (0, 255, 0)  # Green for good performance
            status = "OPTIMAL"
        elif self.current_fps >= self.target_fps * 0.8:
            fps_color = (0, 255, 255)  # Yellow for acceptable
            status = "GOOD"
        elif self.current_fps >= self.target_fps * 0.5:
            fps_color = (0, 165, 255)  # Orange for moderate
            status = "MODERATE"
        else:
            fps_color = (0, 0, 255)  # Red for poor performance
            status = "SLOW"
        
        cv2.putText(frame, f'Target: {self.target_fps} FPS', (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        cv2.putText(frame, f'Status: {status}', (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        
        # Detection counts
        cv2.putText(frame, f'Faces: {person_count}', (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f'Vehicles: {vehicle_count}', (20, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Traffic Signs: {traffic_sign_count}', (20, 185), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Resolution
        cv2.putText(frame, f'Resolution: {frame_width}x{frame_height}', (20, 215), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Active zones
        if detected_zones:
            zone_text = f'Active Zones: {", ".join([self.zone_names[i] for i in detected_zones])}'
            cv2.putText(frame, zone_text, (20, 215), 
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
        """Calculate and update FPS with improved accuracy for normal performance"""
        current_time = time.time()
        self.fps_frame_count += 1
        
        # Calculate FPS every 30 frames for normal updates
        if self.fps_frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            if elapsed_time > 0:
                self.current_fps = self.fps_frame_count / elapsed_time
                
                # Track frame processing time
                if len(self.frame_times) >= 30:
                    self.frame_times.pop(0)  # Remove oldest time
                self.frame_times.append(elapsed_time / self.fps_frame_count)
                
                # Reset counters
                self.start_time = current_time
                self.fps_frame_count = 0
    
    def run(self):
        """Main detection loop"""
        print("Enhanced Unified Detection Started!")
        print("Features: Face Detection + Vehicle Detection + Traffic Sign Detection")
        
        # Test camera connection
        print("\n🔍 Testing camera connection...")
        if not self.test_camera_connection():
            print("❌ Camera connection failed. Exiting...")
            return
        
        # Apply high FPS optimization
        print("\n⚡ Applying high FPS optimization...")
        self.optimize_for_high_fps()
        
        # Show configuration status
        print(f"\n📊 Configuration Status:")
        print(f"   JSON Output: {'✅ Enabled' if self.json_output_enabled else '❌ Disabled'}")
        print(f"   Gemini LLM: {'✅ Enabled' if self.gemini_enabled else '❌ Disabled'}")
        print(f"   Gemini API Key: {'✅ Configured' if self.gemini_api_key else '❌ Missing'}")
        print(f"   Target FPS: {self.target_fps}")
        print(f"   Frame Size: 320x240 (Responsive)")
        print(f"   Lightweight Mode: {'✅ ON' if self.lightweight_mode else '❌ OFF'}")
        print(f"   Voice Feedback: {'✅ ON' if self.voice_enabled else '❌ OFF'}")
        
        print("\n🎮 Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save screenshot")
        print("- Press 'r' to reset FPS counter")
        print("- Press 'f' to toggle face detection zones")
        print("- Press 'v' to toggle vehicle detection")
        print("- Press 't' to toggle traffic sign detection")
        print("- Press 'c' to test camera connection")
        print("- Press 'r' to reconnect to HTTPS camera")
        print("- Press 'l' to toggle lightweight mode")
        print("- Press 'g' to ask Gemini what it sees")
        print("- Press 'j' to toggle JSON output")
        print("- Press 'f' to force camera refresh")
        print("- Press 't' to test Gemini API connection")
        print("- Press 'r' to reset Gemini rate limits")
        print("- Press 's' to show Gemini status")
        print("- Press 'v' to toggle voice feedback")
        print("- Press 'a' to speak current detection summary")
        print("- Press 'x' to test voice feedback")
        print("- Press 'i' to test intelligent AI voice feedback")
        
        # System startup voice announcement
        if self.voice_enabled and self.voice.running:
            if self.gemini_enabled:
                self.voice.announce('startup', "i-sight ready with AI-powered voice feedback. All systems operational.")
            else:
                self.voice.announce('startup', "i-sight ready. All systems operational.")
        
        show_zones = True
        show_vehicles = True
        show_traffic_signs = self.traffic_sign_enabled
        
        # Variables to track detection results for JSON output
        current_faces = []
        current_vehicles = []
        current_traffic_signs = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Setup zones on first frame
            if not self.zones:
                self.setup_detection_zones(frame.shape[1], frame.shape[0])
            
            # Optimize detection processing for normal FPS (30 FPS)
            # Process face detection at normal frequency
            if self.frame_count % 3 == 0:  # Process every 3rd frame for 30 FPS
                detected_zones, person_count, faces = self.detect_people_in_zones(frame)
                current_faces = faces # Update current_faces for JSON
                self.face_count = person_count
                
                # Intelligent voice announcement for face detection (every 30 frames = ~1 second)
                if self.voice_enabled and person_count > 0 and self.frame_count % 30 == 0:
                    self.generate_intelligent_voice_feedback('faces', person_count, frame=frame)
            else:
                detected_zones, person_count = [], 0
                self.face_count = person_count
            
            # Process vehicle detection at normal frequency
            if show_vehicles and self.frame_count % 5 == 0:  # Process every 5th frame for 30 FPS
                frame, vehicle_detections = self.detect_vehicles(frame)
                current_vehicles = vehicle_detections # Update current_vehicles for JSON
                self.vehicle_count = len(vehicle_detections)
                
                # Intelligent voice announcement for vehicle detection (every 60 frames = ~2 seconds)
                if self.voice_enabled and len(vehicle_detections) > 0 and self.frame_count % 60 == 0:
                    try:
                        vehicle_types = []
                        for v in vehicle_detections:
                            if isinstance(v, dict) and 'class' in v:
                                vehicle_types.append(v['class'])
                            elif isinstance(v, str):
                                vehicle_types.append(v)
                            else:
                                vehicle_types.append('vehicle')
                        vehicle_details = f"{', '.join(vehicle_types)}"
                        self.generate_intelligent_voice_feedback('vehicles', len(vehicle_detections), vehicle_details, frame=frame)
                    except Exception as e:
                        self.generate_intelligent_voice_feedback('vehicles', len(vehicle_detections), frame=frame)
            else:
                self.vehicle_count = 0
            
            # Process traffic sign detection at normal frequency
            if show_traffic_signs and self.frame_count % 10 == 0:  # Process every 10th frame for 30 FPS
                frame, traffic_sign_detections = self.detect_traffic_signs(frame)
                current_traffic_signs = traffic_sign_detections # Update current_traffic_signs for JSON
                self.traffic_sign_count = len(traffic_sign_detections)
                
                # Intelligent voice announcement for traffic sign detection (every 90 frames = ~3 seconds)
                if self.voice_enabled and len(traffic_sign_detections) > 0 and self.frame_count % 90 == 0:
                    try:
                        sign_types = []
                        for s in traffic_sign_detections:
                            if isinstance(s, dict) and 'class' in s:
                                sign_types.append(s['class'])
                            elif isinstance(s, str):
                                sign_types.append(s)
                            else:
                                sign_types.append('traffic_sign')
                        sign_details = f"{', '.join(sign_types)}"
                        self.generate_intelligent_voice_feedback('traffic_signs', len(traffic_sign_detections), sign_details, frame=frame)
                    except Exception as e:
                        self.generate_intelligent_voice_feedback('traffic_signs', len(traffic_sign_detections), frame=frame)
            else:
                self.traffic_sign_count = 0
            
            # Draw zones and curves
            if show_zones:
                self.draw_optimized_curves(frame)
                self.add_zone_highlights(frame, detected_zones)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Generate and save JSON output periodically
            current_time = time.time()
            if self.json_output_enabled and current_time - self.last_json_update >= self.json_update_interval:
                detection_data = self.generate_detection_json(
                    detected_zones, self.face_count, self.vehicle_count, self.traffic_sign_count,
                    current_faces, current_vehicles, current_traffic_signs
                )
                self.save_json_output(detection_data)
                
                # Voice feedback is now handled by individual detection functions
                # Removed JSON-based summary to prevent voice overlap
                
                self.last_json_update = current_time
            
            # Display information
            self.display_info(frame, detected_zones, self.face_count, self.vehicle_count, self.traffic_sign_count)
            
            # Show frame
            cv2.imshow('Enhanced Unified Detection', frame)
            
            # Handle key presses - use 1ms delay for normal performance
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"enhanced_detection_{timestamp}.jpg"
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
            elif key == ord('t'):
                if self.traffic_sign_enabled:
                    show_traffic_signs = not show_traffic_signs
                    print(f"Traffic sign detection: {'ON' if show_traffic_signs else 'OFF'}")
                else:
                    print("Traffic sign detection not available")
            elif key == ord('c'):
                print("🔍 Testing camera connection...")
                if self.test_camera_connection():
                    print("✅ Camera connection is working")
                else:
                    print("❌ Camera connection failed")
            elif key == ord('r'):
                if self.reconnect_camera():
                    print("✅ Camera reconnected successfully")
                else:
                    print("❌ Camera reconnection failed")
            elif key == ord('l'):
                self.lightweight_mode = not self.lightweight_mode
                print(f"Lightweight mode: {'ON' if self.lightweight_mode else 'OFF'}")
            elif key == ord('g'):
                if self.gemini_enabled:
                    question = "What can you see in this image?"
                    response = self.query_gemini_vision(frame, question)
                    print(f"Gemini Response: {response}")
                    # Save response to JSON
                    self.save_gemini_response(question, response, self.frame_count)
                else:
                    print("Gemini API not configured. Set GEMINI_API_KEY environment variable.")
            elif key == ord('j'):
                self.json_output_enabled = not self.json_output_enabled
                print(f"JSON output enabled: {'ON' if self.json_output_enabled else 'OFF'}")
            elif key == ord('f'):
                self.force_camera_refresh()
            elif key == ord('t'):
                if self.gemini_enabled:
                    print("🔍 Testing Gemini API connection...")
                    test_response = self.query_gemini_vision(frame, "Test: Can you see this image?")
                    print(f"Gemini Test Response: {test_response}")
                else:
                    print("❌ Gemini API not configured")
            elif key == ord('r'):
                self.reset_gemini_rate_limit()
                print("✅ Gemini rate limits reset")
            elif key == ord('s'):
                status = self.get_gemini_status()
                print(f"Gemini Status: {json.dumps(status, indent=2)}")
            elif key == ord('v'):
                self.voice_enabled = not self.voice_enabled
                if self.voice_enabled and not self.voice.running:
                    # Reinitialize voice if it was stopped
                    self.voice = VoiceManager()
                print(f"Voice feedback: {'ON' if self.voice_enabled else 'OFF'}")
            elif key == ord('a'):
                if self.voice_enabled and self.voice.running:
                    detection_data = self.generate_detection_json(
                        detected_zones, self.face_count, self.vehicle_count, self.traffic_sign_count,
                        current_faces, current_vehicles, current_traffic_signs
                    )
                    self.speak_detection_summary(detection_data)
                else:
                    print("Voice feedback is disabled. Enable it in configuration.")
            elif key == ord('x'):
                self.test_voice()
            elif key == ord('i'):
                if self.voice_enabled:
                    print("🤖 Testing intelligent AI voice feedback...")
                    self.generate_intelligent_voice_feedback('faces', 1, frame=frame)
                    time.sleep(2)
                    self.generate_intelligent_voice_feedback('vehicles', 1, 'car', frame=frame)
                    time.sleep(2)
                    self.generate_intelligent_voice_feedback('traffic_signs', 1, 'stop_sign', frame=frame)
                else:
                    print("❌ Intelligent voice feedback requires voice system to be enabled")
            
            self.frame_count += 1
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")
    
    def cleanup(self):
        """Clean up resources properly"""
        try:
            # Stop voice system
            if hasattr(self, 'voice'):
                self.voice.stop()
            
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Resources cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

def main():
    detector = None
    try:
    detector = EnhancedUnifiedDetector()
    detector.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if detector:
            detector.cleanup()
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main() 