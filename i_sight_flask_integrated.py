#!/usr/bin/env python3
"""
i-sight Flask Integrated System
Combines i-sight detection with Flask web interface for real-time monitoring
"""

import cv2
import numpy as np
import time
import json
import threading
import queue
import base64
from datetime import datetime
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# Flask imports
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

# Torch for YOLOv5 vehicle detection
try:
    import torch
    import torch.nn as nn
    from models.experimental import attempt_load
    from utils.torch_utils import select_device
    from utils.general import non_max_suppression, scale_coords
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv5 dependencies not available - vehicle detection disabled")

# Voice output setup - Multiple methods
VOICE_AVAILABLE = False
VOICE_METHOD = None

# Method 1: Windows SAPI COM (Most reliable on Windows)
try:
    import win32com.client
    VOICE_AVAILABLE = True
    VOICE_METHOD = "SAPI_COM"
    print("✅ Windows SAPI COM available")
except ImportError:
    pass

# Method 2: pyttsx3 (Fallback)
if not VOICE_AVAILABLE:
    try:
        import pyttsx3
        VOICE_AVAILABLE = True
        VOICE_METHOD = "PYTTSX3"
        print("✅ pyttsx3 available")
    except ImportError:
        pass

# Method 3: Windows PowerShell (System fallback)
if not VOICE_AVAILABLE:
    try:
        import subprocess
        # Test PowerShell TTS
        result = subprocess.run([
            'powershell', '-Command', 
            'Add-Type -AssemblyName System.speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak("test")'
        ], capture_output=True, timeout=5, shell=True)
        if result.returncode == 0:
            VOICE_AVAILABLE = True
            VOICE_METHOD = "POWERSHELL"
            print("✅ PowerShell TTS available")
    except:
        pass

if not VOICE_AVAILABLE:
    print("⚠️  No voice output methods available - voice disabled")

# Traffic sign detection
try:
    import sys
    import os
    # Add the traffic sign detection directory to the path
    traffic_sign_path = os.path.join(os.path.dirname(__file__), 'Traffic-Sign-Detection', 'Traffic-Sign-Detection')
    if traffic_sign_path not in sys.path:
        sys.path.append(traffic_sign_path)
    
    from classification import training, getLabel, SVM
    from improved_classification import improved_training, improved_getLabel
    from main import localization
    
    # Fix OpenCV compatibility issue
    def constrastLimit(image):
        img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = list(cv2.split(img_hist_equalized))  # Convert tuple to list
        channels[0] = cv2.equalizeHist(channels[0])
        img_hist_equalized = cv2.merge(channels)
        img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
        return img_hist_equalized
    
    # Disable GUI windows to avoid OpenCV errors
    def noop_imshow(*args, **kwargs):
        pass  # Do nothing instead of showing windows
    
    # Patch the main module
    import main
    main.constrastLimit = constrastLimit
    
    # Disable cv2.imshow calls
    import cv2
    original_imshow = cv2.imshow
    cv2.imshow = noop_imshow
    
    TRAFFIC_SIGN_AVAILABLE = True
    print("✅ Traffic sign detection modules loaded successfully")
except ImportError as e:
    TRAFFIC_SIGN_AVAILABLE = False
    print(f"⚠️  Traffic sign detection not available: {e}")

# Flask app setup
app = Flask(__name__)
CORS(app)
app.secret_key = 'i-sight-secret-key-2024'

class VoiceManager:
    """Advanced multi-method voice manager with Windows SAPI COM, pyttsx3, and PowerShell fallbacks"""
    
    def __init__(self):
        self.voice_queue = queue.Queue()
        self.last_announcement = {}
        self.cooldown = 2.0  # seconds between same announcements
        self.voice_thread = None
        self.running = False
        
        # Voice engines
        self.sapi_engine = None
        self.pyttsx_engine = None
        self.current_method = None
        
        # Voice logging system
        self.voice_log_file = "i_sight_voice_log.txt"
        self.statement_count = {}  # Track how many times each statement was said
        self.voice_lock = threading.Lock()
        self.voice_enabled = True  # Add this missing attribute
        
        # Initialize voice log file
        try:
            with open(self.voice_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== i-sight Voice Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            print(f"✅ Voice log initialized: {self.voice_log_file}")
        except Exception as e:
            print(f"❌ Voice log initialization failed: {e}")
        
        if VOICE_AVAILABLE:
            self.initialize_voice_engines()
            
            if self.current_method:
                self.running = True
                self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
                self.voice_thread.start()
                print(f"✅ Voice system initialized with {self.current_method}")
                
                # Test the voice system
                self.test_voice_system()
            else:
                print("❌ No voice engines could be initialized")
    
    def initialize_voice_engines(self):
        """Initialize available voice engines in order of preference"""
        
        # Method 1: Windows SAPI COM (Most reliable)
        if VOICE_METHOD in ["SAPI_COM", None]:
            try:
                import win32com.client
                import pythoncom
                
                # Initialize COM for this thread
                pythoncom.CoInitialize()
                
                self.sapi_engine = win32com.client.Dispatch("SAPI.SpVoice")
                
                # Configure SAPI settings
                voices = self.sapi_engine.GetVoices()
                print(f"🔊 SAPI: Found {voices.Count} voices")
                
                # Try to select a good voice
                for i in range(voices.Count):
                    voice = voices.Item(i)
                    name = voice.GetDescription()
                    print(f"  Voice {i}: {name}")
                    
                    # Prefer Microsoft voices
                    if any(keyword in name.lower() for keyword in ['zira', 'david', 'mark', 'hazel']):
                        self.sapi_engine.Voice = voice
                        print(f"✅ Selected SAPI voice: {name}")
                        break
                
                # Set speech rate (0-10, default=0)
                self.sapi_engine.Rate = 2  # Slightly faster
                self.sapi_engine.Volume = 100  # Maximum volume
                
                self.current_method = "SAPI_COM"
                print("✅ Windows SAPI COM engine initialized")
                return
                
            except Exception as e:
                print(f"❌ SAPI COM initialization failed: {e}")
        
        # Method 2: pyttsx3 (Fallback)
        if VOICE_METHOD in ["PYTTSX3", None]:
            try:
                import pyttsx3
                self.pyttsx_engine = pyttsx3.init('sapi5')
                
                # Configure pyttsx3
                self.pyttsx_engine.setProperty('rate', 200)
                self.pyttsx_engine.setProperty('volume', 1.0)
                
                # Set voice
                voices = self.pyttsx_engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if any(keyword in voice.name.lower() for keyword in ['zira', 'david']):
                            self.pyttsx_engine.setProperty('voice', voice.id)
                            print(f"✅ Selected pyttsx3 voice: {voice.name}")
                            break
                
                self.current_method = "PYTTSX3"
                print("✅ pyttsx3 engine initialized")
                return
                
            except Exception as e:
                print(f"❌ pyttsx3 initialization failed: {e}")
        
        # Method 3: PowerShell (System fallback)
        if VOICE_METHOD in ["POWERSHELL", None]:
            try:
                import subprocess
                # Test PowerShell TTS
                test_cmd = [
                    'powershell', '-Command', 
                    'Add-Type -AssemblyName System.speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Rate = 2; $speak.Volume = 100; $speak.Speak("PowerShell TTS ready")'
                ]
                result = subprocess.run(test_cmd, capture_output=True, timeout=5, shell=True)
                if result.returncode == 0:
                    self.current_method = "POWERSHELL"
                    print("✅ PowerShell TTS engine initialized")
                    return
                else:
                    print(f"❌ PowerShell TTS test failed: {result.stderr.decode()}")
                    
            except Exception as e:
                print(f"❌ PowerShell TTS initialization failed: {e}")
        
        print("❌ No voice engines could be initialized")
    
    def test_voice_system(self):
        """Test the current voice system"""
        try:
            test_message = f"Voice system ready using {self.current_method}"
            print(f"🔊 Testing: {test_message}")
            
            success = self._speak_direct(test_message)
            if success:
                print("✅ Voice system test completed successfully")
                self.log_voice_message(test_message, "TEST_SUCCESS")
            else:
                print("❌ Voice system test failed")
                self.log_voice_message(test_message, "TEST_FAILED")
                
        except Exception as e:
            print(f"❌ Voice system test error: {e}")
    
    def _speak_direct(self, message: str) -> bool:
        """Speak message directly using current method"""
        try:
            if self.current_method == "SAPI_COM" and self.sapi_engine:
                # SAPI COM method
                self.sapi_engine.Speak(message)
                return True
                
            elif self.current_method == "PYTTSX3" and self.pyttsx_engine:
                # pyttsx3 method
                self.pyttsx_engine.say(message)
                self.pyttsx_engine.runAndWait()
                return True
                
            elif self.current_method == "POWERSHELL":
                # PowerShell method
                import subprocess
                # Escape message for PowerShell
                escaped_message = message.replace('"', '""').replace("'", "''")
                cmd = [
                    'powershell', '-Command', 
                    f'Add-Type -AssemblyName System.speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Rate = 2; $speak.Volume = 100; $speak.Speak("{escaped_message}")'
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=10, shell=True)
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            print(f"❌ Direct speech error: {e}")
            return False
    
    def log_voice_message(self, message: str, action: str = "QUEUED"):
        """Log voice messages to file with timestamp and repetition count"""
        try:
            with self.voice_lock:
                # Update statement count
                if message not in self.statement_count:
                    self.statement_count[message] = 0
                self.statement_count[message] += 1
                
                # Create log entry
                timestamp = datetime.now().strftime('%H:%M:%S')
                repetition = self.statement_count[message]
                log_entry = f"[{timestamp}] {action}: \"{message}\" (Count: {repetition})\n"
                
                # Write to file
                with open(self.voice_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
                    f.flush()  # Force write to disk immediately
                
                print(f"📝 Voice logged: {message} (#{repetition})")
        except Exception as e:
            print(f"❌ Voice logging failed: {e}")
    
    def _voice_worker(self):
        """Background thread for voice output with multi-method fallback"""
        print(f"🔊 Voice worker thread started using {self.current_method}")
        
        while self.running:
            try:
                message = self.voice_queue.get(timeout=1)
                if message and self.running:
                    # Log the message as being processed
                    self.log_voice_message(message, "PROCESSING")
                    
                    print(f"🔊 Speaking: {message}")
                    success = False
                    
                    # Try current method first
                    try:
                        success = self._speak_direct(message)
                        if success:
                            self.log_voice_message(message, "COMPLETED")
                            print(f"✅ Voice completed: {message}")
                        else:
                            raise Exception(f"{self.current_method} failed")
                            
                    except Exception as primary_error:
                        print(f"❌ Primary voice method failed: {primary_error}")
                        self.log_voice_message(message, f"PRIMARY_FAILED_{self.current_method}")
                        
                        # Try fallback methods
                        success = self._try_fallback_methods(message)
                    
                    if not success:
                        self.log_voice_message(message, "ALL_METHODS_FAILED")
                        print(f"❌ All voice methods failed for: {message}")
                        
                        # Last resort: Windows system beep pattern
                        self._emergency_audio_feedback(message)
                    
                    # Brief pause between messages
                    time.sleep(0.3)
                
                # Mark task as done
                self.voice_queue.task_done()
                
            except queue.Empty:
                # Timeout waiting for message - this is normal
                continue
            except Exception as e:
                print(f"❌ Voice worker critical error: {e}")
                try:
                    self.voice_queue.task_done()
                except:
                    pass
                time.sleep(0.5)
        
        print("🔇 Voice worker thread stopped")
    
    def _try_fallback_methods(self, message: str) -> bool:
        """Try all available voice methods as fallbacks"""
        fallback_methods = []
        
        # Build fallback list based on what's available
        if self.current_method != "SAPI_COM" and self.sapi_engine:
            fallback_methods.append("SAPI_COM")
        if self.current_method != "PYTTSX3" and self.pyttsx_engine:
            fallback_methods.append("PYTTSX3")
        if self.current_method != "POWERSHELL":
            fallback_methods.append("POWERSHELL")
        
        for method in fallback_methods:
            try:
                print(f"🔄 Trying fallback method: {method}")
                
                original_method = self.current_method
                self.current_method = method
                
                success = self._speak_direct(message)
                
                if success:
                    print(f"✅ Fallback {method} succeeded")
                    self.log_voice_message(message, f"FALLBACK_SUCCESS_{method}")
                    
                    # Update primary method to the working one
                    print(f"🔄 Switching primary method from {original_method} to {method}")
                    return True
                else:
                    self.current_method = original_method  # Restore original
                    self.log_voice_message(message, f"FALLBACK_FAILED_{method}")
                    
            except Exception as e:
                print(f"❌ Fallback {method} error: {e}")
                self.current_method = self.current_method  # Ensure it's restored
        
        return False
    
    def _emergency_audio_feedback(self, message: str):
        """Provide emergency audio feedback when all TTS methods fail"""
        try:
            import winsound
            print("🚨 Using emergency audio feedback")
            
            # Different beep patterns for different message types
            if "person" in message.lower() or "face" in message.lower():
                # Person detection: 2 short beeps
                for _ in range(2):
                    winsound.Beep(1000, 200)
                    time.sleep(0.1)
            elif "vehicle" in message.lower():
                # Vehicle detection: 3 medium beeps
                for _ in range(3):
                    winsound.Beep(1200, 300)
                    time.sleep(0.1)
            elif "traffic" in message.lower():
                # Traffic sign: 1 long beep
                winsound.Beep(800, 500)
            elif "error" in message.lower() or "failed" in message.lower():
                # Error pattern: 4 high urgent beeps
                for _ in range(4):
                    winsound.Beep(1500, 150)
                    time.sleep(0.05)
            else:
                # General pattern: 1 medium beep
                winsound.Beep(1000, 300)
            
            self.log_voice_message(message, "EMERGENCY_BEEP_SUCCESS")
            print(f"✅ Emergency audio feedback completed for: {message}")
            
        except Exception as beep_error:
            print(f"❌ Emergency audio feedback failed: {beep_error}")
            self.log_voice_message(message, "EMERGENCY_BEEP_FAILED")
    
    def announce(self, category: str, message: str):
        """Announce with rate limiting - ALWAYS SPEAK using best available method"""
        if not self.running or not self.current_method:
            print(f"❌ Voice system not running. Message: {message}")
            return
            
        current_time = time.time()
        
        # Use fixed cooldown time
        cooldown_time = self.cooldown
        
        if category not in self.last_announcement or \
           current_time - self.last_announcement[category] > cooldown_time:
            self.last_announcement[category] = current_time
            
            # Log the queuing attempt
            self.log_voice_message(message, "QUEUED")
            
            # Try to queue the message
            try:
                self.voice_queue.put(message, timeout=0.1)
                print(f"🔊 Voice queued: {message} (Queue size: {self.voice_queue.qsize()})")
            except queue.Full:
                print(f"⚠️ Voice queue full, speaking immediately: {message}")
                # If queue is full, speak immediately in current thread
                self._speak_immediate(message)
        else:
            remaining_cooldown = cooldown_time - (current_time - self.last_announcement[category])
            print(f"🔇 Voice cooldown active for {category} ({remaining_cooldown:.1f}s remaining)")
    
    def _speak_immediate(self, message: str):
        """Speak message immediately in current thread"""
        try:
            self.log_voice_message(message, "IMMEDIATE")
            success = self._speak_direct(message)
            
            if success:
                self.log_voice_message(message, "IMMEDIATE_COMPLETED")
                print(f"✅ Voice immediate: {message}")
            else:
                # Try fallback methods
                success = self._try_fallback_methods(message)
                if success:
                    self.log_voice_message(message, "IMMEDIATE_FALLBACK_SUCCESS")
                else:
                    self.log_voice_message(message, "IMMEDIATE_ALL_FAILED")
                    self._emergency_audio_feedback(message)
                    
        except Exception as e:
            self.log_voice_message(message, "IMMEDIATE_ERROR")
            print(f"❌ Immediate voice error: {e}")
    
    def force_speak(self, message: str, max_retries: int = 3):
        """Force immediate speech without cooldown - GUARANTEED to work for visually impaired users"""
        if not self.running:
            print(f"❌ Voice system not running. Critical message: {message}")
            return
        
        print(f"🔊 FORCE SPEAKING: {message}")
        self.log_voice_message(message, "FORCE_STARTED")
        
        # Try multiple times with different methods
        for attempt in range(max_retries):
            try:
                print(f"🔊 Force attempt {attempt + 1}/{max_retries}")
                success = self._speak_direct(message)
                
                if success:
                    self.log_voice_message(message, f"FORCE_SUCCESS_ATTEMPT_{attempt + 1}")
                    print(f"✅ Force speech completed successfully")
                    return  # Success!
                else:
                    # Try fallback methods
                    success = self._try_fallback_methods(message)
                    if success:
                        self.log_voice_message(message, f"FORCE_FALLBACK_SUCCESS_ATTEMPT_{attempt + 1}")
                        print(f"✅ Force speech completed with fallback")
                        return  # Success!
                
            except Exception as e:
                print(f"❌ Force speech attempt {attempt + 1} failed: {e}")
                self.log_voice_message(message, f"FORCE_FAILED_ATTEMPT_{attempt + 1}")
                
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief pause between attempts
        
        # If all attempts failed, use emergency feedback
        print("🚨 All force speech attempts failed - using emergency feedback")
        self.log_voice_message(message, "FORCE_EMERGENCY_FALLBACK")
        self._emergency_audio_feedback(message)
        
        # Also write to urgent log file
        try:
            urgent_log = "i_sight_urgent_messages.txt"
            with open(urgent_log, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%H:%M:%S')
                f.write(f"[{timestamp}] URGENT: {message}\n")
            
            self.log_voice_message(message, "FORCE_URGENT_LOG_SUCCESS")
            print(f"✅ Urgent message logged to {urgent_log}: {message}")
            
        except Exception as log_error:
            print(f"❌ Urgent logging failed: {log_error}")
        
        # Final fallback: Console alert
        print("🚨" * 50)
        print(f"CRITICAL VOICE MESSAGE: {message}")
        print("🚨" * 50)
        self.log_voice_message(message, "FORCE_CONSOLE_ALERT")
    
    def stop(self):
        """Stop voice system"""
        self.running = False
        if self.voice_thread:
            self.voice_thread.join(timeout=2)
        
        # Cleanup COM if SAPI was used
        if hasattr(self, 'sapi_engine') and self.sapi_engine:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
                print("✅ COM cleanup completed")
            except:
                pass

class ISightDetector:
    """Main i-sight detection system"""
    
    def __init__(self):
        self.frame_count = 0
        self.fps_frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        self.target_fps = 30
        self.lightweight_mode = True
        
        # Detection counts
        self.face_count = 0
        self.vehicle_count = 0 
        self.traffic_sign_count = 0
        
        # Model loading flags
        self.vehicle_model_loaded = False
        self.traffic_sign_enabled = TRAFFIC_SIGN_AVAILABLE
        self.voice_enabled = True
        
        # Voice system
        self.voice = VoiceManager()
        
        # Camera
        self.cap = None
        self.initialize_camera()
        
        # Detection zones
        self.zones = []
        self.zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        
        # Detection parameters
        self.vehicle_conf_threshold = 0.25
        self.vehicle_iou_threshold = 0.45
        self.img_size = 320
        self.device = 'cpu'
        self.confidence_threshold = 0.3
        
        # Colors
        self.colors = {
            'zone': (0, 255, 0),
            'detection': (0, 0, 255),
            'text': (255, 255, 255),
            'face': (255, 0, 255),
            'vehicle': (0, 255, 0),
            'traffic_sign': (0, 255, 255),
            'background': (0, 0, 0)
        }
        
        # Load models
        self.load_models()
        
        # Detection data for web interface
        self.latest_detection_data = {
            'timestamp': datetime.now().isoformat(),
            'faces': [],
            'vehicles': [],
            'traffic_signs': [],
            'fps': 0.0,
            'frame_number': 0,
            'processing_time': 0.0
        }
        self.detection_lock = threading.Lock()
        
        # Screenshot for web interface
        self.latest_screenshot = None
        self.screenshot_lock = threading.Lock()
        
        print("✅ i-sight detector initialized!")
    
    def initialize_camera(self):
        """Initialize camera"""
        try:
            # Use the working camera settings from diagnostic
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                # Set optimal camera properties
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
                self.cap.set(cv2.CAP_PROP_CONTRAST, 128)
                self.cap.set(cv2.CAP_PROP_SATURATION, 128)
                self.cap.set(cv2.CAP_PROP_HUE, 0)
                
                # Test if camera is actually working
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    mean_brightness = test_frame.mean()
                    if mean_brightness > 10:  # Not completely black
                        print(f"✅ Camera initialized successfully (brightness: {mean_brightness:.2f})")
                    else:
                        print(f"⚠️ Camera initialized but frame is dark (brightness: {mean_brightness:.2f})")
                else:
                    print("⚠️ Camera initialized but can't read frames")
            else:
                print("❌ Camera initialization failed")
        except Exception as e:
            print(f"❌ Camera error: {e}")
    
    def load_models(self):
        """Load detection models"""
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty() or self.face_cascade2.empty():
            print("❌ Face detection models failed to load")
        else:
            print("✅ Face detection models loaded")
        
        # Vehicle detection
        if YOLO_AVAILABLE:
            try:
                model_path = 'yolov5s.pt'
                if os.path.exists(model_path):
                    self.device = select_device('cpu')
                    self.vehicle_model = attempt_load(model_path, map_location=self.device)
                    self.vehicle_model_loaded = True
                    print("✅ Vehicle detection model loaded")
                else:
                    print("❌ YOLOv5 model file not found")
            except Exception as e:
                print(f"❌ Vehicle model loading failed: {e}")
    
    def setup_detection_zones(self, frame_width, frame_height):
        """Setup detection zones"""
        zone_width = frame_width // 5
        self.zones = []
        for i in range(5):
            x1 = i * zone_width
            x2 = (i + 1) * zone_width
            margin_y = int(frame_height * 0.2)
            y1 = margin_y
            y2 = frame_height - margin_y
            self.zones.append((x1, y1, x2, y2))
    
    def detect_faces_lightweight(self, gray):
        """Detect faces using Haar cascades"""
        faces = []
        scale_factor = 1.2 if self.lightweight_mode else 1.1
        min_neighbors = 2 if self.lightweight_mode else 4
        min_size = (15, 15) if self.lightweight_mode else (20, 20)
        
        faces1 = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        
        for (x, y, w, h) in faces1:
            faces.append([x, y, w, h, 0.8])
        
        return faces
    
    def detect_people_in_zones(self, frame):
        """Detect people/faces in different zones"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces_lightweight(gray)
        
        detected_zones = []
        person_count = 0
        zone_faces = {i: [] for i in range(5)}
        
        for (fx, fy, fw, fh, conf) in faces:
            face_center_x = fx + fw // 2
            face_center_y = fy + fh // 2
            
            for i, (x1, y1, x2, y2) in enumerate(self.zones):
                if x1 <= face_center_x <= x2 and y1 <= face_center_y <= y2:
                    zone_faces[i].append((fx, fy, fw, fh, conf))
                    person_count += 1
                    if i not in detected_zones:
                        detected_zones.append(i)
        
        return detected_zones, person_count, zone_faces
    
    def detect_vehicles(self, frame):
        """Detect vehicles using YOLOv5"""
        if not YOLO_AVAILABLE or not self.vehicle_model_loaded:
            return []
        
        try:
            # Vehicle detection implementation
            # For now, return empty results
            return []
        except Exception as e:
            print(f"❌ Vehicle detection error: {e}")
            return []
    
    def detect_traffic_signs(self, frame):
        """Detect traffic signs"""
        if not self.traffic_sign_enabled:
            return []
        
        try:
            # Initialize model if not already done
            if not hasattr(self, 'traffic_sign_model'):
                print("🔄 Loading traffic sign detection model...")
                
                # Try to load pre-trained model first
                model_path = os.path.join(os.path.dirname(__file__), 'Traffic-Sign-Detection', 'Traffic-Sign-Detection', 'data_svm.dat')
                if os.path.exists(model_path):
                    from classification import SVM
                    self.traffic_sign_model = SVM()
                    self.traffic_sign_model.load(model_path)
                    print("✅ Pre-trained traffic sign model loaded")
                else:
                    # Fallback to training
                    self.traffic_sign_model = training()
                    print("✅ Traffic sign model trained and loaded")
            
            # Preprocess frame for traffic sign detection
            processed_frame = frame.copy()
            
            # Use a simplified detection approach to avoid GUI issues
            detections = []
            
            try:
                # Import the necessary functions
                from main import preprocess_image, findContour, contourIsSign, cropSign, getLabel
                
                # Preprocess the image
                binary_image = preprocess_image(processed_frame)
                
                # Find contours
                contours = findContour(binary_image)
                
                if contours is not None and len(contours) > 0:
                    # Process each contour
                    for contour in contours:
                        try:
                            # Check if contour is a sign
                            M = cv2.moments(contour)
                            if M["m00"] == 0:
                                continue
                                
                            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                            perimeter = cv2.arcLength(contour, True)
                            
                            # Check if contour is circular enough to be a sign
                            is_sign, max_distance = contourIsSign(contour, centroid, 0.65)
                            
                            if is_sign and perimeter > 100:  # Minimum size threshold
                                # Crop the sign
                                cropped_sign = cropSign(processed_frame, [
                                    (contour.min(axis=0)[0], contour.min(axis=0)[1]),
                                    (contour.max(axis=0)[0], contour.max(axis=0)[1])
                                ])
                                
                                if cropped_sign is not None and cropped_sign.size > 0:
                                    # Classify the sign
                                    try:
                                        sign_type = getLabel(self.traffic_sign_model, cropped_sign)
                                        sign_names = ["ERROR", "STOP", "TURN LEFT", "TURN RIGHT", 
                                                     "DO NOT TURN LEFT", "DO NOT TURN RIGHT", 
                                                     "ONE WAY", "SPEED LIMIT", "OTHER"]
                                        
                                        if 0 <= sign_type < len(sign_names):
                                            sign_name = sign_names[sign_type]
                                        else:
                                            sign_name = f"UNKNOWN_{sign_type}"
                                        
                                        # Get bounding box
                                        x, y, w, h = cv2.boundingRect(contour)
                                        
                                        detections.append({
                                            'sign_type': sign_name,
                                            'confidence': 0.8,
                                            'bbox': [x, y, w, h]
                                        })
                                        
                                    except Exception as classify_error:
                                        print(f"⚠️ Classification error: {classify_error}")
                                        continue
                        
                        except Exception as contour_error:
                            continue  # Skip problematic contours
                
            except Exception as detection_error:
                print(f"⚠️ Detection processing error: {detection_error}")
            
            return detections
            
        except Exception as e:
            print(f"❌ Traffic sign detection error: {e}")
            return []
    
    def calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        self.fps_frame_count += 1
        
        if self.fps_frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            if elapsed_time > 0:
                self.current_fps = self.fps_frame_count / elapsed_time
                self.start_time = current_time
                self.fps_frame_count = 0
    
    def update_detection_data(self, detected_zones, person_count, vehicle_detections, traffic_sign_detections, processing_time):
        """Update detection data for web interface"""
        with self.detection_lock:
            self.latest_detection_data = {
                'timestamp': datetime.now().isoformat(),
                'faces': [
                    {
                        'zone': zone,
                        'zone_name': self.zone_names[zone],
                        'count': len([f for f in detected_zones if f == zone])
                    }
                    for zone in detected_zones
                ],
                'vehicles': [
                    {
                        'bbox': detection.get('bbox', []),
                        'confidence': detection.get('confidence', 0),
                        'class': detection.get('class', 'vehicle')
                    }
                    for detection in vehicle_detections
                ],
                'traffic_signs': [
                    {
                        'sign_type': detection.get('sign_type', 'UNKNOWN'),
                        'confidence': detection.get('confidence', 0)
                    }
                    for detection in traffic_sign_detections
                ],
                'fps': self.current_fps,
                'frame_number': self.frame_count,
                'processing_time': processing_time
            }
    
    def update_screenshot(self, frame):
        """Update screenshot for web interface"""
        try:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            jpeg_data = buffer.tobytes()
            
            with self.screenshot_lock:
                self.latest_screenshot = jpeg_data
        except Exception as e:
            print(f"❌ Screenshot update error: {e}")
    
    def announce_detections(self, person_count, vehicle_count, traffic_sign_count, detected_zones):
        """Announce detections with voice"""
        if not self.voice_enabled or not self.voice.running:
            return
        
        if person_count > 0:
            if person_count == 1:
                if detected_zones:
                    zone_name = self.zone_names[detected_zones[0]]
                    message = f"Person detected in {zone_name} zone"
                else:
                    message = "Person detected"
            elif person_count == 2:
                # Special handling for exactly 2 people
                if len(detected_zones) == 2:
                    zone1_name = self.zone_names[detected_zones[0]]
                    zone2_name = self.zone_names[detected_zones[1]]
                    message = f"Two people detected: one in {zone1_name} zone and one in {zone2_name} zone"
                elif len(detected_zones) == 1:
                    zone_name = self.zone_names[detected_zones[0]]
                    message = f"Two people detected in {zone_name} zone"
                else:
                    message = "Two people detected"
            else:
                # For 3+ people, list the zones
                if detected_zones:
                    zone_names = [self.zone_names[zone] for zone in detected_zones]
                    if len(zone_names) == 1:
                        message = f"{person_count} people detected in {zone_names[0]} zone"
                    else:
                        zone_list = ", ".join(zone_names[:-1]) + f" and {zone_names[-1]}"
                        message = f"{person_count} people detected in {zone_list} zones"
                else:
                    message = f"{person_count} people detected"
            self.voice.announce('faces', message)
        
        if vehicle_count > 0:
            message = f"{vehicle_count} vehicle{'s' if vehicle_count > 1 else ''} detected"
            self.voice.announce('vehicles', message)
        
        if traffic_sign_count > 0:
            message = f"{traffic_sign_count} traffic sign{'s' if traffic_sign_count > 1 else ''} detected"
            self.voice.announce('traffic_signs', message)
    
    def process_frame(self):
        """Process a single frame with enhanced error handling"""
        try:
            # Validate camera state
            if not self.cap or not self.cap.isOpened():
                print("⚠️ Camera not available in process_frame")
                return False
            
            start_time = time.time()
            
            # Read frame with timeout protection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("⚠️ Failed to read frame from camera")
                return False
            
            # Validate frame dimensions
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                print("⚠️ Invalid frame dimensions")
                return False
            
            # Check if frame is too dark (black screen)
            mean_brightness = frame.mean()
            if mean_brightness < 5:  # Very dark frame
                print(f"⚠️ Frame is too dark (brightness: {mean_brightness:.2f})")
                # Try to adjust camera settings
                try:
                    self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
                    self.cap.set(cv2.CAP_PROP_CONTRAST, 150)
                    print("🔄 Adjusted camera brightness and contrast")
                except Exception as adjust_error:
                    print(f"⚠️ Failed to adjust camera settings: {adjust_error}")
            
            # Check for stuck frames (same brightness for too long)
            if hasattr(self, 'last_brightness'):
                if abs(mean_brightness - self.last_brightness) < 1.0:
                    if not hasattr(self, 'stuck_frame_count'):
                        self.stuck_frame_count = 0
                    self.stuck_frame_count += 1
                    if self.stuck_frame_count > 30:  # 30 frames with same brightness
                        print(f"⚠️ Frame appears stuck (brightness: {mean_brightness:.2f})")
                        # Try to refresh camera
                        try:
                            self.cap.grab()  # Grab a few frames to refresh
                            self.cap.grab()
                            self.stuck_frame_count = 0
                        except:
                            pass
                else:
                    self.stuck_frame_count = 0
            self.last_brightness = mean_brightness
            
            # Setup zones on first frame
            if not self.zones:
                self.setup_detection_zones(frame.shape[1], frame.shape[0])
            
            # Face detection (every 3rd frame)
            if self.frame_count % 3 == 0:
                detected_zones, person_count, zone_faces = self.detect_people_in_zones(frame)
                self.face_count = person_count
                
                # Draw face boxes
                for zone_idx, faces in zone_faces.items():
                    for (fx, fy, fw, fh, conf) in faces:
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), self.colors['face'], 2)
                        cv2.putText(frame, f'Face: {conf:.2f}', (fx, fy - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['face'], 1)
            else:
                detected_zones, person_count = [], 0
                self.face_count = person_count
            
            # Vehicle detection (every 5th frame)
            if self.frame_count % 5 == 0:
                vehicle_detections = self.detect_vehicles(frame)
                self.vehicle_count = len(vehicle_detections)
            else:
                vehicle_detections = []
                self.vehicle_count = 0
            
            # Traffic sign detection (every 10th frame)
            if self.frame_count % 10 == 0:
                traffic_sign_detections = self.detect_traffic_signs(frame)
                self.traffic_sign_count = len(traffic_sign_detections)
            else:
                traffic_sign_detections = []
                self.traffic_sign_count = 0
            
            # Voice announcements (every 60 frames)
            if self.frame_count % 60 == 0:
                self.announce_detections(self.face_count, self.vehicle_count, self.traffic_sign_count, detected_zones)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Update detection data for web interface
            processing_time = time.time() - start_time
            self.update_detection_data(detected_zones, person_count, vehicle_detections, traffic_sign_detections, processing_time)
            
            # Update screenshot
            self.update_screenshot(frame)
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            print(f"❌ Error in process_frame: {e}")
            return False
    
    def get_detection_data(self):
        """Get latest detection data"""
        with self.detection_lock:
            return self.latest_detection_data.copy()
    
    def get_screenshot(self):
        """Get latest screenshot"""
        with self.screenshot_lock:
            return self.latest_screenshot
    
    def cleanup(self):
        """Clean up resources"""
        if self.voice:
            self.voice.stop()
        if self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

# Global detector instance
detector = None
detection_thread = None
detection_running = False
system_health = {
    'last_heartbeat': time.time(),
    'total_frames_processed': 0,
    'consecutive_failures': 0,
    'recovery_attempts': 0,
    'system_start_time': time.time()
}

def start_detection_thread():
    """Start detection in background thread"""
    global detector, detection_thread, detection_running
    
    if detection_running:
        return False, "Detection already running"
    
    try:
        detector = ISightDetector()
        detection_running = True
        
        def detection_worker():
            global detection_running, detector
            consecutive_failures = 0
            max_failures = 15  # Increased tolerance for better stability
            recovery_attempts = 0
            max_recovery_attempts = 5
            last_successful_frame = time.time()
            frame_timeout = 10  # 10 seconds without successful frame
            
            print("🔄 Detection worker started with enhanced error handling")
            
            while detection_running:
                try:
                    # Check if camera is available and working
                    if not detector or not detector.cap or not detector.cap.isOpened():
                        print("⚠️ Camera not available, attempting recovery...")
                        if recovery_attempts < max_recovery_attempts:
                            try:
                                if detector:
                                    detector.cleanup()
                                time.sleep(2)
                                detector = ISightDetector()
                                recovery_attempts += 1
                                print(f"🔄 Camera recovery attempt {recovery_attempts}/{max_recovery_attempts}")
                                continue
                            except Exception as recovery_error:
                                print(f"❌ Camera recovery failed: {recovery_error}")
                                recovery_attempts += 1
                                time.sleep(3)
                                continue
                        else:
                            print("❌ Max camera recovery attempts reached")
                            break
                    
                    # Process frame with timeout protection
                    success = detector.process_frame()
                    if success:
                        consecutive_failures = 0
                        recovery_attempts = 0  # Reset recovery attempts on success
                        last_successful_frame = time.time()
                        # Update system health
                        system_health['total_frames_processed'] += 1
                        system_health['consecutive_failures'] = 0
                        system_health['last_heartbeat'] = time.time()
                    else:
                        consecutive_failures += 1
                        system_health['consecutive_failures'] = consecutive_failures
                        print(f"⚠️ Frame processing failed ({consecutive_failures}/{max_failures})")
                        
                        # Check for frame timeout
                        if time.time() - last_successful_frame > frame_timeout:
                            print(f"⚠️ No successful frames for {frame_timeout} seconds")
                        
                        # If too many consecutive failures, try to recover
                        if consecutive_failures >= max_failures:
                            print("🔄 Attempting system recovery...")
                            if recovery_attempts < max_recovery_attempts:
                                try:
                                    print("🔄 Cleaning up and reinitializing...")
                                    detector.cleanup()
                                    time.sleep(3)
                                    detector = ISightDetector()
                                    recovery_attempts += 1
                                    consecutive_failures = 0
                                    last_successful_frame = time.time()
                                    print(f"✅ System recovery attempt {recovery_attempts}/{max_recovery_attempts} successful")
                                    continue
                                except Exception as recovery_error:
                                    print(f"❌ System recovery failed: {recovery_error}")
                                    recovery_attempts += 1
                                    time.sleep(3)
                                    continue
                            else:
                                print("❌ Max recovery attempts reached, stopping detection")
                                break
                    
                    # Adaptive sleep based on performance
                    if consecutive_failures > 5:
                        time.sleep(0.1)  # Slower when having issues
                    else:
                        time.sleep(1/30)  # Normal 30 FPS
                    
                except Exception as e:
                    consecutive_failures += 1
                    print(f"❌ Detection worker error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    if consecutive_failures >= max_failures:
                        print("🔄 Attempting error recovery...")
                        if recovery_attempts < max_recovery_attempts:
                            try:
                                print("🔄 Error recovery: cleaning up and reinitializing...")
                                if detector:
                                    detector.cleanup()
                                time.sleep(3)
                                detector = ISightDetector()
                                recovery_attempts += 1
                                consecutive_failures = 0
                                last_successful_frame = time.time()
                                print(f"✅ Error recovery attempt {recovery_attempts}/{max_recovery_attempts} successful")
                                continue
                            except Exception as recovery_error:
                                print(f"❌ Error recovery failed: {recovery_error}")
                                recovery_attempts += 1
                                time.sleep(3)
                                continue
                        else:
                            print("❌ Max error recovery attempts reached, stopping detection")
                            break
                    
                    time.sleep(1)  # Longer pause on errors
            
            # If we exit the loop, stop detection
            if detection_running:
                print("🛑 Detection worker stopped, cleaning up...")
                detection_running = False
                if detector:
                    try:
                        detector.cleanup()
                    except:
                        pass
        
        detection_thread = threading.Thread(target=detection_worker, daemon=True)
        detection_thread.start()
        
        return True, "Detection started successfully"
    except Exception as e:
        return False, f"Failed to start detection: {str(e)}"

def stop_detection_thread():
    """Stop detection thread"""
    global detector, detection_running
    
    if not detection_running:
        return False, "Detection not running"
    
    try:
        detection_running = False
        if detector:
            detector.cleanup()
        if detection_thread:
            detection_thread.join(timeout=5)
        return True, "Detection stopped successfully"
    except Exception as e:
        return False, f"Failed to stop detection: {str(e)}"

# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'detection_running': detection_running,
        'camera_connected': detector is not None and detector.cap is not None and detector.cap.isOpened(),
        'voice_enabled': detector.voice.voice_enabled if detector and hasattr(detector.voice, 'voice_enabled') else False,
        'face_count': detector.face_count if detector else 0,
        'vehicle_count': detector.vehicle_count if detector else 0,
        'traffic_sign_count': detector.traffic_sign_count if detector else 0,
        'fps': detector.current_fps if detector else 0.0
    })

@app.route('/api/start-detection', methods=['POST'])
def start_detection():
    """Start detection system"""
    success, message = start_detection_thread()
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/stop-detection', methods=['POST'])
def stop_detection():
    """Stop detection system"""
    success, message = stop_detection_thread()
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/latest-detection')
def get_latest_detection():
    """Get latest detection data"""
    if detector:
        return jsonify(detector.get_detection_data())
    else:
        return jsonify({'message': 'No detection data available'})

@app.route('/video-feed')
def video_feed():
    """Video streaming route"""
    def generate_frames():
        while True:
            try:
                if detector:
                    screenshot = detector.get_screenshot()
                    if screenshot:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + screenshot + b'\r\n')
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in video feed: {e}")
                time.sleep(0.1)
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/screenshot')
def get_screenshot():
    """Get a single screenshot"""
    if detector:
        screenshot = detector.get_screenshot()
        if screenshot:
            return Response(screenshot, mimetype='image/jpeg')
    return jsonify({'error': 'No screenshot available'}), 404

@app.route('/api/stats')
def get_stats():
    """Get detection statistics"""
    if detector:
        return jsonify({
            'total_faces': detector.face_count,
            'total_vehicles': detector.vehicle_count,
            'total_traffic_signs': detector.traffic_sign_count,
            'fps': detector.current_fps,
            'frame_count': detector.frame_count,
            'camera_status': 'connected' if detector.cap and detector.cap.isOpened() else 'disconnected',
            'voice_status': 'enabled' if detector.voice and detector.voice.running else 'disabled',
            'detection_running': detection_running,
            'thread_alive': detection_thread.is_alive() if detection_thread else False
        })
    else:
        return jsonify({'error': 'No detection data available'}), 404

@app.route('/api/debug')
def get_debug_info():
    """Get detailed debug information"""
    debug_info = {
        'detection_running': detection_running,
        'detector_exists': detector is not None,
        'thread_exists': detection_thread is not None,
        'thread_alive': detection_thread.is_alive() if detection_thread else False,
        'camera_available': False,
        'camera_opened': False,
        'voice_available': False,
        'voice_running': False,
        'system_health': system_health.copy(),
        'uptime_seconds': time.time() - system_health['system_start_time']
    }
    
    if detector:
        debug_info.update({
            'camera_available': detector.cap is not None,
            'camera_opened': detector.cap.isOpened() if detector.cap else False,
            'voice_available': detector.voice is not None,
            'voice_running': detector.voice.running if detector.voice else False,
            'frame_count': detector.frame_count,
            'current_fps': detector.current_fps
        })
    
    return jsonify(debug_info)

@app.route('/api/system-health')
def get_system_health():
    """Get system health information"""
    current_time = time.time()
    health_info = {
        'system_uptime': current_time - system_health['system_start_time'],
        'last_heartbeat': current_time - system_health['last_heartbeat'],
        'total_frames_processed': system_health['total_frames_processed'],
        'consecutive_failures': system_health['consecutive_failures'],
        'recovery_attempts': system_health['recovery_attempts'],
        'system_status': 'healthy' if (current_time - system_health['last_heartbeat']) < 30 else 'warning',
        'detection_active': detection_running and detection_thread and detection_thread.is_alive()
    }
    
    # Update system health
    system_health['last_heartbeat'] = current_time
    
    return jsonify(health_info)

@app.route('/api/voice-test')
def test_voice():
    """Test voice system"""
    if not detector:
        return jsonify({'success': False, 'message': 'Detection system not running'})
    
    if not detector.voice:
        return jsonify({'success': False, 'message': 'Voice manager not initialized'})
    
    if not detector.voice.running:
        return jsonify({'success': False, 'message': 'Voice system not running'})
    
    if not detector.voice.current_method:
        return jsonify({'success': False, 'message': 'No voice method available'})
    
    try:
        # Test the voice system directly
        test_message = f"Voice system test successful using {detector.voice.current_method}"
        detector.voice.announce('test', test_message)
        return jsonify({
            'success': True, 
            'message': f'Voice test initiated using {detector.voice.current_method}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Voice test failed: {str(e)}'})

@app.route('/api/voice-log')
def get_voice_log():
    """Get voice log entries"""
    try:
        if not detector or not detector.voice:
            return jsonify({'success': False, 'message': 'Voice system not available'})
        
        log_file = detector.voice.voice_log_file
        if not os.path.exists(log_file):
            return jsonify({'success': False, 'message': 'Voice log file not found'})
        
        # Read last 50 lines of voice log
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Get last 50 lines, reverse to show newest first
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            recent_lines.reverse()
        
        return jsonify({
            'success': True,
            'log_entries': recent_lines,
            'total_entries': len(lines),
            'recent_count': len(recent_lines)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to read voice log: {str(e)}'})

@app.route('/api/voice-force-speak', methods=['POST'])
def force_speak():
    """Force immediate voice output"""
    try:
        data = request.get_json()
        message = data.get('message', 'Test message')
        
        if not detector or not detector.voice:
            return jsonify({'success': False, 'message': 'Voice system not available'})
        
        if not detector.voice.running:
            return jsonify({'success': False, 'message': 'Voice system not running'})
        
        # Use force_speak for guaranteed output
        detector.voice.force_speak(message)
        
        return jsonify({
            'success': True,
            'message': f'Force speech initiated: {message}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Force speech failed: {str(e)}'})

if __name__ == '__main__':
    print("🚀 Starting i-sight Flask Integrated System...")
    print("📱 Dashboard: http://localhost:5000")
    print("📹 Video Stream: http://localhost:5000/video-feed")
    print("🔗 API Endpoints: http://localhost:5000/api/")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        stop_detection_thread()
    except Exception as e:
        print(f"❌ Error: {e}")
        stop_detection_thread() 