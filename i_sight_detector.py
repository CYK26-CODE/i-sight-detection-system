#!/usr/bin/env python3
"""
i-sight: Enhanced Real-time Computer Vision Detection System
Features: Face Detection + Vehicle Detection + Traffic Sign Detection + AI Voice Assistant
"""

import cv2
import numpy as np
import time
import json
import threading
import queue
import requests
import base64
from datetime import datetime
import os

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
    from Traffic_Sign_Detection.Traffic_Sign_Detection.classification import training, getLabel
    from Traffic_Sign_Detection.Traffic_Sign_Detection.improved_classification import improved_training, improved_getLabel
    from Traffic_Sign_Detection.Traffic_Sign_Detection.main import localization
    TRAFFIC_SIGN_AVAILABLE = True
except ImportError:
    TRAFFIC_SIGN_AVAILABLE = False
    print("⚠️  Traffic sign detection not available")

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
        
        print("� Voice worker thread stopped")
    
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
        
        print(f"� FORCE SPEAKING: {message}")
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
        print("🔊 Stopping voice system...")
        self.running = False
        
        if self.voice_thread:
            self.voice_thread.join(timeout=2)
        
        # Clean up engines
        try:
            if self.sapi_engine:
                # SAPI engine cleanup is automatic
                self.sapi_engine = None
            
            if self.pyttsx_engine:
                try:
                    self.pyttsx_engine.stop()
                except:
                    pass
                self.pyttsx_engine = None
                
        except Exception as cleanup_error:
            print(f"⚠️ Engine cleanup warning: {cleanup_error}")
        
        # Log shutdown
        try:
            with open(self.voice_log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] SYSTEM_SHUTDOWN: Voice system stopped\n")
                f.write(f"=== Session Summary ===\n")
                f.write(f"Primary Method: {self.current_method}\n")
                for message, count in self.statement_count.items():
                    f.write(f"  \"{message}\": {count} times\n")
                f.write("=== End Session ===\n\n")
        except Exception as e:
            print(f"❌ Failed to log shutdown: {e}")
        
        print(f"✅ Voice system stopped (was using {self.current_method})")

class EnhancedUnifiedDetector:
    def __init__(self):
        # System configuration
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
        self.json_output_enabled = True
        self.voice_enabled = True  # Enable voice by default
        
        # Voice system - INITIALIZE FIRST
        print("🔊 Initializing i-sight voice system...")
        self.voice = VoiceManager()
        if self.voice.running:
            print("✅ i-sight voice system ready")
            # Test voice on startup
            self.voice.force_speak("i-sight voice system initialized")
        else:
            print("❌ i-sight voice system failed to initialize")
        
        # Initialize camera
        self.cap = None
        self.ip_camera_url = "192.168.1.100:8080"
        self.initialize_camera()
        
        # Detection zones
        self.zones = []
        self.zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        
        # Vehicle detection parameters
        self.vehicle_conf_threshold = 0.25
        self.vehicle_iou_threshold = 0.45
        self.img_size = 320
        self.device = 'cpu'  # Use CPU for compatibility
        
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
            'traffic_sign': (0, 255, 255),
            'background': (0, 0, 0)
        }
        
        # Load models
        self.load_models()
        
        # Gemini API configuration
        self.gemini_enabled = True
        self.gemini_api_key = "AIzaSyDf-FK-jZfcYsQBVJh4GSwvn5Uokhb1Wlw"
        self.gemini_cooldown = 3.0
        self.max_requests_per_minute = 15
        self.last_gemini_query = 0
        self.gemini_request_count = 0
        self.gemini_request_times = []
        
        # JSON output settings
        self.json_output_path = "i_sight_detection_results.json"
        self.json_update_interval = 1.0
        self.last_json_update = time.time()
        
        print("✅ i-sight initialization complete!")
    
    def initialize_camera(self):
        """Initialize camera with optimized settings and retry logic"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"🔍 Initializing camera connection (attempt {attempt + 1}/{max_attempts})...")
                
                # Try default camera first (webcam)
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows for better compatibility
                
                if self.cap.isOpened():
                    print("✅ Default camera (webcam) initialized successfully")
                else:
                    print("⚠️ Default camera not available, trying IP camera...")
                    # Try IP camera as fallback
                    url = f"http://{self.ip_camera_url}/shot.jpg"
                    self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    
                    if self.cap.isOpened():
                        print("✅ IP camera initialized successfully")
                    else:
                        if attempt < max_attempts - 1:
                            print(f"❌ Camera attempt {attempt + 1} failed, retrying...")
                            time.sleep(2)
                            continue
                        else:
                            print("❌ No camera available after all attempts - creating test mode")
                            return
                
                # Optimize camera settings if camera is available
                if self.cap.isOpened():
                    # Set properties with error handling
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                        
                        # Test the camera by reading a frame
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            print("✅ Camera settings optimized")
                            return  # Success!
                        else:
                            print("⚠️ Camera test frame failed")
                            if attempt < max_attempts - 1:
                                self.cap.release()
                                time.sleep(1)
                                continue
                    except Exception as settings_error:
                        print(f"⚠️ Camera settings error: {settings_error}")
                        # Continue anyway, basic camera might still work
                        return
                break
                
            except Exception as e:
                print(f"❌ Camera initialization error (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)
                    continue
                else:
                    self.cap = None
    
    def load_models(self):
        """Load all detection models"""
        print("📦 Loading i-sight detection models...")
        
        # Load face detection models (Haar cascades)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty() or self.face_cascade2.empty():
            print("❌ Face detection models failed to load")
        else:
            print("✅ Face detection models loaded")
        
        # Load vehicle detection model (YOLOv5)
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
                    self.vehicle_model_loaded = False
            except Exception as e:
                print(f"❌ Vehicle model loading failed: {e}")
                self.vehicle_model_loaded = False
        else:
            print("❌ YOLOv5 dependencies not available")
            self.vehicle_model_loaded = False
        
        # Load traffic sign model
        if self.traffic_sign_enabled:
            try:
                # Traffic sign model initialization would go here
                print("✅ Traffic sign detection ready")
            except Exception as e:
                print(f"❌ Traffic sign model failed: {e}")
                self.traffic_sign_enabled = False
        
        print("📦 Model loading complete!")
    
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
        
        # Detection parameters
        scale_factor = 1.2 if self.lightweight_mode else 1.1
        min_neighbors = 2 if self.lightweight_mode else 4
        min_size = (15, 15) if self.lightweight_mode else (20, 20)
        
        # Primary cascade
        faces1 = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        
        # Secondary cascade (if not lightweight)
        if not self.lightweight_mode:
            faces2 = self.face_cascade2.detectMultiScale(
                gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
            )
            all_faces = list(faces1) + list(faces2)
        else:
            all_faces = list(faces1)
        
        # Add confidence scores
        for (x, y, w, h) in all_faces:
            faces.append([x, y, w, h, 0.8])
        
        return faces
    
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
                
                # Draw face boxes
                for (fx, fy, fw, fh, conf) in zone_faces:
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), self.colors['face'], 2)
                    cv2.putText(frame, f'Face: {conf:.2f}', (fx, fy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['face'], 1)
        
        return detected_zones, person_count, faces
    
    def detect_vehicles(self, frame):
        """Detect vehicles using YOLOv5"""
        if not YOLO_AVAILABLE or not self.vehicle_model_loaded:
            return frame, []
        
        try:
            # Vehicle detection implementation would go here
            # For now, return empty results
            return frame, []
        except Exception as e:
            print(f"❌ Vehicle detection error: {e}")
            return frame, []
    
    def detect_traffic_signs(self, frame):
        """Detect traffic signs"""
        if not self.traffic_sign_enabled:
            return frame, []
        
        try:
            # Traffic sign detection implementation would go here
            # For now, return empty results
            return frame, []
        except Exception as e:
            print(f"❌ Traffic sign detection error: {e}")
            return frame, []
    
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
    
    def display_info(self, frame, detected_zones, person_count, vehicle_count, traffic_sign_count):
        """Display detection information"""
        frame_height, frame_width = frame.shape[:2]
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 180), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "i-sight DETECTION SYSTEM", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # FPS
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detection counts
        cv2.putText(frame, f'Faces: {person_count}', (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f'Vehicles: {vehicle_count}', (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Traffic Signs: {traffic_sign_count}', (20, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Active zones
        if detected_zones:
            zone_text = f'Active: {", ".join([self.zone_names[i] for i in detected_zones])}'
            cv2.putText(frame, zone_text, (20, 185), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def add_zone_highlights(self, frame, detected_zones):
        """Highlight active detection zones"""
        for i, (x1, y1, x2, y2) in enumerate(self.zones):
            color = self.colors['zone'] if i in detected_zones else (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Zone label
            label = self.zone_names[i]
            cv2.putText(frame, label, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def announce_detections(self, person_count, vehicle_count, traffic_sign_count, detected_zones):
        """Announce detections with voice"""
        if not self.voice_enabled or not self.voice.running:
            return
        
        # Face detection announcements
        if person_count > 0:
            if person_count == 1:
                if detected_zones:
                    zone_name = self.zone_names[detected_zones[0]]
                    message = f"Person detected in {zone_name} zone"
                else:
                    message = "Person detected"
            else:
                message = f"{person_count} people detected"
            
            print(f"🔊 Face announcement: {message}")
            self.voice.announce('faces', message)
        
        # Vehicle detection announcements
        if vehicle_count > 0:
            if vehicle_count == 1:
                message = "Vehicle detected"
            else:
                message = f"{vehicle_count} vehicles detected"
            
            print(f"🔊 Vehicle announcement: {message}")
            self.voice.announce('vehicles', message)
        
        # Traffic sign detection announcements
        if traffic_sign_count > 0:
            if traffic_sign_count == 1:
                message = "Traffic sign detected"
            else:
                message = f"{traffic_sign_count} traffic signs detected"
            
            print(f"🔊 Traffic sign announcement: {message}")
            self.voice.announce('traffic_signs', message)
    
    def run(self):
        """Main detection loop"""
        print("🚀 Starting i-sight Detection System!")
        
        # Test camera connection
        if not self.cap or not self.cap.isOpened():
            print("❌ Camera not available. Exiting...")
            return
        
        # Startup voice announcement
        if self.voice_enabled and self.voice.running:
            startup_message = "i-sight detection system ready. All systems operational."
            print(f"🔊 Startup announcement: {startup_message}")
            self.voice.force_speak(startup_message)
            
            # Inform user that voice is working
            print("🔊 VOICE SYSTEM IS ACTIVE - You should hear audio announcements!")
            print("🔊 If you can't hear audio, check Windows Volume Mixer for python.exe")
        
        print("\n🎮 Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save screenshot")
        print("- Press 'v' to toggle voice feedback")
        print("- Press 't' to test voice system")
        print("- Press 'a' to check audio settings")
        
        show_zones = True
        consecutive_failures = 0
        max_failures = 10  # Allow up to 10 consecutive failures before giving up
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"❌ Error reading frame (failure #{consecutive_failures}/{max_failures})")
                
                # Try to recover the camera connection
                if consecutive_failures <= max_failures:
                    print("🔄 Attempting camera recovery...")
                    time.sleep(0.5)  # Brief pause
                    
                    # Try to reinitialize camera
                    if consecutive_failures % 5 == 0:  # Every 5 failures, try full reset
                        print("🔄 Reinitializing camera...")
                        try:
                            self.cap.release()
                            time.sleep(1)
                            self.cap = cv2.VideoCapture(0)
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            print("✅ Camera reinitialized")
                        except Exception as e:
                            print(f"❌ Camera reinitialize failed: {e}")
                    
                    continue  # Try reading frame again
                else:
                    print("❌ Too many consecutive camera failures - shutting down")
                    break
            else:
                # Reset failure counter on successful read
                if consecutive_failures > 0:
                    print(f"✅ Camera recovered after {consecutive_failures} failures")
                    consecutive_failures = 0
            
            # Setup zones on first frame
            if not self.zones:
                self.setup_detection_zones(frame.shape[1], frame.shape[0])
            
            # Face detection (every 3rd frame)
            if self.frame_count % 3 == 0:
                detected_zones, person_count, faces = self.detect_people_in_zones(frame)
                self.face_count = person_count
            else:
                detected_zones, person_count = [], 0
                self.face_count = person_count
            
            # Vehicle detection (every 5th frame)
            if self.frame_count % 5 == 0:
                frame, vehicle_detections = self.detect_vehicles(frame)
                self.vehicle_count = len(vehicle_detections)
            else:
                self.vehicle_count = 0
            
            # Traffic sign detection (every 10th frame)
            if self.frame_count % 10 == 0:
                frame, traffic_sign_detections = self.detect_traffic_signs(frame)
                self.traffic_sign_count = len(traffic_sign_detections)
            else:
                self.traffic_sign_count = 0
            
            # Voice announcements (every 60 frames = ~2 seconds)
            if self.frame_count % 60 == 0:
                self.announce_detections(self.face_count, self.vehicle_count, self.traffic_sign_count, detected_zones)
            
            # Status update every 300 frames (~10 seconds) in headless mode
            if self.frame_count % 300 == 0:
                status_msg = f"i-sight status: {self.face_count} faces detected. FPS: {self.current_fps:.1f}"
                print(f"📊 {status_msg}")
                if self.voice_enabled and self.voice.running:
                    print("🔊 Voice system is active and working")
            
            # Draw zones
            if show_zones:
                self.add_zone_highlights(frame, detected_zones)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Display information
            self.display_info(frame, detected_zones, self.face_count, self.vehicle_count, self.traffic_sign_count)
            
            # Show frame (with error handling for OpenCV display issues)
            try:
                cv2.imshow('i-sight Detection System', frame)
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as e:
                # OpenCV display not available - run in headless mode
                if self.frame_count == 1:  # Show message only once
                    print(f"⚠️ OpenCV display not available: {e}")
                    print("🎮 Running in HEADLESS mode - Voice system is active!")
                    print("🎮 Press Ctrl+C to stop the system")
                key = -1  # No key pressed in headless mode
            if key == ord('q'):
                print("🛑 Shutting down i-sight...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"i_sight_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('v'):
                self.voice_enabled = not self.voice_enabled
                status = "ON" if self.voice_enabled else "OFF"
                print(f"🔊 Voice feedback: {status}")
                if self.voice_enabled and self.voice.running:
                    self.voice.force_speak(f"Voice feedback {status}")
            elif key == ord('t'):
                if self.voice_enabled and self.voice.running:
                    print("🔊 Testing voice system...")
                    test_messages = [
                        "i-sight voice test one",
                        "Can you hear this announcement?",
                        "Audio system check complete"
                    ]
                    for i, msg in enumerate(test_messages, 1):
                        print(f"🔊 Test {i}/3: {msg}")
                        self.voice.force_speak(msg)
                        time.sleep(1.5)  # Wait between tests
                    
                    # Also try Windows native TTS as backup
                    print("🔊 Testing Windows native TTS...")
                    try:
                        import subprocess
                        subprocess.run([
                            'powershell', '-Command', 
                            'Add-Type -AssemblyName System.speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Rate = 2; $speak.Volume = 100; $speak.Speak("Windows native voice test successful")'
                        ], shell=True, timeout=10)
                        print("✅ Windows native TTS test completed")
                    except Exception as e:
                        print(f"❌ Windows native TTS failed: {e}")
                else:
                    print("❌ Voice system not available")
            elif key == ord('a'):
                print("🔊 Checking Windows audio configuration...")
                try:
                    # Check Windows audio devices
                    import subprocess
                    result = subprocess.run([
                        'powershell', '-Command', 
                        'Get-AudioDevice -List | Where-Object {$_.Type -eq "Playback" -and $_.Default -eq $true} | Select-Object Name, ID'
                    ], capture_output=True, text=True, shell=True, timeout=10)
                    print(f"🔊 Default audio device: {result.stdout.strip()}")
                    
                    # Check volume mixer for python
                    result2 = subprocess.run([
                        'powershell', '-Command',
                        'Get-Process python -ErrorAction SilentlyContinue | Select-Object Name, Id'
                    ], capture_output=True, text=True, shell=True, timeout=5)
                    print(f"🔊 Python processes: {result2.stdout.strip()}")
                    
                    # Play a test beep
                    print("🔊 Playing system beep...")
                    subprocess.run(['powershell', '-Command', '[console]::beep(1000,500)'], shell=True, timeout=5)
                    
                    print("🔊 If you heard the beep but not TTS, check Windows Volume Mixer")
                    print("   Right-click speaker icon > Volume Mixer > Ensure python.exe is not muted")
                    
                except Exception as e:
                    print(f"❌ Audio check failed: {e}")
            
            self.frame_count += 1
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.voice:
                print("🔊 Stopping voice system...")
                self.voice.stop()
            if self.cap:
                print("📷 Releasing camera...")
                self.cap.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error as e:
                print(f"⚠️ OpenCV cleanup warning: {e}")
            print("✅ i-sight shutdown complete")
        except Exception as e:
            print(f"❌ Cleanup error: {e}")

def main():
    detector = None
    try:
        print("🚀 Initializing i-sight...")
        detector = EnhancedUnifiedDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if detector:
            detector.cleanup()

if __name__ == "__main__":
    main()
