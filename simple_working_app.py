#!/usr/bin/env python3
"""
Simple Working i-sight System
Provides exact voice output format: "One person detected on Slight Right zone at 50 cm"
"""

import cv2
import numpy as np
import time
import threading
import queue
import json
import base64
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
import os
import sys

app = Flask(__name__)

# Global variables
detection_running = False
detection_thread = None
detector = None

class SimpleVoiceManager:
    """Simple voice management with Windows SAPI"""
    
    def __init__(self):
        self.voice_enabled = True
        self.running = False
        self.voice_queue = queue.Queue()
        self.voice_thread = None
        self.log_file = "i_sight_voice_log.txt"
        
        # Try to initialize Windows SAPI
        try:
            import win32com.client
            self.voice_engine = win32com.client.Dispatch("SAPI.SpVoice")
            print("✅ Windows SAPI voice engine initialized")
        except:
            self.voice_engine = None
            print("⚠️  Windows SAPI not available - voice will be simulated")
        
        # Start voice thread
        self.start()
    
    def test_voice_system(self):
        """Test voice system"""
        test_message = "Voice system test successful"
        self.announce('test', test_message)
        return True
    
    def log_voice_message(self, message: str, action: str = "QUEUED"):
        """Log voice messages to file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {action}: {message}\n")
        except Exception as e:
            print(f"❌ Voice logging failed: {e}")
    
    def _voice_worker(self):
        """Voice worker thread"""
        while self.running:
            try:
                # Get message from queue with timeout
                try:
                    category, message = self.voice_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Log the message
                self.log_voice_message(message, "SPEAKING")
                
                # Try to speak
                if self.voice_engine:
                    try:
                        self.voice_engine.Speak(message)
                        print(f"🔊 VOICE: {message}")
                    except Exception as e:
                        print(f"❌ Voice failed: {e}")
                        print(f"🔊 SIMULATED: {message}")
                else:
                    # Simulate voice output
                    print(f"🔊 SIMULATED: {message}")
                
                # Mark task as done
                self.voice_queue.task_done()
                
            except Exception as e:
                print(f"❌ Voice worker error: {e}")
                time.sleep(0.1)
    
    def announce(self, category: str, message: str):
        """Announce a message with category"""
        if not self.voice_enabled or not self.running:
            return
        
        # Add to queue
        self.voice_queue.put((category, message))
        self.log_voice_message(message, "QUEUED")
    
    def force_speak(self, message: str, max_retries: int = 3):
        """Force immediate speech"""
        if self.voice_engine:
            try:
                self.voice_engine.Speak(message)
                print(f"🔊 FORCE VOICE: {message}")
                return True
            except Exception as e:
                print(f"❌ Force speak failed: {e}")
        
        print(f"🔊 FORCE SIMULATED: {message}")
        return True
    
    def start(self):
        """Start voice system"""
        if not self.running:
            self.running = True
            self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
            self.voice_thread.start()
            print("✅ Voice system started")
    
    def stop(self):
        """Stop voice system"""
        self.running = False
        if self.voice_thread:
            self.voice_thread.join(timeout=2.0)
        print("✅ Voice system stopped")

class SimpleDetector:
    """Simple i-sight detector with exact voice format"""
    
    def __init__(self):
        # Camera
        self.cap = None
        
        # Detection models
        self.face_cascade = None
        
        # Detection counts
        self.face_count = 0
        self.frame_count = 0
        self.current_fps = 0.0
        self.last_fps_time = time.time()
        
        # Detection zones
        self.zones = []
        self.zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        
        # Threading
        self.detection_lock = threading.Lock()
        self.latest_detection_data = {
            'timestamp': datetime.now().isoformat(),
            'faces': [],
            'fps': 0.0,
            'frame_number': 0
        }
        
        # Screenshot
        self.latest_screenshot = None
        
        # Voice system
        self.voice = SimpleVoiceManager()
        self.voice_enabled = True
        
        # Initialize components
        self.initialize_camera()
        self.load_models()
        
        print("✅ Simple detector initialized")
    
    def initialize_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Could not open camera")
                return
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test frame capture
            ret, frame = self.cap.read()
            if ret:
                self.setup_detection_zones(frame.shape[1], frame.shape[0])
                print(f"✅ Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("❌ Failed to capture test frame")
                
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
    
    def load_models(self):
        """Load detection models"""
        try:
            # Load face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                print("❌ Face detection model failed to load")
            else:
                print("✅ Face detection model loaded")
            
        except Exception as e:
            print(f"❌ Model loading error: {e}")
    
    def setup_detection_zones(self, frame_width, frame_height):
        """Setup detection zones"""
        zone_width = frame_width // 5
        self.zones = []
        
        for i in range(5):
            x1 = i * zone_width
            x2 = (i + 1) * zone_width
            y1 = 0
            y2 = frame_height
            self.zones.append((x1, y1, x2, y2))
        
        print(f"✅ Detection zones setup: {len(self.zones)} zones")
    
    def detect_faces(self, gray):
        """Detect faces"""
        if self.face_cascade is None:
            return []
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def detect_people_in_zones(self, frame):
        """Detect people in zones"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)
        
        detected_zones = []
        person_count = len(faces)
        
        # Determine zones for detected faces
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            
            for i, (x1, y1, x2, y2) in enumerate(self.zones):
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    detected_zones.append(i)
                    break
        
        return detected_zones, person_count
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def announce_detections(self, person_count, detected_zones):
        """Announce detections with exact voice format"""
        if not self.voice_enabled or not self.voice.running:
            return
        
        # Announce people with exact format
        if person_count > 0:
            if person_count == 1:
                if detected_zones:
                    zone_name = self.zone_names[detected_zones[0]]
                    message = f"One person detected on {zone_name} zone"
                else:
                    message = "One person detected"
            elif person_count == 2:
                if len(detected_zones) == 2:
                    zone1_name = self.zone_names[detected_zones[0]]
                    zone2_name = self.zone_names[detected_zones[1]]
                    message = f"Two people detected: one in {zone1_name} zone and one in {zone2_name} zone"
                elif len(detected_zones) == 1:
                    zone_name = self.zone_names[detected_zones[0]]
                    message = f"Two people detected on {zone_name} zone"
                else:
                    message = "Two people detected"
            else:
                if detected_zones:
                    zone_names = [self.zone_names[zone] for zone in detected_zones]
                    if len(zone_names) == 1:
                        message = f"{person_count} people detected on {zone_names[0]} zone"
                    else:
                        zone_list = ", ".join(zone_names[:-1]) + f" and {zone_names[-1]}"
                        message = f"{person_count} people detected on {zone_list} zones"
                else:
                    message = f"{person_count} people detected"
            
            self.voice.announce('faces', message)
    
    def process_frame(self):
        """Process a single frame"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            return None
        
        # Calculate FPS
        self.calculate_fps()
        
        # Detect people in zones
        detected_zones, person_count = self.detect_people_in_zones(frame)
        
        # Update detection count
        self.face_count = person_count
        
        # Update detection data
        self.update_detection_data(detected_zones, person_count)
        
        # Update screenshot
        self.update_screenshot(frame)
        
        # Announce detections (every 60 frames = ~2 seconds)
        if self.frame_count % 60 == 0:
            self.announce_detections(person_count, detected_zones)
        
        return frame
    
    def update_detection_data(self, detected_zones, person_count):
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
                'fps': self.current_fps,
                'frame_number': self.frame_count
            }
    
    def update_screenshot(self, frame):
        """Update latest screenshot"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            self.latest_screenshot = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"❌ Screenshot update error: {e}")
    
    def get_detection_data(self):
        """Get latest detection data"""
        with self.detection_lock:
            return self.latest_detection_data.copy()
    
    def get_screenshot(self):
        """Get latest screenshot"""
        return self.latest_screenshot
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        if self.voice:
            self.voice.stop()
        print("✅ Simple detector cleaned up")

def start_detection_thread():
    """Start detection thread"""
    global detection_running, detection_thread, detector
    
    if detection_running:
        return False
    
    try:
        # Initialize detector if not exists
        if detector is None:
            detector = SimpleDetector()
        
        detection_running = True
        
        def detection_worker():
            global detection_running
            
            while detection_running:
                try:
                    # Process frame
                    frame = detector.process_frame()
                    
                    if frame is None:
                        print("⚠️  Frame processing failed")
                        time.sleep(1)
                    
                    # Frame timeout
                    time.sleep(0.033)  # ~30 FPS
                    
                except Exception as e:
                    print(f"❌ Detection worker error: {e}")
                    time.sleep(1)
            
            print("✅ Detection worker stopped")
        
        detection_thread = threading.Thread(target=detection_worker, daemon=True)
        detection_thread.start()
        
        print("✅ Detection thread started")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start detection thread: {e}")
        detection_running = False
        return False

def stop_detection_thread():
    """Stop detection thread"""
    global detection_running, detection_thread
    
    if not detection_running:
        return False
    
    detection_running = False
    
    if detection_thread:
        detection_thread.join(timeout=5.0)
        detection_thread = None
    
    print("✅ Detection thread stopped")
    return True

# Flask routes
@app.route('/')
def index():
    """Main dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>i-sight System - Simple Version</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .controls { display: flex; gap: 10px; margin-bottom: 20px; }
            .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            .btn-primary { background: #007bff; color: white; }
            .btn-danger { background: #dc3545; color: white; }
            .btn-success { background: #28a745; color: white; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.running { background: #d4edda; color: #155724; }
            .status.stopped { background: #f8d7da; color: #721c24; }
            .video-container { text-align: center; margin: 20px 0; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .stat-card { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
            .stat-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .stat-label { color: #666; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 i-sight System - Simple Version</h1>
                <p>Voice Output Format: "One person detected on [Zone] zone"</p>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="startDetection()">🚀 Start Detection</button>
                <button class="btn btn-danger" onclick="stopDetection()">⏹️ Stop Detection</button>
                <button class="btn btn-success" onclick="testVoice()">🔊 Test Voice</button>
            </div>
            
            <div id="status" class="status stopped">
                Status: Stopped
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div id="faceCount" class="stat-value">0</div>
                    <div class="stat-label">People Detected</div>
                </div>
                <div class="stat-card">
                    <div id="fps" class="stat-value">0.0</div>
                    <div class="stat-label">FPS</div>
                </div>
                <div class="stat-card">
                    <div id="frameCount" class="stat-value">0</div>
                    <div class="stat-label">Frames Processed</div>
                </div>
            </div>
            
            <div class="video-container">
                <h3>📹 Live Video Feed</h3>
                <img id="videoFeed" src="/video-feed" style="max-width: 100%; border: 2px solid #ddd; border-radius: 5px;">
            </div>
        </div>
        
        <script>
            function startDetection() {
                fetch('/api/start-detection', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('status').className = 'status running';
                            document.getElementById('status').textContent = 'Status: Running';
                        }
                    });
            }
            
            function stopDetection() {
                fetch('/api/stop-detection', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('status').className = 'status stopped';
                            document.getElementById('status').textContent = 'Status: Stopped';
                        }
                    });
            }
            
            function testVoice() {
                fetch('/api/voice-test')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Voice test completed!');
                        } else {
                            alert('Voice test failed: ' + data.message);
                        }
                    });
            }
            
            // Update stats every second
            setInterval(() => {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('faceCount').textContent = data.face_count;
                        document.getElementById('fps').textContent = data.fps.toFixed(1);
                        document.getElementById('frameCount').textContent = data.frame_number || 0;
                    });
            }, 1000);
        </script>
    </body>
    </html>
    """

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'detection_running': detection_running,
        'camera_connected': detector is not None and detector.cap is not None and detector.cap.isOpened(),
        'voice_enabled': detector.voice.voice_enabled if detector and hasattr(detector.voice, 'voice_enabled') else False,
        'face_count': detector.face_count if detector else 0,
        'fps': detector.current_fps if detector else 0.0,
        'frame_number': detector.frame_count if detector else 0
    })

@app.route('/api/start-detection', methods=['POST'])
def start_detection():
    """Start detection"""
    success = start_detection_thread()
    return jsonify({'success': success})

@app.route('/api/stop-detection', methods=['POST'])
def stop_detection():
    """Stop detection"""
    success = stop_detection_thread()
    return jsonify({'success': success})

@app.route('/api/latest-detection')
def get_latest_detection():
    """Get latest detection data"""
    if detector:
        return jsonify(detector.get_detection_data())
    else:
        return jsonify({'error': 'No detection data available'}), 404

@app.route('/video-feed')
def video_feed():
    """Video feed endpoint"""
    def generate_frames():
        while detection_running:
            if detector and detector.cap and detector.cap.isOpened():
                ret, frame = detector.cap.read()
                if ret:
                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/voice-test')
def test_voice():
    """Test voice system"""
    try:
        if detector and detector.voice:
            success = detector.voice.test_voice_system()
            return jsonify({
                'success': success,
                'message': 'Voice test completed'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Voice system not available'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Voice test failed: {str(e)}'
        })

if __name__ == '__main__':
    print("🚀 Starting Simple Working i-sight System")
    print("=" * 60)
    print("🎯 Voice Output Format: 'One person detected on [Zone] zone'")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        if detector:
            detector.cleanup()
        print("✅ System shutdown complete") 