#!/usr/bin/env python3
"""
Backend API Server for Computer Vision Detection System
Handles detection control and provides REST API endpoints
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import signal
import os
import time
import threading
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

class DetectionController:
    def __init__(self):
        self.detection_process = None
        self.detection_running = False
        self.start_time = None
        self.stats = {
            'total_faces': 0,
            'total_vehicles': 0,
            'total_traffic_signs': 0,
            'uptime': 0
        }
    
    def start_detection(self, config=None):
        """Start the detection process"""
        if self.detection_running:
            return False, "Detection already running"
        
        try:
            # Start the detection process
            self.detection_process = subprocess.Popen(
                ['python', 'unified_detector_with_traffic_signs.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.detection_running = True
            self.start_time = time.time()
            
            # Start monitoring thread
            threading.Thread(target=self._monitor_process, daemon=True).start()
            
            return True, "Detection started successfully"
            
        except Exception as e:
            return False, f"Failed to start detection: {str(e)}"
    
    def stop_detection(self):
        """Stop the detection process"""
        if not self.detection_running:
            return False, "Detection not running"
        
        try:
            if self.detection_process:
                # Send SIGTERM first
                self.detection_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.detection_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    self.detection_process.kill()
                    self.detection_process.wait()
                
                self.detection_process = None
            
            self.detection_running = False
            return True, "Detection stopped successfully"
            
        except Exception as e:
            return False, f"Failed to stop detection: {str(e)}"
    
    def _monitor_process(self):
        """Monitor the detection process"""
        while self.detection_running and self.detection_process:
            if self.detection_process.poll() is not None:
                # Process has ended
                self.detection_running = False
                self.detection_process = None
                break
            time.sleep(1)
    
    def get_status(self):
        """Get current system status"""
        return {
            'detection_running': self.detection_running,
            'camera_connected': True,  # Assume camera is connected if detection is running
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'stats': self.stats
        }
    
    def get_stats(self):
        """Get detection statistics"""
        uptime = time.time() - self.start_time if self.start_time else 0
        return {
            'total_faces': self.stats['total_faces'],
            'total_vehicles': self.stats['total_vehicles'],
            'total_traffic_signs': self.stats['total_traffic_signs'],
            'uptime': uptime,
            'uptime_formatted': f"{int(uptime//60)}m {int(uptime%60)}s"
        }

# Global detection controller
detection_controller = DetectionController()

@app.route('/status')
def get_status():
    """Get system status"""
    return jsonify(detection_controller.get_status())

@app.route('/start-detection', methods=['POST'])
def start_detection():
    """Start detection system"""
    config = request.get_json() if request.is_json else None
    success, message = detection_controller.start_detection(config)
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/stop-detection', methods=['POST'])
def stop_detection():
    """Stop detection system"""
    success, message = detection_controller.stop_detection()
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    return jsonify(detection_controller.get_stats())

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Backend API Server...")
    print("📡 API: http://localhost:8000")
    print("🔗 Endpoints:")
    print("   - GET  /status")
    print("   - POST /start-detection")
    print("   - POST /stop-detection")
    print("   - GET  /stats")
    print("   - GET  /health")
    
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)