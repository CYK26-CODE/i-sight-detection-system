#!/usr/bin/env python3
"""
Flask Frontend for Computer Vision Detection System
Provides real-time video streaming and detection visualization using Flask
"""

from flask import Flask, render_template, jsonify, request, Response
import requests
import json
import time
import base64
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import threading
import queue

# Configuration
BACKEND_URL = "http://localhost:8000"

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

class DetectionManager:
    def __init__(self):
        self.detection_running = False
        self.latest_frame = None
        self.latest_detection_data = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.stats = {
            'total_faces': 0,
            'total_vehicles': 0,
            'total_traffic_signs': 0,
            'current_fps': 0.0
        }
    
    def get_system_status(self):
        """Get system status from backend"""
        try:
            response = requests.get(f"{BACKEND_URL}/status", timeout=2)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}
    
    def get_latest_detection_data(self):
        """Get the latest detection data from backend"""
        try:
            response = requests.get(f"{BACKEND_URL}/latest-detection", timeout=2)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'timestamp' in data:
                    self.latest_detection_data = data
                    return data
        except:
            pass
        return None
    
    def get_screenshot(self):
        """Get screenshot from backend"""
        try:
            response = requests.get(f"{BACKEND_URL}/screenshot", timeout=2)
            if response.status_code == 200:
                data = response.json()
                image_data = base64.b64decode(data['image'])
                return image_data
        except:
            pass
        return None
    
    def start_detection(self, config=None):
        """Start detection on backend"""
        if config is None:
            config = {
                "enable_faces": True,
                "enable_vehicles": True,
                "enable_traffic_signs": True,
                "confidence_threshold": 0.3,
                "vehicle_confidence_threshold": 0.25,
                "process_every_n_frames": 2
            }
        
        try:
            response = requests.post(f"{BACKEND_URL}/start-detection", json=config)
            if response.status_code == 200:
                self.detection_running = True
                return True, "Detection started successfully"
            else:
                return False, f"Failed to start detection: {response.text}"
        except Exception as e:
            return False, f"Error connecting to backend: {str(e)}"
    
    def stop_detection(self):
        """Stop detection on backend"""
        try:
            response = requests.post(f"{BACKEND_URL}/stop-detection")
            if response.status_code == 200:
                self.detection_running = False
                return True, "Detection stopped successfully"
            else:
                return False, f"Failed to stop detection: {response.text}"
        except Exception as e:
            return False, f"Error connecting to backend: {str(e)}"
    
    def overlay_detections(self, image_data, detection_data):
        """Overlay detection results on image"""
        if not detection_data:
            return image_data
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_data))
            draw = ImageDraw.Draw(image)
            width, height = image.size
            
            # Font for text (fallback to default if Arial not available)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                small_font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw face detection zones
            faces = detection_data.get('faces', [])
            for face_info in faces:
                zone = face_info.get('zone', 0)
                zone_name = face_info.get('zone_name', f'Zone {zone}')
                count = face_info.get('count', 0)
                
                # Calculate zone position
                zone_width = width // 5
                x1 = zone * zone_width
                x2 = (zone + 1) * zone_width
                y1 = int(height * 0.2)
                y2 = int(height * 0.8)
                
                # Draw zone rectangle
                draw.rectangle([x1, y1, x2, y2], outline='magenta', width=3)
                draw.text((x1 + 5, y1 + 5), f"{zone_name}: {count}", fill='magenta', font=font)
            
            # Draw vehicle detections
            vehicles = detection_data.get('vehicles', [])
            for vehicle in vehicles:
                bbox = vehicle.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    confidence = vehicle.get('confidence', 0)
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
                    
                    # Draw label with background
                    label = f"Vehicle: {confidence:.2f}"
                    bbox_coords = draw.textbbox((x1, y1-25), label, font=font)
                    draw.rectangle([bbox_coords[0]-2, bbox_coords[1]-2, bbox_coords[2]+2, bbox_coords[3]+2], fill='green')
                    draw.text((x1, y1-25), label, fill='white', font=font)
            
            # Draw traffic signs
            traffic_signs = detection_data.get('traffic_signs', [])
            for i, sign in enumerate(traffic_signs):
                sign_type = sign.get('sign_type', 'UNKNOWN')
                confidence = sign.get('confidence', 0)
                
                # Draw text at top of frame
                y_pos = 30 + i * 25
                label = f"Traffic Sign: {sign_type} ({confidence:.2f})"
                bbox_coords = draw.textbbox((10, y_pos), label, font=font)
                draw.rectangle([bbox_coords[0]-2, bbox_coords[1]-2, bbox_coords[2]+2, bbox_coords[3]+2], fill='yellow')
                draw.text((10, y_pos), label, fill='black', font=font)
            
            # Draw FPS and frame info
            fps = detection_data.get('fps', 0)
            frame_num = detection_data.get('frame_number', 0)
            processing_time = detection_data.get('processing_time', 0)
            
            info_text = f"FPS: {fps:.1f} | Frame: {frame_num} | Time: {processing_time:.3f}s"
            bbox_coords = draw.textbbox((width-300, height-30), info_text, font=small_font)
            draw.rectangle([bbox_coords[0]-2, bbox_coords[1]-2, bbox_coords[2]+2, bbox_coords[3]+2], fill='black')
            draw.text((width-300, height-30), info_text, fill='white', font=small_font)
            
            # Convert back to bytes
            buffer = BytesIO()
            image.save(buffer, format='JPEG')
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error overlaying detections: {e}")
            return image_data

# Global detection manager
detection_manager = DetectionManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    backend_status = detection_manager.get_system_status()
    return jsonify({
        'detection_running': detection_manager.detection_running,
        'backend_status': backend_status,
        'stats': detection_manager.stats
    })

@app.route('/api/start-detection', methods=['POST'])
def start_detection():
    """Start detection system"""
    config = request.get_json() if request.is_json else None
    success, message = detection_manager.start_detection(config)
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/stop-detection', methods=['POST'])
def stop_detection():
    """Stop detection system"""
    success, message = detection_manager.stop_detection()
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/latest-detection')
def get_latest_detection():
    """Get latest detection data"""
    data = detection_manager.get_latest_detection_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'message': 'No detection data available'})

@app.route('/video-feed')
def video_feed():
    """Video streaming route"""
    def generate_frames():
        while True:
            try:
                # Get screenshot from backend
                image_data = detection_manager.get_screenshot()
                if image_data:
                    # Get latest detection data
                    detection_data = detection_manager.get_latest_detection_data()
                    
                    # Overlay detections if available
                    if detection_data:
                        image_data = detection_manager.overlay_detections(image_data, detection_data)
                    
                    # Yield frame in multipart format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + image_data + b'\r\n')
                else:
                    # If no frame available, wait and continue
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in video feed: {e}")
                time.sleep(0.1)
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/screenshot')
def get_screenshot():
    """Get a single screenshot"""
    image_data = detection_manager.get_screenshot()
    if image_data:
        # Get latest detection data
        detection_data = detection_manager.get_latest_detection_data()
        
        # Overlay detections if available
        if detection_data:
            image_data = detection_manager.overlay_detections(image_data, detection_data)
        
        return Response(image_data, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'No screenshot available'}), 404

@app.route('/api/stats')
def get_stats():
    """Get detection statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'error': 'Could not get stats from backend'}), 500
    except Exception as e:
        return jsonify({'error': f'Error connecting to backend: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask Computer Vision Detection Frontend...")
    print("📱 Dashboard: http://localhost:5000")
    print("📹 Video Stream: http://localhost:5000/video-feed")
    print("🔗 API Endpoints: http://localhost:5000/api/")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
