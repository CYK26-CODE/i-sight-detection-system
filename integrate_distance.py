#!/usr/bin/env python3
"""
Distance Estimation Integration Script
Adds distance estimation functionality to the i-sight Flask application
"""

import cv2
import numpy as np
import time
from typing import List, Optional

def add_distance_estimation_to_flask():
    """
    This function contains the code that needs to be added to i_sight_flask_integrated.py
    to enable distance estimation functionality.
    """
    
    # Add these imports at the top of the file (after existing imports):
    print("""
# Add these imports to i_sight_flask_integrated.py:

# Distance estimation imports
try:
    from distance_estimator import DistanceEstimator
    DISTANCE_ESTIMATION_AVAILABLE = True
    print("✅ Distance estimation available")
except ImportError:
    DISTANCE_ESTIMATION_AVAILABLE = False
    print("⚠️  Distance estimation not available")
""")

    # Add to ISightDetector.__init__ method:
    print("""
# Add to ISightDetector.__init__ method (after voice system initialization):

# Distance estimation system
self.distance_estimator = None
if self.distance_estimation_enabled:
    self.distance_estimator = DistanceEstimator()
    # Try to calibrate automatically
    if not self.distance_estimator.calibrate_from_image():
        print("📷 Automatic calibration failed - manual calibration required")

# Add to detection counts:
self.distance_detections = []  # Distance estimation results

# Add to model loading flags:
self.distance_estimation_enabled = DISTANCE_ESTIMATION_AVAILABLE

# Add to latest_detection_data:
'distance_detections': [],  # Distance estimation results
""")

    # Add distance estimation methods:
    print("""
# Add these methods to ISightDetector class:

def announce_distance_detections(self, distance_detections):
    \"\"\"Announce distance detection results with voice\"\"\"
    if not self.voice_enabled or not self.voice.running:
        return
    
    if distance_detections:
        # Group by type and distance category
        faces = [d for d in distance_detections if d['type'] == 'face']
        persons = [d for d in distance_detections if d['type'] == 'person']
        
        if faces:
            # Announce closest face
            closest_face = min(faces, key=lambda x: x['distance'])
            message = f"Face detected at {closest_face['distance']:.1f} centimeters, {closest_face['category']} range"
            self.voice.announce('distance_face', message)
        
        if persons:
            # Announce closest person
            closest_person = min(persons, key=lambda x: x['distance'])
            message = f"Person detected at {closest_person['distance']:.1f} centimeters, {closest_person['category']} range"
            self.voice.announce('distance_person', message)

def update_detection_data_with_distance(self, detected_zones, person_count, vehicle_detections, 
                                       traffic_sign_detections, yolo11n_detections, distance_detections, processing_time):
    \"\"\"Update detection data for web interface with distance information\"\"\"
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
            'objects': [
                {
                    'class_name': detection.get('class_name', 'unknown'),
                    'confidence': detection.get('confidence', 0),
                    'bbox': detection.get('bbox', []),
                    'zone': self.get_object_zone(detection.get('bbox', []))
                }
                for detection in yolo11n_detections
            ],
            'distance_detections': [
                {
                    'type': detection.get('type', 'unknown'),
                    'distance': detection.get('distance', 0),
                    'category': detection.get('category', 'Unknown'),
                    'confidence': detection.get('confidence', 0),
                    'bbox': detection.get('bbox', [])
                }
                for detection in distance_detections
            ],
            'fps': self.current_fps,
            'frame_number': self.frame_count,
            'processing_time': processing_time
        }
""")

    # Add to process_frame method:
    print("""
# Add to process_frame method (after YOLO11n detection):

# Distance estimation (every 5th frame for performance)
if self.frame_count % 5 == 0 and self.distance_estimator and self.distance_estimator.calibrated:
    try:
        # Process frame with distance estimation
        processed_frame, distance_detections = self.distance_estimator.process_frame_with_distance(frame)
        self.distance_detections = distance_detections
        
        # Update frame with distance annotations
        frame = processed_frame
        
        # Voice announcements for distance
        if distance_detections and self.frame_count % 120 == 0:  # Every 4 seconds
            self.announce_distance_detections(distance_detections)
            
    except Exception as e:
        print(f"❌ Distance estimation error: {e}")
        self.distance_detections = []
else:
    self.distance_detections = []

# Update the call to update_detection_data:
self.update_detection_data_with_distance(detected_zones, person_count, vehicle_detections, 
                                        traffic_sign_detections, yolo11n_detections, self.distance_detections, processing_time)
""")

    # Add API endpoints:
    print("""
# Add these API endpoints to the Flask app:

@app.route('/api/distance-calibrate', methods=['POST'])
def calibrate_distance():
    \"\"\"Calibrate distance estimation system\"\"\"
    try:
        data = request.get_json()
        target_distance = data.get('target_distance', 50.0)
        
        if not detector or not detector.distance_estimator:
            return jsonify({'success': False, 'message': 'Distance estimation not available'})
        
        # Start calibration
        success = detector.distance_estimator.calibrate_from_camera(target_distance)
        
        return jsonify({
            'success': success,
            'message': 'Calibration completed' if success else 'Calibration failed'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Calibration error: {str(e)}'})

@app.route('/api/distance-status')
def get_distance_status():
    \"\"\"Get distance estimation system status\"\"\"
    if not detector or not detector.distance_estimator:
        return jsonify({'available': False})
    
    status = detector.distance_estimator.get_calibration_status()
    status['available'] = True
    status['detections'] = len(detector.distance_detections)
    
    return jsonify(status)

@app.route('/api/distance-detections')
def get_distance_detections():
    \"\"\"Get current distance detection results\"\"\"
    if not detector:
        return jsonify({'detections': []})
    
    return jsonify({
        'detections': detector.distance_detections,
        'timestamp': datetime.now().isoformat()
    })
""")

    # Add to status endpoint:
    print("""
# Add to /api/status endpoint:

'distance_estimation_enabled': detector.distance_estimation_enabled if detector else False,
'distance_calibrated': detector.distance_estimator.calibrated if detector and detector.distance_estimator else False,
'distance_detections': len(detector.distance_detections) if detector else 0,
""")

    # Add to stats endpoint:
    print("""
# Add to /api/stats endpoint:

'total_distance_detections': len(detector.distance_detections),
'distance_estimation_status': 'calibrated' if detector.distance_estimator and detector.distance_estimator.calibrated else 'not_calibrated',
""")

def create_distance_test_script():
    """Create a simple test script for distance estimation"""
    
    test_script = '''#!/usr/bin/env python3
"""
Distance Estimation Test Script
Tests the distance estimation functionality
"""

import cv2
import numpy as np
import time

def test_distance_estimation():
    """Test distance estimation with camera"""
    
    # Constants
    KNOWN_DISTANCE = 50  # cm
    KNOWN_FACE_WIDTH = 15  # cm
    
    # Load face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("✅ Camera opened")
    print("📷 Position your face at 50cm and press 'C' to calibrate")
    
    focal_length = None
    calibrated = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if calibrated and focal_length:
                    # Calculate distance
                    distance = (KNOWN_FACE_WIDTH * focal_length) / w
                    cv2.putText(frame, f"{distance:.1f}cm", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"{w}x{h}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instructions
            if not calibrated:
                cv2.putText(frame, "Press 'C' to calibrate at 50cm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"Focal Length: {focal_length:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press 'Q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Distance Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and len(faces) > 0 and not calibrated:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                focal_length = (w * KNOWN_DISTANCE) / KNOWN_FACE_WIDTH
                calibrated = True
                print(f"✅ Calibrated! Focal length: {focal_length:.2f}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_distance_estimation()
'''
    
    with open('test_distance_quick.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_distance_quick.py")

if __name__ == "__main__":
    print("🔧 Distance Estimation Integration Guide")
    print("=" * 50)
    
    add_distance_estimation_to_flask()
    
    print("\n" + "=" * 50)
    print("📝 INSTRUCTIONS:")
    print("1. Copy the code snippets above into i_sight_flask_integrated.py")
    print("2. Make sure distance_estimator.py is in the same directory")
    print("3. Test the distance estimation with test_distance_quick.py")
    print("4. Run the Flask app to see distance estimation in action")
    
    create_distance_test_script()
    
    print("\n✅ Integration guide completed!")
    print("📁 Files created:")
    print("   - test_distance_quick.py (simple test script)")
    print("   - distance_estimator.py (main distance estimation module)") 