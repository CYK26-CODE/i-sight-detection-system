#!/usr/bin/env python3
"""
Test vehicle detection with direction and distance announcements
"""

import sys
import os
import time
import cv2
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from i_sight_detector import EnhancedUnifiedDetector

def create_test_frame_with_vehicle(direction="right", vehicle_type="car", distance_simulation=300):
    """Create a test frame with a simulated vehicle"""
    # Create a blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate vehicle size based on distance and type
    vehicle_sizes = {
        'bicycle': {'base_width': 40, 'base_height': 60},
        'car': {'base_width': 80, 'base_height': 50},
        'truck': {'base_width': 100, 'base_height': 70}
    }
    
    base_size = vehicle_sizes.get(vehicle_type, vehicle_sizes['car'])
    
    # Scale based on distance (closer = larger)
    scale_factor = 500 / max(distance_simulation, 100)
    vehicle_width = int(base_size['base_width'] * scale_factor)
    vehicle_height = int(base_size['base_height'] * scale_factor)
    
    # Position based on direction
    frame_height, frame_width = frame.shape[:2]
    
    if direction == "left":
        x = 50  # Left side
        y = frame_height // 2 - vehicle_height // 2
    elif direction == "right":
        x = frame_width - vehicle_width - 50  # Right side
        y = frame_height // 2 - vehicle_height // 2
    else:  # front
        x = frame_width // 2 - vehicle_width // 2  # Center
        y = frame_height // 2 - vehicle_height // 2
    
    # Draw vehicle rectangle
    if vehicle_type == "bicycle":
        color = (0, 255, 255)  # Yellow for bicycle
    elif vehicle_type == "truck":
        color = (0, 0, 255)    # Red for truck
    else:
        color = (255, 0, 0)    # Blue for car
    
    cv2.rectangle(frame, (x, y), (x + vehicle_width, y + vehicle_height), color, -1)
    
    # Add some details to make it look more like a vehicle
    # Wheels
    wheel_radius = max(5, vehicle_height // 8)
    wheel_y = y + vehicle_height - wheel_radius
    cv2.circle(frame, (x + wheel_radius * 2, wheel_y), wheel_radius, (255, 255, 255), -1)
    cv2.circle(frame, (x + vehicle_width - wheel_radius * 2, wheel_y), wheel_radius, (255, 255, 255), -1)
    
    # Windshield
    windshield_height = vehicle_height // 3
    cv2.rectangle(frame, (x + 5, y + 5), (x + vehicle_width - 5, y + windshield_height), (200, 200, 200), -1)
    
    return frame

def main():
    print("=" * 60)
    print("🚗 TESTING VEHICLE DETECTION WITH DIRECTION & DISTANCE")
    print("=" * 60)
    
    # Initialize detector
    print("\n1. Initializing i-sight detector...")
    detector = EnhancedUnifiedDetector()
    
    if not detector.voice.running:
        print("❌ Voice system failed to initialize!")
        return
    
    print(f"✅ Detector ready with voice method: {detector.voice.current_method}")
    
    # Test different vehicle scenarios
    test_scenarios = [
        {"direction": "left", "vehicle_type": "car", "distance": 200},
        {"direction": "right", "vehicle_type": "bicycle", "distance": 150},
        {"direction": "front", "vehicle_type": "truck", "distance": 400},
        {"direction": "right", "vehicle_type": "car", "distance": 100},
        {"direction": "left", "vehicle_type": "bicycle", "distance": 80}
    ]
    
    print("\n2. Testing vehicle detection scenarios...")
    
    for i, scenario in enumerate(test_scenarios, 1):
        direction = scenario["direction"]
        vehicle_type = scenario["vehicle_type"]
        distance = scenario["distance"]
        
        print(f"\n   Test {i}/5: {vehicle_type} approaching from {direction} at ~{distance}cm")
        
        # Create test frame with simulated vehicle
        frame = create_test_frame_with_vehicle(direction, vehicle_type, distance)
        
        # Setup zones if not already done
        if not detector.zones:
            detector.setup_detection_zones(frame.shape[1], frame.shape[0])
        
        # Process the frame (simulate vehicle detection)
        # Since we don't have YOLO, we'll manually create detection data
        simulated_detection = {
            'class': vehicle_type,
            'confidence': 0.85,
            'bbox': (100, 200, 200, 250),  # Example bbox
            'center': (150, 225),
            'distance': distance,
            'direction': direction,
            'size': (100, 50)
        }
        
        vehicle_detections = [simulated_detection]
        detector.current_vehicle_info = vehicle_detections
        detector.vehicle_count = 1
        
        print(f"   Simulated: {vehicle_type} from {direction}, distance: {distance}cm")
        
        # Test voice announcement with simulated data
        detector.announce_detections(
            0,  # No faces
            1,  # One vehicle 
            0,  # No traffic signs
            [],  # No face zones
            [],  # No face zone info
            vehicle_detections  # Vehicle info
        )
        
        # Wait for voice to complete
        time.sleep(4.0)
    
    print("\n3. Testing multiple vehicles scenario...")
    
    # Test multiple vehicles
    multiple_vehicles = [
        {
            'class': 'car',
            'confidence': 0.9,
            'bbox': (50, 200, 150, 250),
            'center': (100, 225),
            'distance': 250,
            'direction': 'left',
            'size': (100, 50)
        },
        {
            'class': 'bicycle',
            'confidence': 0.8,
            'bbox': (500, 200, 560, 260),
            'center': (530, 230),
            'distance': 120,
            'direction': 'right',
            'size': (60, 60)
        }
    ]
    
    detector.current_vehicle_info = multiple_vehicles
    detector.vehicle_count = 2
    
    print("   Multiple vehicles: car from left (250cm), bicycle from right (120cm)")
    
    detector.announce_detections(
        0,  # No faces
        2,  # Two vehicles
        0,  # No traffic signs
        [],  # No face zones
        [],  # No face zone info
        multiple_vehicles  # Vehicle info
    )
    
    time.sleep(5.0)
    
    print("\n4. Testing real camera feed...")
    
    # Brief real camera test with fallback vehicle detection
    if detector.cap and detector.cap.isOpened():
        print("   Testing with real camera for 15 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 15:
            ret, frame = detector.cap.read()
            if ret:
                if not detector.zones:
                    detector.setup_detection_zones(frame.shape[1], frame.shape[0])
                
                # Test vehicle detection
                frame, vehicle_detections = detector.detect_vehicles(frame)
                
                if vehicle_detections:
                    print(f"   Real vehicle detection: {len(vehicle_detections)} vehicle(s)")
                    for vehicle in vehicle_detections:
                        print(f"     - {vehicle['class']} from {vehicle['direction']}, {vehicle.get('distance', 'unknown')}cm")
                    
                    detector.announce_detections(
                        0, len(vehicle_detections), 0, [], [], vehicle_detections
                    )
                    time.sleep(3)
            
            time.sleep(0.2)
    else:
        print("   No camera available for real test")
    
    # Cleanup
    print("\n5. Shutting down...")
    detector.cleanup()
    
    print("\n" + "=" * 60)
    print("🎉 VEHICLE DETECTION TEST COMPLETED!")
    print("🔊 Check i_sight_voice_log.txt for detailed results")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
