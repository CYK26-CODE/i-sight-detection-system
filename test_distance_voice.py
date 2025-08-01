#!/usr/bin/env python3
"""
Test distance-aware voice messages
"""

import sys
import os
import time
import cv2
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from i_sight_detector import EnhancedUnifiedDetector

def create_test_frame_with_face(distance_simulation=100):
    """Create a test frame with a simulated face at different distances"""
    # Create a blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate face size based on distance
    # Closer objects appear larger
    if distance_simulation < 50:
        face_size = 120  # Very close - large face
    elif distance_simulation < 100:
        face_size = 80   # Close - medium face
    elif distance_simulation < 200:
        face_size = 50   # Medium distance - smaller face
    else:
        face_size = 30   # Far - very small face
    
    # Draw a simple face rectangle in the center-right zone
    x = 400  # Right side of frame
    y = 200  # Center vertically
    
    # Draw face rectangle
    cv2.rectangle(frame, (x, y), (x + face_size, y + face_size), (255, 255, 255), -1)
    
    # Draw simple face features
    eye_size = max(5, face_size // 10)
    cv2.circle(frame, (x + face_size//4, y + face_size//3), eye_size, (0, 0, 0), -1)  # Left eye
    cv2.circle(frame, (x + 3*face_size//4, y + face_size//3), eye_size, (0, 0, 0), -1)  # Right eye
    cv2.rectangle(frame, (x + face_size//3, y + 2*face_size//3), (x + 2*face_size//3, y + 3*face_size//4), (0, 0, 0), -1)  # Mouth
    
    return frame

def main():
    print("=" * 60)
    print("🔊 TESTING DISTANCE-AWARE VOICE MESSAGES")
    print("=" * 60)
    
    # Initialize detector
    print("\n1. Initializing i-sight detector...")
    detector = EnhancedUnifiedDetector()
    
    if not detector.voice.running:
        print("❌ Voice system failed to initialize!")
        return
    
    print(f"✅ Detector ready with voice method: {detector.voice.current_method}")
    
    # Test different distances
    test_distances = [30, 60, 120, 250, 400]  # cm
    
    print("\n2. Testing distance estimation and voice announcements...")
    
    for i, distance in enumerate(test_distances, 1):
        print(f"\n   Test {i}/5: Simulating face at {distance}cm")
        
        # Create test frame with simulated face
        frame = create_test_frame_with_face(distance)
        
        # Setup zones if not already done
        if not detector.zones:
            detector.setup_detection_zones(frame.shape[1], frame.shape[0])
        
        # Process the frame
        detected_zones, person_count, faces, zone_info = detector.detect_people_in_zones(frame)
        
        print(f"   Detected: {person_count} person(s) in zones: {detected_zones}")
        
        if zone_info:
            for zone_data in zone_info:
                zone_name = zone_data['zone_name']
                for face in zone_data['faces']:
                    if len(face) >= 6 and face[5]:
                        estimated_distance = face[5]
                        print(f"   Zone: {zone_name}, Estimated distance: {estimated_distance}cm")
        
        # Test voice announcement
        detector.announce_detections(person_count, 0, 0, detected_zones, zone_info)
        
        # Wait for voice to complete
        time.sleep(4.0)
    
    print("\n3. Testing real camera feed with distance...")
    
    # Brief real camera test
    if detector.cap and detector.cap.isOpened():
        print("   Testing with real camera for 10 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 10:
            ret, frame = detector.cap.read()
            if ret:
                if not detector.zones:
                    detector.setup_detection_zones(frame.shape[1], frame.shape[0])
                
                detected_zones, person_count, faces, zone_info = detector.detect_people_in_zones(frame)
                
                if person_count > 0:
                    print(f"   Real detection: {person_count} person(s)")
                    detector.announce_detections(person_count, 0, 0, detected_zones, zone_info)
                    break
            
            time.sleep(0.1)
    else:
        print("   No camera available for real test")
    
    # Cleanup
    print("\n4. Shutting down...")
    detector.cleanup()
    
    print("\n" + "=" * 60)
    print("🎉 DISTANCE-AWARE VOICE TEST COMPLETED!")
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
