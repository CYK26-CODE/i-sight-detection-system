#!/usr/bin/env python3
"""
Test script for the enhanced Flask integrated i-sight system
Tests all new distance-aware and vehicle detection features
"""

import sys
import time
import requests
import threading
from datetime import datetime

def test_flask_api():
    """Test the Flask API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Flask i-sight Enhanced API")
    print("=" * 50)
    
    # Test endpoints
    endpoints = [
        ("/", "Main page"),
        ("/api/detection-data", "Detection data API"),
        ("/api/screenshot", "Screenshot API"),
        ("/api/voice-test", "Voice test API"),
        ("/api/stats", "Statistics API")
    ]
    
    for endpoint, description in endpoints:
        try:
            print(f"🔍 Testing {description}: {base_url}{endpoint}")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                print(f"✅ {description}: OK")
                
                # Check specific content for detection data
                if endpoint == "/api/detection-data":
                    data = response.json()
                    print(f"   - Timestamp: {data.get('timestamp', 'N/A')}")
                    print(f"   - Face count: {len(data.get('faces', []))}")
                    print(f"   - Vehicle count: {len(data.get('vehicles', []))}")
                    print(f"   - Traffic sign count: {len(data.get('traffic_signs', []))}")
                    print(f"   - Object count: {len(data.get('objects', []))}")
                    print(f"   - FPS: {data.get('fps', 0):.1f}")
                    
                    # Check for distance and direction info
                    if data.get('vehicles'):
                        for i, vehicle in enumerate(data['vehicles'][:3]):  # Show first 3
                            vehicle_class = vehicle.get('class', 'unknown')
                            direction = vehicle.get('direction', 'unknown')
                            distance = vehicle.get('distance', 'unknown')
                            print(f"   - Vehicle {i+1}: {vehicle_class} from {direction} at {distance}cm")
                    
                    if data.get('faces'):
                        for i, face in enumerate(data['faces'][:3]):  # Show first 3
                            zone_name = face.get('zone_name', 'unknown')
                            distance = face.get('distance', 'unknown')
                            print(f"   - Face {i+1}: in {zone_name} zone at {distance}cm")
                
            else:
                print(f"❌ {description}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {description}: Connection error - {e}")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")
        
        time.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "=" * 50)

def test_voice_announcements():
    """Test voice announcements through API"""
    base_url = "http://localhost:5000"
    
    print("🔊 Testing Enhanced Voice Announcements")
    print("=" * 50)
    
    # Test voice messages
    test_messages = [
        "Person detected in Center zone very close at 45 centimeters",
        "car approaching from left close at 2.5 meters",
        "bicycle approaching from right very close at 120 centimeters",
        "3 vehicles detected, 1 from left, 1 from right, 1 from front, closest close at 1.8 meters"
    ]
    
    for i, message in enumerate(test_messages, 1):
        try:
            print(f"🔊 Test {i}/4: {message}")
            response = requests.post(f"{base_url}/api/voice-announce", 
                                   json={'message': message}, 
                                   timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Voice test {i}: Sent successfully")
            else:
                print(f"❌ Voice test {i}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Voice test {i}: Error - {e}")
        
        time.sleep(3)  # Wait for voice to complete
    
    print("\n" + "=" * 50)

def monitor_detections(duration=30):
    """Monitor detection data for specified duration"""
    base_url = "http://localhost:5000"
    
    print(f"📊 Monitoring Enhanced Detections for {duration} seconds")
    print("=" * 50)
    
    start_time = time.time()
    detection_count = 0
    
    while time.time() - start_time < duration:
        try:
            response = requests.get(f"{base_url}/api/detection-data", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                detection_count += 1
                
                current_time = datetime.now().strftime('%H:%M:%S')
                faces = len(data.get('faces', []))
                vehicles = len(data.get('vehicles', []))
                traffic_signs = len(data.get('traffic_signs', []))
                objects = len(data.get('objects', []))
                fps = data.get('fps', 0)
                
                # Enhanced display with distance/direction info
                status = f"[{current_time}] Faces: {faces} | Vehicles: {vehicles} | Signs: {traffic_signs} | Objects: {objects} | FPS: {fps:.1f}"
                
                # Add distance info if available
                distance_info = []
                if data.get('vehicles'):
                    for vehicle in data['vehicles'][:2]:  # Show first 2
                        v_class = vehicle.get('class', 'vehicle')
                        v_dir = vehicle.get('direction', 'unknown')
                        v_dist = vehicle.get('distance', 'unknown')
                        distance_info.append(f"{v_class} from {v_dir} at {v_dist}cm")
                
                if data.get('faces'):
                    for face in data['faces'][:2]:  # Show first 2
                        f_zone = face.get('zone_name', 'unknown')
                        f_dist = face.get('distance', 'unknown')
                        distance_info.append(f"face in {f_zone} at {f_dist}cm")
                
                if distance_info:
                    status += f" | Details: {', '.join(distance_info)}"
                
                print(status)
                
                # Special notifications for interesting detections
                if vehicles > 0:
                    print(f"🚗 {vehicles} vehicle(s) detected with direction/distance info!")
                if faces > 0:
                    print(f"👤 {faces} person(s) detected with zone/distance info!")
                
            else:
                print(f"❌ API error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        
        time.sleep(2)  # Check every 2 seconds
    
    print(f"\n✅ Monitoring complete. Total API calls: {detection_count}")
    print("=" * 50)

def main():
    """Main test function"""
    print("🚀 Enhanced Flask i-sight System Test")
    print("=" * 70)
    print("Testing all new features:")
    print("- Distance-aware face detection with zone information")
    print("- Vehicle detection with direction (left/right/front) and distance")
    print("- Enhanced voice announcements with detailed information")
    print("- Flask API integration with enhanced data")
    print("=" * 70)
    
    # Test 1: Basic API functionality
    test_flask_api()
    
    # Test 2: Enhanced voice announcements
    test_voice_announcements()
    
    # Test 3: Monitor live detections
    print("🔄 Starting live detection monitoring...")
    print("👁️  Look at the camera to test face detection with distance")
    print("🚗 Move objects in front of camera to test vehicle detection")
    print("🔊 Listen for enhanced voice announcements")
    
    monitor_detections(30)
    
    print("\n🎉 All tests completed!")
    print("💡 If you heard detailed voice announcements like:")
    print("   - 'Person detected in Center zone very close at 45 centimeters'")
    print("   - 'car approaching from left close at 2.5 meters'")
    print("   Then the enhanced system is working perfectly!")

if __name__ == "__main__":
    print("⚠️  Make sure the Flask i-sight server is running first!")
    print("   Run: python i_sight_flask_integrated.py")
    print()
    
    try:
        # Quick connection test
        response = requests.get("http://localhost:5000", timeout=3)
        print("✅ Flask server is running - starting tests...\n")
        main()
    except requests.exceptions.RequestException:
        print("❌ Flask server not running or not accessible")
        print("   Please start the server first: python i_sight_flask_integrated.py")
        sys.exit(1)
