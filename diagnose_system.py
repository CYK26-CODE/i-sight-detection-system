#!/usr/bin/env python3
"""
i-sight System Diagnostic Tool
Monitors the Flask system to identify why detection might be stopping
"""

import requests
import time
import json
from datetime import datetime

def check_system_health():
    """Check the overall system health"""
    base_url = "http://localhost:5000"
    
    print("🔍 i-sight System Diagnostic")
    print("=" * 50)
    
    try:
        # Check if Flask server is running
        print("📡 Checking Flask server...")
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("✅ Flask server is running")
            print(f"   Detection Running: {status.get('detection_running', False)}")
            print(f"   Camera Connected: {status.get('camera_connected', False)}")
            print(f"   Voice Enabled: {status.get('voice_enabled', False)}")
        else:
            print(f"❌ Flask server error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Flask server not accessible. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error checking Flask server: {e}")
        return False
    
    # Get detailed debug info
    try:
        print("\n🔧 Getting detailed debug info...")
        response = requests.get(f"{base_url}/api/debug", timeout=5)
        if response.status_code == 200:
            debug = response.json()
            print("✅ Debug info retrieved")
            print(f"   Detection Running: {debug.get('detection_running', False)}")
            print(f"   Detector Exists: {debug.get('detector_exists', False)}")
            print(f"   Thread Exists: {debug.get('thread_exists', False)}")
            print(f"   Thread Alive: {debug.get('thread_alive', False)}")
            print(f"   Camera Available: {debug.get('camera_available', False)}")
            print(f"   Camera Opened: {debug.get('camera_opened', False)}")
            print(f"   Voice Available: {debug.get('voice_available', False)}")
            print(f"   Voice Running: {debug.get('voice_running', False)}")
            
            # Identify potential issues
            issues = []
            if not debug.get('detection_running', False):
                issues.append("Detection is not running")
            if not debug.get('detector_exists', False):
                issues.append("Detector not initialized")
            if not debug.get('thread_alive', False):
                issues.append("Detection thread is dead")
            if not debug.get('camera_available', False):
                issues.append("Camera not available")
            if not debug.get('camera_opened', False):
                issues.append("Camera not opened")
            
            if issues:
                print(f"\n⚠️  Potential Issues Found:")
                for issue in issues:
                    print(f"   • {issue}")
            else:
                print(f"\n✅ No obvious issues detected")
                
        else:
            print(f"❌ Debug info error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting debug info: {e}")
    
    # Get stats
    try:
        print("\n📊 Getting system stats...")
        response = requests.get(f"{base_url}/api/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Stats retrieved")
            print(f"   Frame Count: {stats.get('frame_count', 0)}")
            print(f"   FPS: {stats.get('fps', 0.0):.1f}")
            print(f"   Face Count: {stats.get('total_faces', 0)}")
            print(f"   Vehicle Count: {stats.get('total_vehicles', 0)}")
            print(f"   Traffic Signs: {stats.get('total_traffic_signs', 0)}")
        else:
            print(f"❌ Stats error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    return True

def monitor_system(duration=60, interval=5):
    """Monitor the system for a specified duration"""
    print(f"\n📈 Monitoring system for {duration} seconds (checking every {interval}s)...")
    print("=" * 50)
    
    start_time = time.time()
    check_count = 0
    
    while time.time() - start_time < duration:
        check_count += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        
        try:
            # Check status
            response = requests.get("http://localhost:5000/api/status", timeout=2)
            if response.status_code == 200:
                status = response.json()
                detection_running = status.get('detection_running', False)
                camera_connected = status.get('camera_connected', False)
                
                status_icon = "🟢" if detection_running else "🔴"
                camera_icon = "🟢" if camera_connected else "🔴"
                
                print(f"[{current_time}] Check #{check_count}: {status_icon} Detection {camera_icon} Camera")
            else:
                print(f"[{current_time}] Check #{check_count}: ❌ Server error {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"[{current_time}] Check #{check_count}: ❌ Server not accessible")
        except Exception as e:
            print(f"[{current_time}] Check #{check_count}: ❌ Error: {e}")
        
        time.sleep(interval)
    
    print(f"\n✅ Monitoring completed ({check_count} checks)")

def start_detection():
    """Start detection if not running"""
    try:
        print("\n🚀 Attempting to start detection...")
        response = requests.post("http://localhost:5000/api/start-detection", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get('success', False):
                print("✅ Detection started successfully")
            else:
                print(f"❌ Failed to start detection: {result.get('message', 'Unknown error')}")
        else:
            print(f"❌ Start detection error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error starting detection: {e}")

def stop_detection():
    """Stop detection"""
    try:
        print("\n🛑 Stopping detection...")
        response = requests.post("http://localhost:5000/api/stop-detection", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get('success', False):
                print("✅ Detection stopped successfully")
            else:
                print(f"❌ Failed to stop detection: {result.get('message', 'Unknown error')}")
        else:
            print(f"❌ Stop detection error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error stopping detection: {e}")

def main():
    """Main diagnostic function"""
    print("🔍 i-sight System Diagnostic Tool")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Check system health")
        print("2. Monitor system (60 seconds)")
        print("3. Start detection")
        print("4. Stop detection")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            check_system_health()
        elif choice == "2":
            monitor_system()
        elif choice == "3":
            start_detection()
        elif choice == "4":
            stop_detection()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 