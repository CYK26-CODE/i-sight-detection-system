#!/usr/bin/env python3
"""
i-sight System Monitor
Monitors system health and provides real-time status
"""

import requests
import time
import json
from datetime import datetime

def check_system_health():
    """Check system health status"""
    try:
        # Check system health
        health_response = requests.get("http://localhost:5000/api/system-health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            return health_data
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def check_stats():
    """Check system statistics"""
    try:
        stats_response = requests.get("http://localhost:5000/api/stats", timeout=5)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            return stats_data
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def check_status():
    """Check basic status"""
    try:
        status_response = requests.get("http://localhost:5000/api/status", timeout=5)
        if status_response.status_code == 200:
            status_data = status_response.json()
            return status_data
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def monitor_system(interval=5, duration=None):
    """Monitor system continuously"""
    print("🔍 i-sight System Monitor")
    print("=" * 50)
    print(f"📡 Monitoring interval: {interval} seconds")
    if duration:
        print(f"⏱️  Duration: {duration} seconds")
    print("=" * 50)
    
    start_time = time.time()
    iteration = 0
    
    while True:
        iteration += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n🕐 [{current_time}] Iteration {iteration}")
        print("-" * 30)
        
        # Check basic status
        status = check_status()
        if status:
            print(f"✅ Detection Running: {status.get('detection_running', False)}")
            print(f"📹 Camera Connected: {status.get('camera_connected', False)}")
            print(f"🔊 Voice Enabled: {status.get('voice_enabled', False)}")
        else:
            print("❌ Cannot connect to Flask server")
            print("💡 Make sure the Flask application is running")
            time.sleep(interval)
            continue
        
        # Check system health
        health = check_system_health()
        if health:
            print(f"💓 System Status: {health.get('system_status', 'unknown')}")
            print(f"⏱️  Uptime: {health.get('system_uptime', 0):.1f}s")
            print(f"📊 Total Frames: {health.get('total_frames_processed', 0)}")
            print(f"⚠️  Consecutive Failures: {health.get('consecutive_failures', 0)}")
            print(f"🔄 Recovery Attempts: {health.get('recovery_attempts', 0)}")
        
        # Check detailed stats
        stats = check_stats()
        if stats:
            print(f"🎯 FPS: {stats.get('fps', 0):.1f}")
            print(f"📸 Frame Count: {stats.get('frame_count', 0)}")
            print(f"👥 Face Count: {stats.get('total_faces', 0)}")
            print(f"🚗 Vehicle Count: {stats.get('total_vehicles', 0)}")
            print(f"🛑 Traffic Signs: {stats.get('total_traffic_signs', 0)}")
            print(f"📹 Camera Status: {stats.get('camera_status', 'unknown')}")
            print(f"🔊 Voice Status: {stats.get('voice_status', 'unknown')}")
        
        # Check for issues
        if health and health.get('system_status') == 'warning':
            print("⚠️  SYSTEM WARNING: Last heartbeat is old")
        
        if stats and stats.get('fps', 0) < 10:
            print("⚠️  PERFORMANCE WARNING: Low FPS detected")
        
        if stats and stats.get('consecutive_failures', 0) > 5:
            print("⚠️  ERROR WARNING: High failure rate detected")
        
        # Check duration
        if duration and (time.time() - start_time) >= duration:
            print(f"\n⏰ Monitoring completed after {duration} seconds")
            break
        
        time.sleep(interval)

def quick_status():
    """Quick status check"""
    print("🔍 Quick System Status Check")
    print("=" * 30)
    
    status = check_status()
    if status:
        print(f"✅ Detection: {'Running' if status.get('detection_running') else 'Stopped'}")
        print(f"📹 Camera: {'Connected' if status.get('camera_connected') else 'Disconnected'}")
        print(f"🔊 Voice: {'Enabled' if status.get('voice_enabled') else 'Disabled'}")
        
        stats = check_stats()
        if stats:
            print(f"🎯 FPS: {stats.get('fps', 0):.1f}")
            print(f"📸 Frames: {stats.get('frame_count', 0)}")
            print(f"👥 Faces: {stats.get('total_faces', 0)}")
    else:
        print("❌ System not responding")
        print("💡 Make sure the Flask application is running")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "monitor":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else None
            monitor_system(interval, duration)
        elif command == "status":
            quick_status()
        elif command == "health":
            health = check_system_health()
            if health:
                print(json.dumps(health, indent=2))
            else:
                print("❌ Cannot get health data")
        elif command == "stats":
            stats = check_stats()
            if stats:
                print(json.dumps(stats, indent=2))
            else:
                print("❌ Cannot get stats")
        else:
            print("Usage:")
            print("  python monitor_system.py status     - Quick status check")
            print("  python monitor_system.py health     - System health data")
            print("  python monitor_system.py stats      - System statistics")
            print("  python monitor_system.py monitor    - Continuous monitoring")
            print("  python monitor_system.py monitor 10 - Monitor with 10s interval")
    else:
        quick_status()

if __name__ == "__main__":
    main() 