#!/usr/bin/env python3
"""
Final voice system test - comprehensive test of all methods
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from i_sight_detector import VoiceManager

def main():
    print("=" * 60)
    print("🔊 FINAL VOICE SYSTEM TEST")
    print("=" * 60)
    
    # Initialize voice system
    print("\n1. Initializing voice system...")
    voice = VoiceManager()
    
    if not voice.running:
        print("❌ Voice system failed to initialize!")
        return
    
    print(f"✅ Voice system ready using: {voice.current_method}")
    
    # Test 1: Basic announcement
    print("\n2. Testing basic announcements...")
    messages = [
        "Person detected in center zone",
        "Vehicle approaching from left",
        "Multiple faces detected", 
        "Traffic sign identified ahead",
        "i-sight system operational"
    ]
    
    for i, msg in enumerate(messages, 1):
        print(f"   🔊 Test {i}/5: {msg}")
        voice.announce('test', msg)
        time.sleep(3.0)  # Wait for message to complete
    
    # Test 2: Force speak (critical messages)
    print("\n3. Testing force speak (critical messages)...")
    critical_messages = [
        "Emergency: obstacle detected",
        "Critical: system alert active", 
        "Important: navigation warning"
    ]
    
    for i, msg in enumerate(critical_messages, 1):
        print(f"   🚨 Critical {i}/3: {msg}")
        voice.force_speak(msg)
        time.sleep(2.0)
    
    # Test 3: Queue stress test
    print("\n4. Testing queue with rapid messages...")
    rapid_messages = [
        "Rapid test one",
        "Rapid test two", 
        "Rapid test three",
        "Rapid test four",
        "Rapid test five"
    ]
    
    for i, msg in enumerate(rapid_messages, 1):
        print(f"   ⚡ Rapid {i}/5: {msg}")
        voice.announce('rapid', msg)
        time.sleep(0.5)  # Rapid fire
    
    # Wait for all messages to complete
    print("\n5. Waiting for all messages to complete...")
    time.sleep(15)
    
    # Check final status
    print("\n6. Final status check...")
    print(f"   Queue size: {voice.voice_queue.qsize()}")
    print(f"   Voice method: {voice.current_method}")
    print(f"   System running: {voice.running}")
    
    # Shutdown
    print("\n7. Shutting down voice system...")
    voice.stop()
    
    print("\n" + "=" * 60)
    print("🎉 VOICE SYSTEM TEST COMPLETED!")
    print("🔊 Check i_sight_voice_log.txt for detailed results")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
