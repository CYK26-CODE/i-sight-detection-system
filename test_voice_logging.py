#!/usr/bin/env python3
"""
Test Voice Logging System
Simple test to verify voice logging is working correctly
"""

import time
from datetime import datetime
import threading

# Voice output setup
try:
    import pyttsx3
    VOICE_AVAILABLE = True
    print("✅ pyttsx3 available for voice logging test")
except ImportError:
    VOICE_AVAILABLE = False
    print("❌ pyttsx3 not available")

class TestVoiceLogger:
    """Test voice logging functionality"""
    
    def __init__(self):
        self.voice_log_file = "i_sight_voice_log.txt"
        self.statement_count = {}
        self.voice_lock = threading.Lock()
        self.engine = None
        
        # Initialize voice log file with test header
        try:
            with open(self.voice_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== i-sight Voice Log TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            print(f"✅ Test voice log initialized: {self.voice_log_file}")
        except Exception as e:
            print(f"❌ Test voice log initialization failed: {e}")
            return
        
        # Initialize TTS engine
        if VOICE_AVAILABLE:
            try:
                self.engine = pyttsx3.init('sapi5')
                self.engine.setProperty('rate', 160)
                self.engine.setProperty('volume', 1.0)
                print("✅ Test TTS engine initialized")
            except Exception as e:
                print(f"❌ Test TTS engine failed: {e}")
                self.engine = None
    
    def log_voice_message(self, message: str, action: str = "TEST"):
        """Test voice logging functionality"""
        try:
            with self.voice_lock:
                # Update statement count
                if message not in self.statement_count:
                    self.statement_count[message] = 0
                self.statement_count[message] += 1
                
                # Create log entry
                timestamp = datetime.now().strftime('%H:%M:%S')
                repetition = self.statement_count[message]
                log_entry = f"[{timestamp}] {action}: \"{message}\" (Count: {repetition})\n"
                
                # Write to file
                with open(self.voice_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
                    f.flush()  # Force write to disk
                
                print(f"📝 Test log written: {message} (#{repetition})")
                return True
        except Exception as e:
            print(f"❌ Test voice logging failed: {e}")
            return False
    
    def test_voice_and_logging(self, message):
        """Test both voice output and logging"""
        print(f"\n🧪 Testing: {message}")
        
        # Log the start
        log_success = self.log_voice_message(message, "TEST_STARTED")
        
        if self.engine:
            try:
                print(f"🔊 Speaking: {message}")
                self.engine.say(message)
                self.engine.runAndWait()
                
                # Log successful completion
                self.log_voice_message(message, "TEST_COMPLETED")
                print(f"✅ Voice and logging test successful")
                return True
                
            except Exception as e:
                print(f"❌ Voice test failed: {e}")
                self.log_voice_message(message, "TEST_VOICE_FAILED")
                return False
        else:
            print("❌ No TTS engine available")
            self.log_voice_message(message, "TEST_NO_ENGINE")
            return False

def main():
    """Main test function"""
    print("🧪 i-sight Voice Logging Test")
    print("=" * 40)
    
    # Create test logger
    logger = TestVoiceLogger()
    
    # Test messages
    test_messages = [
        "Voice logging test one",
        "Person detected in Center zone",
        "i-sight system test message",
        "Voice feedback system check"
    ]
    
    print(f"\n📝 Testing voice logging with {len(test_messages)} messages...")
    
    # Test each message
    success_count = 0
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Test {i}/{len(test_messages)} ---")
        if logger.test_voice_and_logging(message):
            success_count += 1
        time.sleep(1)  # Brief pause between tests
    
    # Write test summary
    try:
        with open(logger.voice_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== Test Summary ===\n")
            f.write(f"Tests run: {len(test_messages)}\n")
            f.write(f"Tests passed: {success_count}\n")
            f.write(f"Success rate: {(success_count/len(test_messages)*100):.1f}%\n")
            for message, count in logger.statement_count.items():
                f.write(f"  \"{message}\": {count} times\n")
            f.write("=== End Test ===\n\n")
    except Exception as e:
        print(f"❌ Failed to write test summary: {e}")
    
    # Results
    print(f"\n📊 Test Results:")
    print(f"✅ Successful tests: {success_count}/{len(test_messages)}")
    print(f"📁 Log file: {logger.voice_log_file}")
    
    # Show log file content
    try:
        print(f"\n📖 Log file contents:")
        with open(logger.voice_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"❌ Could not read log file: {e}")

if __name__ == "__main__":
    main()
