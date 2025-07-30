#!/usr/bin/env python3
"""
i-sight Real-time Voice Log Reader for Blind Users
Continuously reads and speaks the i-sight voice log in real-time
Activates when first face detection occurs
"""

import time
import os
import threading
from datetime import datetime

# Voice output setup
try:
    import pyttsx3
    VOICE_AVAILABLE = True
    print("✅ Voice system available")
except ImportError:
    VOICE_AVAILABLE = False
    print("❌ Voice system not available")

class RealTimeVoiceLogReader:
    """Real-time voice log reader that activates on first face detection"""
    
    def __init__(self, log_file_path="i_sight_voice_log.txt"):
        self.log_file_path = log_file_path
        self.engine = None
        self.monitoring = False
        self.activated = False  # Becomes True when first face is detected
        self.last_position = 0
        self.monitoring_thread = None
        
        # Initialize voice engine
        if VOICE_AVAILABLE:
            try:
                self.engine = pyttsx3.init('sapi5')
                self.engine.setProperty('rate', 160)  # Slightly slower for clarity
                self.engine.setProperty('volume', 1.0)
                
                # Set voice if available
                voices = self.engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'zira' in voice.name.lower() or 'david' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
                
                print("✅ Voice engine initialized")
            except Exception as e:
                print(f"❌ Voice engine failed: {e}")
                self.engine = None
    
    def speak_immediately(self, message):
        """Speak message immediately"""
        if self.engine and self.activated:
            try:
                print(f"🔊 Speaking: {message}")
                self.engine.stop()  # Clear any pending speech
                self.engine.say(message)
                self.engine.runAndWait()
            except Exception as e:
                print(f"❌ Speech failed: {e}")
    
    def parse_log_entry(self, log_line):
        """Parse log entry and extract meaningful information"""
        try:
            if not log_line.strip() or log_line.startswith('==='):
                return None
            
            # Check for face detection (first activation trigger)
            if "Person detected" in log_line and not self.activated:
                self.activated = True
                self.speak_immediately("i-sight system activated. Starting real-time voice feedback.")
                print("🎯 ACTIVATED: First face detected - voice feedback started")
            
            # Only process entries after activation
            if not self.activated:
                return None
            
            # Parse different types of log entries
            if "QUEUED:" in log_line:
                # Extract the message from quotes
                start_quote = log_line.find('"')
                end_quote = log_line.find('"', start_quote + 1)
                if start_quote != -1 and end_quote != -1:
                    message = log_line[start_quote + 1:end_quote]
                    return f"Detection: {message}"
            
            elif "FORCE_STARTED_ATTEMPT" in log_line:
                start_quote = log_line.find('"')
                end_quote = log_line.find('"', start_quote + 1)
                if start_quote != -1 and end_quote != -1:
                    message = log_line[start_quote + 1:end_quote]
                    return f"System: {message}"
            
            elif "SYSTEM_SHUTDOWN" in log_line:
                return "i-sight system shutting down"
            
            elif "Session Summary" in log_line:
                return "Generating session summary"
            
            return None
            
        except Exception as e:
            print(f"❌ Error parsing log line: {e}")
            return None
    
    def monitor_log_file(self):
        """Monitor log file for new entries"""
        print(f"👁️ Monitoring i-sight voice log: {self.log_file_path}")
        print("⏳ Waiting for first face detection to activate voice feedback...")
        
        # Wait for file to exist
        while self.monitoring:
            if os.path.exists(self.log_file_path):
                break
            print("📁 Waiting for i-sight log file to be created...")
            time.sleep(2)
        
        if not self.monitoring:
            return
        
        # Get initial file position
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                f.seek(0, 2)  # Go to end of file
                self.last_position = f.tell()
            print(f"📄 Starting monitoring from position: {self.last_position}")
        except Exception as e:
            print(f"❌ Error reading initial file position: {e}")
            self.last_position = 0
        
        # Monitor for new content
        while self.monitoring:
            try:
                current_size = os.path.getsize(self.log_file_path)
                
                if current_size > self.last_position:
                    # Read new content
                    with open(self.log_file_path, 'r', encoding='utf-8') as f:
                        f.seek(self.last_position)
                        new_content = f.read()
                        self.last_position = f.tell()
                    
                    # Process new lines
                    if new_content.strip():
                        lines = new_content.strip().split('\n')
                        for line in lines:
                            if line.strip():
                                spoken_message = self.parse_log_entry(line)
                                if spoken_message:
                                    self.speak_immediately(spoken_message)
                
                elif current_size < self.last_position:
                    # File was recreated or truncated
                    print("🔄 Log file was recreated, resetting position")
                    self.last_position = 0
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"❌ Error monitoring file: {e}")
                time.sleep(1)
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        if self.monitoring:
            print("⚠️ Already monitoring")
            return
        
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self.monitor_log_file, daemon=True)
        self.monitoring_thread.start()
        print("✅ Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        print("🛑 Stopping real-time voice feedback...")
        self.monitoring = False
        
        if self.activated and self.engine:
            self.speak_immediately("Real-time voice feedback stopped")
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        
        print("✅ Monitoring stopped")

def main():
    """Main function"""
    print("👁️ i-sight Real-time Voice Log Reader for Blind Users")
    print("=" * 55)
    print("🎯 This system will activate when the first face is detected")
    print("🔊 Once activated, it will speak all i-sight activities in real-time")
    print("🎧 Make sure your speakers are on and volume is up")
    print()
    
    reader = None
    try:
        reader = RealTimeVoiceLogReader()
        
        if not reader.engine:
            print("❌ Voice engine not available - exiting")
            return
        
        # Start monitoring
        reader.start_monitoring()
        
        print("📋 Commands:")
        print("- Just wait for the first face detection to activate the system")
        print("- Press Ctrl+C to stop")
        print()
        
        # Keep running until interrupted
        try:
            while reader.monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if reader:
            reader.stop_monitoring()

if __name__ == "__main__":
    main()
