#!/usr/bin/env python3
"""
i-sight Voice Log Reader - Real-time TTS reader for voice logs
Designed for visually impaired users to hear system activities in real-time
"""

import time
import threading
import queue
import os
from datetime import datetime
import re

# Voice output setup
try:
    import pyttsx3
    VOICE_AVAILABLE = True
    print("✅ pyttsx3 available for voice output")
except ImportError:
    VOICE_AVAILABLE = False
    print("❌ pyttsx3 not available - installing...")
    try:
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyttsx3'])
        import pyttsx3
        VOICE_AVAILABLE = True
        print("✅ pyttsx3 installed successfully")
    except Exception as e:
        print(f"❌ Failed to install pyttsx3: {e}")

class RealTimeVoiceLogReader:
    """Real-time voice log reader for visually impaired users"""
    
    def __init__(self, log_file_path="i_sight_voice_log.txt"):
        self.log_file_path = log_file_path
        self.engine = None
        self.reading_thread = None
        self.running = False
        self.last_file_position = 0
        self.voice_enabled = True
        
        # Initialize TTS engine with maximum reliability
        self.initialize_voice_engine()
        
        # Start monitoring thread (no voice queue thread needed)
        if self.engine:
            self.running = True
            
            self.reading_thread = threading.Thread(target=self._file_monitor, daemon=True)
            self.reading_thread.start()
            
            print("✅ Real-time voice log reader initialized")
            self.speak_immediate("i-sight voice log reader activated. Monitoring system activities.")
            
            # Test voice immediately to ensure it works
            time.sleep(2)
            self.speak_immediate("Voice system test. If you hear this, the voice reader is working correctly.")
        else:
            print("❌ Voice engine failed to initialize")
    
    def initialize_voice_engine(self):
        """Initialize TTS engine with multiple fallback methods"""
        if not VOICE_AVAILABLE:
            print("❌ TTS not available")
            return
            
        print("🔊 Initializing TTS engine for log reader...")
        
        # Try multiple initialization methods
        engines_to_try = ['sapi5', 'nsss', 'espeak', None]
        
        for engine_name in engines_to_try:
            try:
                if engine_name:
                    self.engine = pyttsx3.init(engine_name)
                    print(f"✅ Using {engine_name} engine")
                else:
                    self.engine = pyttsx3.init()
                    print("✅ Using default engine")
                
                # Configure engine for maximum clarity
                self.engine.setProperty('rate', 180)  # Slightly slower for clarity
                self.engine.setProperty('volume', 1.0)  # Maximum volume
                
                # Get and set best available voice
                voices = self.engine.getProperty('voices')
                if voices:
                    print(f"🔊 Found {len(voices)} voices available")
                    
                    # Prefer clear, understandable voices
                    preferred_voices = ['zira', 'david', 'hazel', 'mark']
                    selected_voice = None
                    
                    for pref in preferred_voices:
                        for voice in voices:
                            if pref in voice.name.lower():
                                selected_voice = voice
                                break
                        if selected_voice:
                            break
                    
                    if not selected_voice:
                        selected_voice = voices[0]
                    
                    self.engine.setProperty('voice', selected_voice.id)
                    print(f"✅ Selected voice: {selected_voice.name}")
                
                # Test the engine
                test_result = self.test_voice_engine()
                if test_result:
                    print("✅ Voice engine test successful")
                    break
                else:
                    print(f"❌ {engine_name or 'default'} engine test failed")
                    self.engine = None
                    
            except Exception as e:
                print(f"❌ Failed to initialize {engine_name or 'default'} engine: {e}")
                self.engine = None
                continue
        
        if not self.engine:
            print("❌ All TTS engines failed to initialize")
    
    def test_voice_engine(self):
        """Test if voice engine is working"""
        try:
            self.engine.say("Voice test")
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"❌ Voice engine test failed: {e}")
            return False
    
    def speak_immediate(self, message):
        """Speak message immediately without queuing - REAL-TIME VOICE"""
        if self.running and self.voice_enabled and self.engine:
            try:
                print(f"🔊 Speaking immediately: {message}")
                self.engine.stop()  # Clear any pending speech
                self.engine.say(message)
                self.engine.runAndWait()
                print(f"✅ Voice completed")
            except Exception as e:
                print(f"❌ Immediate voice failed: {e}")
                # Try to recover
                try:
                    self.initialize_voice_engine()
                    if self.engine:
                        self.engine.say(message)
                        self.engine.runAndWait()
                        print(f"✅ Voice recovered and completed")
                except Exception as recovery_error:
                    print(f"❌ Voice recovery failed: {recovery_error}")
        else:
            print(f"❌ Voice disabled or system not running: {message}")
    
    def _file_monitor(self):
        """Monitor log file for new entries"""
        print(f"📁 Monitoring log file: {self.log_file_path}")
        
        # Wait for file to exist
        while self.running:
            if os.path.exists(self.log_file_path):
                break
            print(f"⏳ Waiting for log file to be created...")
            time.sleep(1)
        
        # Get initial file size - but start from current position, not end
        try:
            current_size = os.path.getsize(self.log_file_path)
            print(f"📄 Current file size: {current_size} bytes")
            
            # Start from current end of file to monitor new entries only
            self.last_file_position = current_size
            print(f"📄 Starting to monitor from position {self.last_file_position}")
            
            # But also announce that we're ready
            self.speak_immediate("Voice reader is now monitoring for new detections.")
            
        except Exception as e:
            print(f"❌ Error reading initial file position: {e}")
            self.last_file_position = 0
        
        # Monitor for new content
        while self.running:
            try:
                current_size = os.path.getsize(self.log_file_path)
                
                if current_size > self.last_file_position:
                    # Read new content
                    print(f"📈 File grew from {self.last_file_position} to {current_size} bytes")
                    
                    with open(self.log_file_path, 'r', encoding='utf-8') as f:
                        f.seek(self.last_file_position)
                        new_content = f.read()
                        new_position = f.tell()
                    
                    # Process new lines
                    if new_content.strip():
                        print(f"📖 New content: {new_content[:100]}...")
                        self.process_new_log_entries(new_content)
                    
                    self.last_file_position = new_position
                    
                elif current_size < self.last_file_position:
                    # File was truncated or recreated
                    print(f"🔄 File size decreased from {self.last_file_position} to {current_size}. File may have been recreated.")
                    self.last_file_position = current_size
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"❌ Error monitoring file: {e}")
                time.sleep(1)
    
    def process_new_log_entries(self, content):
        """Process and speak new log entries - IMMEDIATE REAL-TIME VOICE"""
        lines = content.strip().split('\n')
        print(f"📄 Processing {len(lines)} new log lines")
        
        for line_num, line in enumerate(lines, 1):
            if line.strip():
                print(f"📝 Processing line {line_num}: {line.strip()}")
                spoken_message = self.parse_log_line(line)
                if spoken_message:
                    print(f"🔊 Will speak immediately: {spoken_message}")
                    self.speak_immediate(spoken_message)
                    # No delay - speak everything immediately as it comes
                else:
                    print(f"🔇 Skipping line: {line.strip()}")
    
    def parse_log_line(self, line):
        """Parse log line and convert to natural speech - SPEAK EVERYTHING IMPORTANT"""
        try:
            # Skip header lines and empty lines
            if line.startswith('===') or not line.strip():
                return None
            
            print(f"🔍 Parsing log line: {line.strip()}")  # Debug output
            
            # Parse log format: [timestamp] ACTION: "message" (Count: X)
            pattern = r'\[(\d{2}:\d{2}:\d{2})\] (\w+): "([^"]+)" \(Count: (\d+)\)'
            match = re.match(pattern, line)
            
            if match:
                timestamp, action, message, count = match.groups()
                count = int(count)
                
                print(f"🔍 Matched: action={action}, message={message}, count={count}")  # Debug
                
                # Create natural speech based on action - SPEAK MORE ACTIONS
                if action == "QUEUED":
                    return f"Detection queued: {message}"
                elif action == "PROCESSING":
                    # Don't skip processing - speak it but make it brief
                    return f"Processing: {message}"
                elif action == "COMPLETED":
                    return f"Announced: {message}"
                elif action == "FORCE_STARTED":
                    return f"System message: {message}"
                elif action == "FORCE_COMPLETED":
                    return f"System completed: {message}"
                elif action.startswith("FORCE_STARTED_ATTEMPT"):
                    return f"Voice attempting: {message}"
                elif action.startswith("FORCE_COMPLETED_ATTEMPT"):
                    return f"Voice successful: {message}"
                elif action == "ERROR":
                    return f"Voice error for: {message}"
                elif action == "SYSTEM_SHUTDOWN":
                    return "i-sight voice system shutting down"
                else:
                    # Speak any other action we haven't explicitly handled
                    return f"System {action.lower().replace('_', ' ')}: {message}"
            
            # Handle session management lines
            elif "Session Summary" in line:
                return "System generating session summary"
            elif "End Session" in line:
                return "Session ended"
            elif line.strip().startswith('  "') and 'times' in line:
                # Handle summary lines like: "Person detected": 5 times
                return f"Summary: {line.strip()}"
            else:
                # If we can't parse it but it looks important, speak it anyway
                if any(keyword in line.lower() for keyword in ['detected', 'error', 'failed', 'completed', 'started']):
                    clean_line = line.strip()
                    return f"System update: {clean_line}"
                return None
                
        except Exception as e:
            print(f"❌ Error parsing log line '{line}': {e}")
            # If parsing fails, try to speak the raw line if it looks important
            if any(keyword in line.lower() for keyword in ['detected', 'error', 'system', 'voice']):
                return f"System message: {line.strip()}"
            return None
    
    def toggle_voice(self):
        """Toggle voice on/off"""
        self.voice_enabled = not self.voice_enabled
        status = "enabled" if self.voice_enabled else "disabled"
        print(f"🔊 Voice output {status}")
        if self.voice_enabled:
            self.speak_immediate(f"Voice output {status}")
    
    def stop(self):
        """Stop the log reader"""
        print("🛑 Stopping voice log reader...")
        self.speak_immediate("Voice log reader stopping")
        time.sleep(2)  # Give time for final message
        self.running = False
        
        if self.reading_thread:
            self.reading_thread.join(timeout=2)
        if self.voice_thread:
            self.voice_thread.join(timeout=2)
        
        print("✅ Voice log reader stopped")

def main():
    """Main function to run the voice log reader"""
    print("🚀 Starting i-sight Voice Log Reader")
    print("📋 This tool provides real-time voice feedback for visually impaired users")
    print("🎧 Make sure your speakers/headphones are connected and volume is up")
    print()
    
    # Check if log file exists
    log_file = "i_sight_voice_log.txt"
    if not os.path.exists(log_file):
        print(f"⚠️  Log file '{log_file}' not found")
        print("🔄 This tool will wait for the i-sight system to create the log file")
        print("📝 Please start the main i-sight detector to generate voice logs")
        print()
    
    reader = None
    try:
        reader = RealTimeVoiceLogReader(log_file)
        
        if reader.engine:
            print("\n🎮 Voice Log Reader Controls:")
            print("- Press 'v' + Enter to toggle voice on/off")
            print("- Press 'q' + Enter to quit")
            print("- Press Ctrl+C to force quit")
            print()
            
            # Interactive loop
            while reader.running:
                try:
                    user_input = input().strip().lower()
                    if user_input == 'q':
                        break
                    elif user_input == 'v':
                        reader.toggle_voice()
                    elif user_input == 'status':
                        reader.speak_immediate("Voice log reader is active and monitoring system")
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
        else:
            print("❌ Failed to initialize voice system")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if reader:
            reader.stop()

if __name__ == "__main__":
    main()
