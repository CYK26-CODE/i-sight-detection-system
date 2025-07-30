#!/usr/bin/env python3
"""
Test Voice Queue Processing
Test to ensure the voice worker thread processes all queued messages
"""

import time
import threading
import queue
from datetime import datetime

# Voice output setup
try:
    import pyttsx3
    VOICE_AVAILABLE = True
    print("✅ pyttsx3 available for queue test")
except ImportError:
    VOICE_AVAILABLE = False
    print("❌ pyttsx3 not available")

class VoiceQueueTester:
    """Test voice queue processing"""
    
    def __init__(self):
        self.engine = None
        self.voice_queue = queue.Queue()
        self.running = False
        self.voice_thread = None
        self.processed_count = 0
        self.voice_lock = threading.Lock()
        
        # Initialize TTS engine
        if VOICE_AVAILABLE:
            try:
                self.engine = pyttsx3.init('sapi5')
                self.engine.setProperty('rate', 220)  # Faster for testing
                self.engine.setProperty('volume', 1.0)
                print("✅ Test TTS engine initialized")
            except Exception as e:
                print(f"❌ Test TTS engine failed: {e}")
                self.engine = None
    
    def _voice_worker(self):
        """Test voice worker thread"""
        print("🔊 Test voice worker thread started")
        while self.running:
            try:
                message = self.voice_queue.get(timeout=1)
                if self.engine and message and self.running:
                    print(f"🔊 Processing: {message}")
                    
                    try:
                        # Clear any pending speech
                        self.engine.stop()
                        
                        # Speak the message
                        self.engine.say(message)
                        self.engine.runAndWait()
                        
                        with self.voice_lock:
                            self.processed_count += 1
                        
                        print(f"✅ Completed: {message} (Total processed: {self.processed_count})")
                        
                        # Small delay between messages
                        time.sleep(0.5)
                        
                    except Exception as voice_error:
                        print(f"❌ Voice processing failed: {voice_error}")
                
                # Mark task as done
                self.voice_queue.task_done()
                
            except queue.Empty:
                # Timeout - continue
                continue
            except Exception as e:
                print(f"❌ Voice worker error: {e}")
                try:
                    self.voice_queue.task_done()
                except:
                    pass
                time.sleep(0.5)
        
        print("🔊 Test voice worker thread stopped")
    
    def start_voice_worker(self):
        """Start the voice worker thread"""
        if self.running:
            print("⚠️ Voice worker already running")
            return
        
        self.running = True
        self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.voice_thread.start()
        print("✅ Test voice worker started")
    
    def queue_message(self, message):
        """Queue a message for speaking"""
        if not self.running:
            print("❌ Voice worker not running")
            return False
        
        try:
            self.voice_queue.put(message, timeout=0.1)
            print(f"📝 Queued: {message} (Queue size: {self.voice_queue.qsize()})")
            return True
        except queue.Full:
            print(f"❌ Queue full, cannot add: {message}")
            return False
    
    def stop_voice_worker(self):
        """Stop the voice worker"""
        print("🛑 Stopping test voice worker...")
        self.running = False
        
        if self.voice_thread:
            self.voice_thread.join(timeout=3)
        
        print(f"✅ Test completed. Total messages processed: {self.processed_count}")

def main():
    """Main test function"""
    print("🧪 Voice Queue Processing Test")
    print("=" * 40)
    
    # Create tester
    tester = VoiceQueueTester()
    
    if not tester.engine:
        print("❌ No TTS engine available - exiting")
        return
    
    # Start voice worker
    tester.start_voice_worker()
    
    # Test messages
    test_messages = [
        "Test message one",
        "Test message two", 
        "Test message three",
        "Test message four",
        "Test message five"
    ]
    
    print(f"\n📝 Queuing {len(test_messages)} test messages...")
    
    # Queue all messages
    queued_count = 0
    for i, message in enumerate(test_messages, 1):
        if tester.queue_message(f"{message} - {i}"):
            queued_count += 1
        time.sleep(0.2)  # Small delay between queuing
    
    print(f"\n⏳ Waiting for all messages to be processed...")
    print(f"📊 Queued: {queued_count}, Processing...")
    
    # Wait for processing to complete
    start_time = time.time()
    timeout = 30  # 30 second timeout
    
    while time.time() - start_time < timeout:
        if tester.processed_count >= queued_count:
            break
        print(f"📊 Progress: {tester.processed_count}/{queued_count} processed...")
        time.sleep(1)
    
    # Stop worker
    tester.stop_voice_worker()
    
    # Results
    print(f"\n📊 Test Results:")
    print(f"✅ Messages queued: {queued_count}")
    print(f"✅ Messages processed: {tester.processed_count}")
    print(f"✅ Success rate: {(tester.processed_count/queued_count*100):.1f}%" if queued_count > 0 else "N/A")
    
    if tester.processed_count == queued_count:
        print("🎉 ALL MESSAGES PROCESSED SUCCESSFULLY!")
    else:
        print("❌ Some messages were not processed")

if __name__ == "__main__":
    main()
