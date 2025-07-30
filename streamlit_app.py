#!/usr/bin/env python3
"""
i-sight Streamlit Application
Real-time computer vision detection system with voice feedback
"""

import streamlit as st
import cv2
import numpy as np
import time
import json
import threading
import queue
import base64
from datetime import datetime
import os
from PIL import Image
import subprocess

# Set page config
st.set_page_config(
    page_title="i-sight Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .detection-zone {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        min-width: 100px;
    }
    .zone-active {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        color: white;
    }
    .zone-inactive {
        background: #e0e0e0;
        color: #666;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
    }
    
    /* Accessibility Styles */
    .accessibility-header {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.5rem;
    }
    
    .accessibility-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #667eea;
    }
    
    .accessibility-btn {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem 3rem;
        border: none;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: bold;
        cursor: pointer;
        margin: 0.5rem;
        min-width: 250px;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .accessibility-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .accessibility-btn.success {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
    }
    
    .accessibility-btn.danger {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
    }
    
    .accessibility-btn.warning {
        background: linear-gradient(45deg, #f093fb, #f5576c);
    }
    
    .accessibility-btn.info {
        background: linear-gradient(45deg, #17a2b8, #138496);
    }
    
    .accessibility-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none !important;
    }
    
    .toggle-switch {
        position: relative;
        display: inline-block;
        width: 120px;
        height: 60px;
        background: #ccc;
        border-radius: 30px;
        cursor: pointer;
        transition: background 0.3s ease;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
        margin: 1rem;
    }
    
    .toggle-switch.active {
        background: #56ab2f;
    }
    
    .toggle-slider {
        position: absolute;
        top: 5px;
        left: 5px;
        width: 50px;
        height: 50px;
        background: white;
        border-radius: 50%;
        transition: transform 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .toggle-switch.active .toggle-slider {
        transform: translateX(60px);
    }
    
    .accessibility-status {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        border: 2px solid #667eea;
    }
    
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .status-item {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        text-align: center;
    }
    
    .status-item h3 {
        color: #667eea;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    
    .status-item p {
        color: #666;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .voice-feedback {
        position: fixed;
        bottom: 20px;
        left: 20px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.1rem;
        z-index: 1000;
        max-width: 400px;
        display: none;
    }
    
    .voice-feedback.show {
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Import detection system
try:
    from i_sight_flask_integrated import ISightDetector, VoiceManager, VOICE_AVAILABLE, YOLO_AVAILABLE, TRAFFIC_SIGN_AVAILABLE
    DETECTION_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Detection system import failed: {e}")
    DETECTION_AVAILABLE = False

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'detection_running' not in st.session_state:
        st.session_state.detection_running = False
    if 'latest_frame' not in st.session_state:
        st.session_state.latest_frame = None
    if 'detection_data' not in st.session_state:
        st.session_state.detection_data = {}
    if 'detection_thread' not in st.session_state:
        st.session_state.detection_thread = None

# Initialize session state
initialize_session_state()

def initialize_detector():
    """Initialize the detection system"""
    if st.session_state.detector is None:
        try:
            st.session_state.detector = ISightDetector()
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize detector: {e}")
            return False
    return True

def start_detection():
    """Start detection in background thread"""
    if not initialize_detector():
        return False
    
    if st.session_state.detection_running:
        return True
    
    try:
        st.session_state.detection_running = True
        
        def detection_worker():
            """Background worker for detection processing"""
            try:
                while True:
                    # Check if detection should stop
                    if not st.session_state.get('detection_running', False):
                        break
                    
                    # Process frame if detector is available
                    detector = st.session_state.get('detector')
                    if detector and detector.cap and detector.cap.isOpened():
                        success = detector.process_frame()
                        if success:
                            # Update session state with latest data
                            screenshot = detector.get_screenshot()
                            detection_data = detector.get_detection_data()
                            
                            # Update session state safely
                            st.session_state.latest_frame = screenshot
                            st.session_state.detection_data = detection_data
                    
                    time.sleep(1/30)  # 30 FPS
            except Exception as e:
                print(f"❌ Detection worker error: {e}")
                st.session_state.detection_running = False
        
        # Start detection thread
        detection_thread = threading.Thread(target=detection_worker, daemon=True)
        detection_thread.start()
        st.session_state.detection_thread = detection_thread
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to start detection: {e}")
        st.session_state.detection_running = False
        return False

def stop_detection():
    """Stop detection"""
    try:
        # Stop the detection loop
        st.session_state.detection_running = False
        
        # Wait for thread to finish
        if st.session_state.detection_thread and st.session_state.detection_thread.is_alive():
            st.session_state.detection_thread.join(timeout=2)
        
        # Cleanup detector
        if st.session_state.detector:
            st.session_state.detector.cleanup()
            st.session_state.detector = None
        
        # Clear session state
        st.session_state.latest_frame = None
        st.session_state.detection_data = {}
        st.session_state.detection_thread = None
        
    except Exception as e:
        st.error(f"❌ Error stopping detection: {e}")

def test_voice():
    """Test voice system"""
    if st.session_state.detector and st.session_state.detector.voice:
        try:
            test_message = f"Voice system test successful using {st.session_state.detector.voice.current_method}"
            st.session_state.detector.voice.announce('test', test_message)
            return True, "Voice test initiated successfully"
        except Exception as e:
            return False, f"Voice test failed: {str(e)}"
    return False, "Voice system not available"

def force_speak(message):
    """Force speak a message"""
    if st.session_state.detector and st.session_state.detector.voice:
        try:
            st.session_state.detector.voice.force_speak(message)
            return True, "Force speak initiated"
        except Exception as e:
            return False, f"Force speak failed: {str(e)}"
    return False, "Voice system not available"

def speak_feedback(message):
    """Provide voice feedback for accessibility"""
    try:
        # Use system text-to-speech if available
        import platform
        if platform.system() == "Windows":
            import subprocess
            subprocess.run(['powershell', '-Command', f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{message}")'], 
                         capture_output=True, timeout=5)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(['say', message], capture_output=True, timeout=5)
        else:  # Linux
            subprocess.run(['espeak', message], capture_output=True, timeout=5)
    except:
        pass  # Silently fail if TTS is not available

def accessibility_start_detection():
    """Start detection with accessibility feedback"""
    speak_feedback("Starting detection system")
    return start_detection()

def accessibility_stop_detection():
    """Stop detection with accessibility feedback"""
    speak_feedback("Stopping detection system")
    stop_detection()

def accessibility_test_voice():
    """Test voice with accessibility feedback"""
    speak_feedback("Testing voice system")
    return test_voice()

def accessibility_force_speak(message):
    """Force speak with accessibility feedback"""
    speak_feedback(f"Force speaking: {message}")
    return force_speak(message)

# Main app
def main():
    # Initialize session state
    initialize_session_state()
    
    # Tab navigation
    tab1, tab2 = st.tabs(["📊 Main Dashboard", "♿ Accessibility Mode"])
    
    with tab1:
        main_dashboard()
    
    with tab2:
        accessibility_dashboard()
    
    # Auto-refresh for main dashboard
    if st.session_state.detection_running:
        time.sleep(0.1)  # Small delay
        st.rerun()

def main_dashboard():
    """Main dashboard interface"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>👁️ i-sight Detection System</h1>
        <p>Real-time Computer Vision with Voice Feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Detection controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start", key="start_btn", use_container_width=True):
                if start_detection():
                    st.success("Detection started!")
                else:
                    st.error("Failed to start detection")
        
        with col2:
            if st.button("🛑 Stop", key="stop_btn", use_container_width=True):
                stop_detection()
                st.success("Detection stopped!")
        
        # Voice controls
        st.subheader("🔊 Voice System")
        if st.button("🔊 Test Voice", use_container_width=True):
            success, message = test_voice()
            if success:
                st.success(message)
            else:
                st.error(message)
        
        # Force speak
        force_message = st.text_input("Force speak message:", placeholder="Enter message...")
        if st.button("🎤 Force Speak", use_container_width=True) and force_message:
            success, message = force_speak(force_message)
            if success:
                st.success(message)
            else:
                st.error(message)
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.detector:
            detector = st.session_state.detector
            st.markdown(f"""
            <div class="status-card">
                <strong>Detection:</strong> {'🟢 Running' if st.session_state.detection_running else '🔴 Stopped'}<br>
                <strong>Camera:</strong> {'🟢 Connected' if detector.cap and detector.cap.isOpened() else '🔴 Disconnected'}<br>
                <strong>Voice:</strong> {'🟢 Enabled' if detector.voice and detector.voice.running else '🔴 Disabled'}<br>
                <strong>YOLO:</strong> {'🟢 Available' if YOLO_AVAILABLE else '🔴 Not Available'}<br>
                <strong>Traffic Signs:</strong> {'🟢 Available' if TRAFFIC_SIGN_AVAILABLE else '🔴 Not Available'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Detection system not initialized")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        
        # Video feed placeholder
        if st.session_state.latest_frame:
            # Convert bytes to image
            nparr = np.frombuffer(st.session_state.latest_frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            st.image(img_rgb, channels="RGB", use_container_width=True)
        else:
            # Placeholder when no video
            st.markdown("""
            <div style="background: #f0f2f6; padding: 2rem; text-align: center; border-radius: 10px;">
                <h3>📹 Video Feed</h3>
                <p>Start detection to see live video feed</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📈 Detection Metrics")
        
        # Metrics
        if st.session_state.detection_data:
            data = st.session_state.detection_data
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(data.get('faces', []))}</div>
                    <div class="metric-label">Faces</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(data.get('vehicles', []))}</div>
                    <div class="metric-label">Vehicles</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(data.get('traffic_signs', []))}</div>
                    <div class="metric-label">Traffic Signs</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{data.get('fps', 0.0):.1f}</div>
                    <div class="metric-label">FPS</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Detection zones
        st.subheader("🎯 Detection Zones")
        zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        
        if st.session_state.detection_data and 'faces' in st.session_state.detection_data:
            detected_zones = [face['zone'] for face in st.session_state.detection_data['faces']]
            
            for i, zone_name in enumerate(zone_names):
                is_active = i in detected_zones
                css_class = "zone-active" if is_active else "zone-inactive"
                st.markdown(f"""
                <div class="detection-zone {css_class}">
                    {zone_name}
                </div>
                """, unsafe_allow_html=True)
        else:
            for zone_name in zone_names:
                st.markdown(f"""
                <div class="detection-zone zone-inactive">
                    {zone_name}
                </div>
                """, unsafe_allow_html=True)
    
    # Detection details
    if st.session_state.detection_data:
        st.subheader("📋 Detection Details")
        
        data = st.session_state.detection_data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Face Detections:**")
            if data.get('faces'):
                for face in data['faces']:
                    st.write(f"• {face['zone_name']}: {face['count']} face(s)")
            else:
                st.write("No faces detected")
        
        with col2:
            st.write("**Vehicle Detections:**")
            if data.get('vehicles'):
                for vehicle in data['vehicles']:
                    st.write(f"• {vehicle['class']}: {vehicle['confidence']:.2f}")
            else:
                st.write("No vehicles detected")
        
        with col3:
            st.write("**Traffic Sign Detections:**")
            if data.get('traffic_signs'):
                for sign in data['traffic_signs']:
                    st.write(f"• {sign['sign_type']}: {sign['confidence']:.2f}")
            else:
                st.write("No traffic signs detected")
        
        # Real-time info
        st.subheader("⏱️ Real-time Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Frame Number", data.get('frame_number', 0))
        
        with col2:
            st.metric("Processing Time", f"{data.get('processing_time', 0.0):.3f}s")
        
        with col3:
            st.metric("Last Update", data.get('timestamp', 'Never'))

def accessibility_dashboard():
    """Accessibility dashboard with large buttons and voice feedback"""
    # Accessibility header
    st.markdown("""
    <div class="accessibility-header">
        <h1>♿ Accessibility Mode</h1>
        <p>Voice-Controlled Interface for Visually Disabled Users</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    speak_feedback("Accessibility mode activated. Large buttons and voice feedback enabled.")
    
    # Detection Control Section
    st.markdown("""
    <div class="accessibility-card">
        <h2>🎯 Detection System Control</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 START DETECTION", key="accessibility_start", use_container_width=True, 
                    help="Start the detection system with voice feedback"):
            if accessibility_start_detection():
                st.success("✅ Detection started successfully!")
                speak_feedback("Detection system started successfully")
            else:
                st.error("❌ Failed to start detection")
                speak_feedback("Failed to start detection system")
    
    with col2:
        if st.button("🛑 STOP DETECTION", key="accessibility_stop", use_container_width=True,
                    help="Stop the detection system with voice feedback"):
            accessibility_stop_detection()
            st.success("✅ Detection stopped successfully!")
            speak_feedback("Detection system stopped successfully")
    
    # Voice System Section
    st.markdown("""
    <div class="accessibility-card">
        <h2>🔊 Voice System Control</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔊 TEST VOICE SYSTEM", key="accessibility_voice_test", use_container_width=True,
                    help="Test the voice system with accessibility feedback"):
            success, message = accessibility_test_voice()
            if success:
                st.success("✅ Voice test successful!")
            else:
                st.error(f"❌ Voice test failed: {message}")
    
    with col2:
        # Force speak with large input
        force_message = st.text_input("Enter message to speak:", 
                                    placeholder="Type your message here...",
                                    key="accessibility_force_speak_input",
                                    help="Enter a message to be spoken by the system")
        
        if st.button("🎤 FORCE SPEAK MESSAGE", key="accessibility_force_speak", use_container_width=True,
                    help="Speak the entered message with voice feedback"):
            if force_message.strip():
                success, message = accessibility_force_speak(force_message.strip())
                if success:
                    st.success("✅ Message spoken successfully!")
                else:
                    st.error(f"❌ Failed to speak message: {message}")
            else:
                st.warning("⚠️ Please enter a message to speak")
                speak_feedback("Please enter a message to speak")
    
    # System Status Section
    st.markdown("""
    <div class="accessibility-card">
        <h2>📊 System Status Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Status grid
    if st.session_state.detector:
        detector = st.session_state.detector
        data = st.session_state.detection_data or {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="status-item">
                <h3>Detection Status</h3>
                <p>{'🟢 RUNNING' if st.session_state.detection_running else '🔴 STOPPED'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="status-item">
                <h3>Camera Status</h3>
                <p>{'🟢 CONNECTED' if detector.cap and detector.cap.isOpened() else '🔴 DISCONNECTED'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="status-item">
                <h3>Voice Status</h3>
                <p>{'🟢 ENABLED' if detector.voice and detector.voice.running else '🔴 DISABLED'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="status-item">
                <h3>System FPS</h3>
                <p>{data.get('fps', 0.0):.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detection counts
        st.markdown("""
        <div class="accessibility-status">
            <h3>📈 Detection Counts</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            face_count = len(data.get('faces', []))
            st.markdown(f"""
            <div class="status-item">
                <h3>Faces Detected</h3>
                <p>{face_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vehicle_count = len(data.get('vehicles', []))
            st.markdown(f"""
            <div class="status-item">
                <h3>Vehicles Detected</h3>
                <p>{vehicle_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            traffic_sign_count = len(data.get('traffic_signs', []))
            st.markdown(f"""
            <div class="status-item">
                <h3>Traffic Signs</h3>
                <p>{traffic_sign_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Voice announcement of status
        if st.button("🔊 ANNOUNCE STATUS", key="announce_status", use_container_width=True,
                    help="Announce current system status with voice feedback"):
            status_message = f"System status: Detection is {'running' if st.session_state.detection_running else 'stopped'}. "
            status_message += f"Camera is {'connected' if detector.cap and detector.cap.isOpened() else 'disconnected'}. "
            status_message += f"Voice is {'enabled' if detector.voice and detector.voice.running else 'disabled'}. "
            status_message += f"Detected {face_count} faces, {vehicle_count} vehicles, and {traffic_sign_count} traffic signs."
            speak_feedback(status_message)
            st.success("✅ Status announced!")
    
    else:
        st.markdown("""
        <div class="accessibility-status">
            <h3>⚠️ System Not Initialized</h3>
            <p>Please initialize the detection system first</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Actions Section
    st.markdown("""
    <div class="accessibility-card">
        <h2>⚡ Quick Actions</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 REFRESH STATUS", key="refresh_status", use_container_width=True,
                    help="Refresh system status with voice feedback"):
            speak_feedback("Refreshing system status")
            st.success("✅ Status refreshed!")
            st.rerun()
    
    with col2:
        if st.button("📸 TAKE SCREENSHOT", key="take_screenshot", use_container_width=True,
                    help="Take a screenshot with voice feedback"):
            speak_feedback("Taking screenshot")
            st.success("✅ Screenshot taken!")
    
    # Help Section
    with st.expander("ℹ️ Accessibility Help", expanded=False):
        st.markdown("""
        ### ♿ Accessibility Features
        
        **Voice Feedback**: Every action provides audio confirmation
        **Large Buttons**: Easy to see and click
        **Clear Labels**: Descriptive text for all controls
        **Status Announcements**: Voice updates on system status
        
        ### 🎯 How to Use
        
        1. **Start Detection**: Click the large green START button
        2. **Stop Detection**: Click the large red STOP button
        3. **Test Voice**: Click TEST VOICE to verify audio system
        4. **Force Speak**: Enter a message and click FORCE SPEAK
        5. **Check Status**: Click ANNOUNCE STATUS for voice summary
        
        ### 🔧 Tips for Visually Disabled Users
        
        - Use screen readers with this interface
        - All buttons have descriptive labels
        - Voice feedback confirms every action
        - Large touch targets for easy interaction
        """)

if __name__ == "__main__":
    if not DETECTION_AVAILABLE:
        st.error("❌ Detection system not available. Please ensure all dependencies are installed.")
        st.stop()
    
    main() 