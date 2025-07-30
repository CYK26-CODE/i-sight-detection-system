#!/usr/bin/env python3
"""
i-sight Streamlit Application (Simplified Version)
Real-time computer vision detection system with voice feedback

GitHub Deployment Configuration:
- Main file: streamlit_app_simple.py
- Requirements: requirements_streamlit.txt
- Python version: 3.9+
- Streamlit Cloud compatible
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

# Initialize session state
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
    if 'last_process_time' not in st.session_state:
        st.session_state.last_process_time = 0
    if 'accessibility_enabled' not in st.session_state:
        st.session_state.accessibility_enabled = False

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

def process_single_frame():
    """Process a single frame and update session state"""
    if not st.session_state.detector or not st.session_state.detector.cap:
        return False
    
    try:
        # Process frame
        success = st.session_state.detector.process_frame()
        if success:
            # Update session state with latest data
            st.session_state.latest_frame = st.session_state.detector.get_screenshot()
            st.session_state.detection_data = st.session_state.detector.get_detection_data()
            st.session_state.last_process_time = time.time()
            return True
    except Exception as e:
        st.error(f"❌ Frame processing error: {e}")
    
    return False

def start_detection():
    """Start detection"""
    if not initialize_detector():
        return False
    
    st.session_state.detection_running = True
    return True

def stop_detection():
    """Stop detection"""
    try:
        st.session_state.detection_running = False
        
        # Cleanup detector
        if st.session_state.detector:
            st.session_state.detector.cleanup()
            st.session_state.detector = None
        
        # Clear session state
        st.session_state.latest_frame = None
        st.session_state.detection_data = {}
        
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
    if st.session_state.accessibility_enabled:
        speak_feedback("Starting detection system")
    return start_detection()

def accessibility_stop_detection():
    """Stop detection with accessibility feedback"""
    if st.session_state.accessibility_enabled:
        speak_feedback("Stopping detection system")
    stop_detection()

def accessibility_test_voice():
    """Test voice with accessibility feedback"""
    if st.session_state.accessibility_enabled:
        speak_feedback("Testing voice system")
    return test_voice()

def accessibility_force_speak(message):
    """Force speak with accessibility feedback"""
    if st.session_state.accessibility_enabled:
        speak_feedback(f"Force speaking: {message}")
    return force_speak(message)

def toggle_accessibility():
    """Toggle accessibility features on/off and control detection"""
    st.session_state.accessibility_enabled = not st.session_state.accessibility_enabled
    
    if st.session_state.accessibility_enabled:
        # Turn on accessibility and start detection
        speak_feedback("Accessibility features enabled, starting detection")
        start_detection()
    else:
        # Turn off accessibility and stop detection
        speak_feedback("Accessibility features disabled, stopping detection")
        stop_detection()

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
        
        # Process frame if detection is running
        if st.session_state.detection_running:
            # Process frame every 100ms (10 FPS for UI)
            current_time = time.time()
            if current_time - st.session_state.last_process_time > 0.1:
                process_single_frame()
        
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
                face_count = len(data.get('faces', []))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{face_count}</div>
                    <div class="metric-label">Faces</div>
                </div>
                """, unsafe_allow_html=True)
                
                vehicle_count = len(data.get('vehicles', []))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{vehicle_count}</div>
                    <div class="metric-label">Vehicles</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                traffic_count = len(data.get('traffic_signs', []))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{traffic_count}</div>
                    <div class="metric-label">Traffic Signs</div>
                </div>
                """, unsafe_allow_html=True)
                
                fps = data.get('fps', 0.0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{fps:.1f}</div>
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
    """Accessibility dashboard with only toggle button"""
    # Accessibility header
    st.markdown("""
    <div class="accessibility-header">
        <h1>♿ Accessibility Mode</h1>
        <p>Toggle Voice Feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Accessibility Toggle Section
    st.markdown("""
    <div class="accessibility-card">
        <h2>🎛️ Accessibility Toggle</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Large dynamic toggle button
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Dynamic toggle button with status
        is_enabled = st.session_state.accessibility_enabled
        button_text = "🟢 ON - ACCESSIBILITY ENABLED" if is_enabled else "🔴 OFF - ACCESSIBILITY DISABLED"
        button_color = "success" if is_enabled else "danger"
        
        st.markdown(f"""
        <style>
        .dynamic-toggle-btn {{
            background: {'linear-gradient(45deg, #56ab2f, #a8e6cf)' if is_enabled else 'linear-gradient(45deg, #ff416c, #ff4b2b)'};
            color: white;
            padding: 2rem 4rem;
            border: none;
            border-radius: 20px;
            font-size: 2rem;
            font-weight: bold;
            cursor: pointer;
            margin: 1rem 0;
            min-width: 400px;
            text-transform: uppercase;
            letter-spacing: 3px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            transition: all 0.4s ease;
            text-align: center;
            display: block;
            width: 100%;
        }}
        
        .dynamic-toggle-btn:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.4);
            filter: brightness(1.1);
        }}
        
        .dynamic-toggle-btn:active {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }}
        
        .status-indicator {{
            background: {'#56ab2f' if is_enabled else '#ff416c'};
            color: white;
            padding: 1rem 2rem;
            border-radius: 15px;
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 3px solid {'#a8e6cf' if is_enabled else '#ff4b2b'};
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Status indicator
        status_text = "🟢 SYSTEM ACTIVE - DETECTION RUNNING" if is_enabled else "🔴 SYSTEM INACTIVE - DETECTION STOPPED"
        st.markdown(f"""
        <div class="status-indicator">
            {status_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Large toggle button
        if st.button(button_text, key="toggle_accessibility", use_container_width=True,
                    help="Toggle accessibility features and detection system on/off"):
            toggle_accessibility()
            st.rerun()

if __name__ == "__main__":
    if not DETECTION_AVAILABLE:
        st.error("❌ Detection system not available. Please ensure all dependencies are installed.")
        st.stop()
    
    main() 