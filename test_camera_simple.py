#!/usr/bin/env python3
"""
Simple camera test script to verify camera functionality
"""

import cv2
import time
import sys

def test_camera():
    """Test camera functionality"""
    print("🔍 Testing Camera Functionality")
    print("=" * 50)
    
    # Try different camera backends
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]
    camera_index = 0
    
    for backend in backends:
        try:
            print(f"🔍 Trying camera backend: {backend}")
            cap = cv2.VideoCapture(camera_index, backend)
            
            if cap.isOpened():
                print(f"✅ Camera opened with backend {backend}")
                
                # Set minimal properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Test reading frames
                frame_count = 0
                start_time = time.time()
                
                print("📸 Testing frame capture (5 seconds)...")
                while frame_count < 150:  # 5 seconds at 30fps
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_count += 1
                        if frame_count % 30 == 0:  # Print every second
                            elapsed = time.time() - start_time
                            fps = frame_count / elapsed
                            mean_brightness = frame.mean()
                            print(f"   Frame {frame_count}: {frame.shape}, brightness: {mean_brightness:.2f}, FPS: {fps:.1f}")
                    else:
                        print(f"❌ Failed to read frame {frame_count}")
                        break
                
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"✅ Camera test completed:")
                print(f"   - Frames captured: {frame_count}")
                print(f"   - Time elapsed: {elapsed:.2f} seconds")
                print(f"   - Actual FPS: {actual_fps:.1f}")
                print(f"   - Frame size: {frame.shape if frame is not None else 'None'}")
                
                cap.release()
                return True
                
            else:
                print(f"❌ Camera initialization failed with backend {backend}")
                cap.release()
                
        except Exception as e:
            print(f"❌ Camera backend {backend} error: {e}")
            if 'cap' in locals():
                cap.release()
            continue
    
    # Try without specific backend
    print("🔄 Trying camera without specific backend...")
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print("✅ Camera opened without specific backend")
            
            # Test reading a few frames
            for i in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   Frame {i+1}: {frame.shape}, brightness: {frame.mean():.2f}")
                else:
                    print(f"❌ Failed to read frame {i+1}")
                    break
            
            cap.release()
            return True
        else:
            print("❌ Camera initialization failed completely")
            return False
            
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting camera test...")
    success = test_camera()
    
    if success:
        print("\n✅ Camera test PASSED - camera is working properly!")
        print("ℹ️ Your camera driver settings are preserved")
    else:
        print("\n❌ Camera test FAILED - check camera connection and permissions")
        print("💡 Try:")
        print("   - Check if camera is connected")
        print("   - Allow camera access in Windows settings")
        print("   - Close other applications using the camera")
        print("   - Restart your computer")
    
    input("\nPress Enter to exit...") 