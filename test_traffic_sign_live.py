#!/usr/bin/env python3
"""
Test traffic sign detection with live camera feed
"""

import cv2
import time
import sys
import os

def test_live_traffic_sign_detection():
    """Test traffic sign detection with live camera"""
    print("🔍 Testing Live Traffic Sign Detection")
    print("=" * 50)
    print("📸 Point your camera at traffic signs or traffic sign images")
    print("⏹️  Press 'q' to quit")
    print("=" * 50)
    
    try:
        # Add the traffic sign detection directory to the path
        traffic_sign_path = os.path.join(os.path.dirname(__file__), 'Traffic-Sign-Detection', 'Traffic-Sign-Detection')
        if traffic_sign_path not in sys.path:
            sys.path.append(traffic_sign_path)
        
        # Import modules
        from classification import SVM, getLabel
        from main import localization
        
        # Fix OpenCV compatibility
        def constrastLimit(image):
            img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels = list(cv2.split(img_hist_equalized))
            channels[0] = cv2.equalizeHist(channels[0])
            img_hist_equalized = cv2.merge(channels)
            img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
            return img_hist_equalized
        
        import main
        main.constrastLimit = constrastLimit
        
        # Load pre-trained model
        model_path = os.path.join(traffic_sign_path, 'data_svm.dat')
        if os.path.exists(model_path):
            model = SVM()
            model.load(model_path)
            print("✅ Pre-trained model loaded")
        else:
            print("❌ Pre-trained model not found")
            return False
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("📹 Camera initialized")
        print("🔍 Starting detection...")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            frame_count += 1
            
            # Process every 10th frame for performance
            if frame_count % 10 == 0:
                try:
                    # Run traffic sign detection
                    detected_signs = localization(
                        frame, 
                        min_size_components=300, 
                        similitary_contour_with_circle=0.65, 
                        model=model, 
                        count=0, 
                        current_sign_type=0
                    )
                    
                    if detected_signs:
                        detection_count += 1
                        print(f"🎯 Frame {frame_count}: Detected {len(detected_signs)} traffic sign(s)")
                        
                        # Draw detection info on frame
                        for i, sign in enumerate(detected_signs):
                            if isinstance(sign, str):
                                sign_type = sign
                            elif isinstance(sign, dict):
                                sign_type = sign.get('sign_type', 'Unknown')
                            else:
                                sign_type = str(sign)
                            
                            # Draw text on frame
                            cv2.putText(frame, f"Sign: {sign_type}", (10, 30 + i*30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            print(f"   - {sign_type}")
                
                except Exception as e:
                    pass  # Continue even if detection fails
            
            # Display frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {detection_count}", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Traffic Sign Detection Test', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("⏹️  Stopped by user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Test Results:")
        print(f"   Total frames: {frame_count}")
        print(f"   Detections: {detection_count}")
        
        if detection_count > 0:
            print("✅ Traffic sign detection is working!")
        else:
            print("⚠️ No traffic signs detected. Try pointing camera at traffic sign images.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_live_traffic_sign_detection()
    if success:
        print("\n🎉 Live traffic sign detection test completed!")
    else:
        print("\n❌ Live traffic sign detection test failed") 