#!/usr/bin/env python3
"""
Test script for YOLO11n integration
"""

import cv2
import time
import sys

def test_yolo11n():
    """Test YOLO11n functionality"""
    print("🔍 Testing YOLO11n Integration")
    print("=" * 50)
    
    # Test ultralytics import
    try:
        from ultralytics import YOLO
        print("✅ ultralytics imported successfully")
    except ImportError as e:
        print(f"❌ ultralytics import failed: {e}")
        print("💡 Install with: pip install ultralytics")
        return False
    
    # Test YOLO11n model loading
    try:
        print("🔄 Loading YOLO11n model...")
        model = YOLO('yolo11n.pt')
        print("✅ YOLO11n model loaded successfully")
        print(f"   - Model size: ~5.2MB")
        print(f"   - Parameters: ~2.6M")
        print(f"   - mAP: 39.5%")
    except Exception as e:
        print(f"❌ YOLO11n model loading failed: {e}")
        return False
    
    # Test camera access
    try:
        print("🔄 Testing camera access...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not available")
            return False
        
        print("✅ Camera accessed successfully")
        
        # Test YOLO11n inference
        print("🔄 Testing YOLO11n inference...")
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            cap.release()
            return False
        
        print(f"✅ Frame captured: {frame.shape}")
        
        # Run YOLO11n detection
        start_time = time.time()
        results = model(frame, conf=0.25, iou=0.45, verbose=False)
        inference_time = time.time() - start_time
        
        print(f"✅ YOLO11n inference completed in {inference_time:.3f} seconds")
        
        # Process results
        detection_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                detection_count = len(boxes)
                print(f"✅ Detected {detection_count} objects")
                
                # Show detected classes
                if detection_count > 0:
                    classes = []
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        confidence = box.conf[0].cpu().numpy()
                        classes.append(f"{class_name} ({confidence:.2f})")
                    
                    print(f"   Objects: {', '.join(classes)}")
        
        cap.release()
        print("✅ YOLO11n test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ YOLO11n test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting YOLO11n test...")
    success = test_yolo11n()
    
    if success:
        print("\n✅ YOLO11n integration test PASSED!")
        print("🎉 Ready to use YOLO11n in i-sight system")
    else:
        print("\n❌ YOLO11n integration test FAILED!")
        print("💡 Check the error messages above")
    
    input("\nPress Enter to exit...") 