#!/usr/bin/env python3
"""
Test Script for PP-PicoDet-L Object Detection System
===================================================

This script tests the object detection system with various inputs
to ensure proper functionality.
"""

import cv2
import numpy as np
import os
import time
from object_detector import PP_PicoDet_Detector

def create_test_image():
    """Create a simple test image with basic shapes"""
    # Create a 640x640 test image
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Draw some basic shapes to simulate objects
    # Rectangle (simulating a person)
    cv2.rectangle(image, (100, 100), (200, 400), (255, 255, 255), -1)
    
    # Circle (simulating a ball)
    cv2.circle(image, (400, 300), 80, (0, 255, 0), -1)
    
    # Triangle (simulating a sign)
    pts = np.array([[500, 100], [600, 200], [500, 300]], np.int32)
    cv2.fillPoly(image, [pts], (0, 0, 255))
    
    return image

def test_detector_initialization():
    """Test detector initialization"""
    print("Testing detector initialization...")
    
    try:
        detector = PP_PicoDet_Detector(
            confidence_threshold=0.5,
            nms_threshold=0.4,
            device='auto'
        )
        print("✓ Detector initialized successfully")
        return detector
    except Exception as e:
        print(f"✗ Detector initialization failed: {e}")
        return None

def test_image_processing(detector):
    """Test image processing functionality"""
    print("\nTesting image processing...")
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Save test image
        cv2.imwrite("test_image.jpg", test_image)
        print("✓ Test image created")
        
        # Process the image
        detections = detector.detect_objects(test_image)
        print(f"✓ Image processing completed. Found {len(detections)} detections")
        
        # Draw detections
        result_image = detector.draw_detections(test_image, detections)
        cv2.imwrite("test_result.jpg", result_image)
        print("✓ Result image saved")
        
        return True
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        return False

def test_video_processing(detector):
    """Test video processing functionality"""
    print("\nTesting video processing...")
    
    try:
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_video.mp4', fourcc, 10.0, (640, 640))
        
        # Create 30 frames of test video
        for i in range(30):
            frame = create_test_image()
            # Add some movement
            cv2.circle(frame, (100 + i*5, 300), 50, (255, 0, 0), -1)
            out.write(frame)
        
        out.release()
        print("✓ Test video created")
        
        # Process a few frames
        cap = cv2.VideoCapture('test_video.mp4')
        frame_count = 0
        max_frames = 5
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = detector.detect_objects(frame)
            frame_count += 1
            
        cap.release()
        print(f"✓ Video processing completed. Processed {frame_count} frames")
        
        # Cleanup
        if os.path.exists('test_video.mp4'):
            os.remove('test_video.mp4')
        
        return True
    except Exception as e:
        print(f"✗ Video processing failed: {e}")
        return False

def test_performance(detector):
    """Test performance metrics"""
    print("\nTesting performance...")
    
    try:
        test_image = create_test_image()
        
        # Measure inference time
        start_time = time.time()
        detections = detector.detect_objects(test_image)
        inference_time = time.time() - start_time
        
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        print(f"✓ Inference time: {inference_time:.3f}s")
        print(f"✓ Estimated FPS: {fps:.1f}")
        print(f"✓ Detections found: {len(detections)}")
        
        return True
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_files = ['test_image.jpg', 'test_result.jpg']
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Cleaned up {file}")

def run_all_tests():
    """Run all tests"""
    print("PP-PicoDet-L Object Detection System Tests")
    print("=" * 50)
    
    # Test 1: Initialization
    detector = test_detector_initialization()
    if detector is None:
        print("\n❌ Initialization failed. Stopping tests.")
        return False
    
    # Test 2: Image Processing
    image_test_passed = test_image_processing(detector)
    
    # Test 3: Video Processing
    video_test_passed = test_video_processing(detector)
    
    # Test 4: Performance
    performance_test_passed = test_performance(detector)
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Initialization: {'✓ PASS' if detector else '✗ FAIL'}")
    print(f"Image Processing: {'✓ PASS' if image_test_passed else '✗ FAIL'}")
    print(f"Video Processing: {'✓ PASS' if video_test_passed else '✗ FAIL'}")
    print(f"Performance: {'✓ PASS' if performance_test_passed else '✗ FAIL'}")
    
    all_passed = detector and image_test_passed and video_test_passed and performance_test_passed
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests() 