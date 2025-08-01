#!/usr/bin/env python3
"""
Quick Demo for PP-PicoDet-L Object Detection
============================================

This script provides a simple demo to test the object detection system.
It will start webcam detection with default settings.
"""

from object_detector import PP_PicoDet_Detector
import cv2
import time

def demo():
    print("PP-PicoDet-L Object Detection Demo")
    print("=" * 40)
    print("Model: PP-PicoDet-L (40.9% mAP, 3.3M parameters)")
    print("Features: 39 FPS on mobile ARM CPUs")
    print("=" * 40)
    
    # Initialize detector with default settings
    print("\nInitializing detector...")
    detector = PP_PicoDet_Detector(
        confidence_threshold=0.5,
        nms_threshold=0.4,
        device='auto'
    )
    
    print("\nStarting webcam detection...")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save screenshot")
    print("- Press 'r' to reset FPS counter")
    print("\nPress any key to start...")
    
    # Wait for user input
    input()
    
    # Start video processing
    detector.process_video(video_source=0)

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error during demo: {e}") 