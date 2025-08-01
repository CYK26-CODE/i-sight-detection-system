#!/usr/bin/env python3
"""
PP-PicoDet-L Object Detection System
====================================

This script demonstrates the PP-PicoDet-L object detection system using OpenCV.
Features:
- 40.9% mAP with only 3.3M parameters
- 39 FPS on mobile ARM CPUs
- Optimized for resource-constrained environments

Usage Examples:
1. Webcam detection: python main.py --webcam
2. Video file detection: python main.py --video path/to/video.mp4
3. Image detection: python main.py --image path/to/image.jpg
4. Save output video: python main.py --video input.mp4 --output output.mp4
"""

import argparse
import sys
import os
from object_detector import PP_PicoDet_Detector

def main():
    parser = argparse.ArgumentParser(
        description="PP-PicoDet-L Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --webcam                    # Use webcam
  python main.py --video input.mp4          # Process video file
  python main.py --image photo.jpg          # Process single image
  python main.py --video input.mp4 --output result.mp4  # Save output video
        """
    )
    
    # Input source options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--webcam', action='store_true', 
                           help='Use webcam for real-time detection')
    input_group.add_argument('--video', type=str, 
                           help='Path to input video file')
    input_group.add_argument('--image', type=str, 
                           help='Path to input image file')
    
    # Output options
    parser.add_argument('--output', type=str, 
                       help='Path to save output video/image')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--nms', type=float, default=0.4,
                       help='NMS threshold (default: 0.4)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on (default: auto)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to custom model weights')
    
    args = parser.parse_args()
    
    # Validate input files
    if args.video and not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        sys.exit(1)
    
    if args.image and not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        sys.exit(1)
    
    # Initialize detector
    print("Initializing PP-PicoDet-L Object Detector...")
    print("=" * 50)
    print("Model Characteristics:")
    print("- 40.9% mAP with only 3.3M parameters")
    print("- 39 FPS on mobile ARM CPUs")
    print("- Optimized for resource-constrained environments")
    print("=" * 50)
    
    try:
        detector = PP_PicoDet_Detector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
            device=args.device
        )
        
        # Process based on input type
        if args.webcam:
            print("\nStarting webcam detection...")
            print("Controls:")
            print("- Press 'q' to quit")
            print("- Press 's' to save screenshot")
            detector.process_video(video_source=0, output_path=args.output)
            
        elif args.video:
            print(f"\nProcessing video: {args.video}")
            if args.output:
                print(f"Output will be saved to: {args.output}")
            detector.process_video(video_source=args.video, output_path=args.output)
            
        elif args.image:
            print(f"\nProcessing image: {args.image}")
            output_path = args.output if args.output else "output_image.jpg"
            result = detector.process_image(args.image, output_path)
            
            if result is not None:
                print(f"Image processing completed. Result saved to: {output_path}")
            else:
                print("Error processing image")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 