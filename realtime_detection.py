import cv2
import torch
import numpy as np
import time
from pathlib import Path
import sys

# Add the current directory to Python path to import YOLOv5 modules
sys.path.append(str(Path(__file__).parent))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

class VehicleDetector:
    def __init__(self, weights_path, device='', img_size=640, conf_thres=0.25, iou_thres=0.45):
        """
        Initialize the vehicle detector
        
        Args:
            weights_path: Path to the trained model weights
            device: Device to run inference on ('cpu' or '0' for GPU)
            img_size: Input image size
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
        """
        self.device = select_device(device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load model
        print(f"Loading model from {weights_path}...")
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.eval()
        
        # Get model stride
        self.stride = int(self.model.stride.max())
        self.img_size = self.check_img_size(self.img_size, s=self.stride)
        
        # Get names
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        print(f"Model loaded successfully. Classes: {self.names}")
        print(f"Using device: {self.device}")
    
    def check_img_size(self, img_size, s=32):
        """Verify img_size is a multiple of stride s"""
        new_size = self.make_divisible(img_size, int(s))
        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size
    
    def make_divisible(self, x, divisor):
        """Returns nearest x divisible by divisor"""
        return int(np.ceil(x / divisor) * divisor)
    
    def preprocess_image(self, img):
        """Preprocess image for model inference"""
        # Resize and pad image
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_resized = img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_resized = np.ascontiguousarray(img_resized)
        img_resized = torch.from_numpy(img_resized).to(self.device)
        img_resized = img_resized.float()
        img_resized /= 255.0
        if img_resized.ndimension() == 3:
            img_resized = img_resized.unsqueeze(0)
        
        return img_resized
    
    def detect(self, img):
        """
        Perform vehicle detection on input image
        
        Args:
            img: Input image (BGR format from OpenCV)
            
        Returns:
            img_with_boxes: Image with bounding boxes drawn
            detections: List of detections [x1, y1, x2, y2, conf, class]
        """
        # Store original image dimensions
        orig_h, orig_w = img.shape[:2]
        
        # Preprocess image
        img_tensor = self.preprocess_image(img)
        
        # Inference
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        detections = []
        img_with_boxes = img.copy()
        
        # Process detections
        if pred[0] is not None and len(pred[0]):
            # Scale coordinates back to original image size
            pred[0][:, :4] = scale_coords(img_tensor.shape[2:], pred[0][:, :4], img.shape).round()
            
            # Draw boxes and labels
            for *xyxy, conf, cls in pred[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                confidence = float(conf)
                
                # Only detect vehicles (assuming class 0 is vehicle based on your training)
                if class_id == 0:  # Vehicle class
                    # Draw bounding box
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f'Vehicle: {confidence:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img_with_boxes, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(img_with_boxes, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return img_with_boxes, detections
    
    def run_realtime(self, source=0):
        """
        Run real-time vehicle detection
        
        Args:
            source: Video source (0 for webcam, or video file path)
        """
        # Initialize video capture
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        print("Starting real-time vehicle detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Perform detection
            frame_with_boxes, detections = self.detect(frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0
            
            # Display FPS on frame
            cv2.putText(frame_with_boxes, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display vehicle count
            vehicle_count = len(detections)
            cv2.putText(frame_with_boxes, f'Vehicles: {vehicle_count}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Real-time Vehicle Detection', frame_with_boxes)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"vehicle_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_boxes)
                print(f"Screenshot saved as {filename}")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Configuration
    weights_path = "runs/train/exp12/weights/best.pt"  # Path to your trained weights
    device = ""  # Empty string for auto-detection, 'cpu' for CPU, '0' for GPU
    
    # Initialize detector
    detector = VehicleDetector(
        weights_path=weights_path,
        device=device,
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45
    )
    
    # Run real-time detection
    # Use 0 for webcam, or provide video file path
    detector.run_realtime(source=0)

if __name__ == "__main__":
    main() 