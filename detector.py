import cv2
import numpy as np
import time
import torch

class ISightDetector:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to('cpu')
        self.midas.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        self.zones = []
        self.zone_names = ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"]
        self.last_guidance_time = time.time()
        self.last_detail_time = time.time()
        self.voice = VoiceManager()  # Assuming VoiceManager exists from earlier logic

    def setup_detection_zones(self, frame_width, frame_height):
        zone_width = frame_width // 5
        self.zones = [(i * zone_width, 0, (i + 1) * zone_width, frame_height) for i in range(5)]

    def get_disparity(self, frame):
        small_frame = cv2.resize(frame, (320, 240))
        img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to('cpu')
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=small_frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        disparity_map = prediction.cpu().numpy()
        disparity_map = cv2.resize(disparity_map, (frame.shape[1], frame.shape[0]))
        disparity_map = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min() + 1e-6)
        return disparity_map

    def get_object_disparity(self, disparity_map, bbox):
        x1, y1, x2, y2 = map(int, bbox[:4])
        object_disparity = disparity_map[y1:y2, x1:x2]
        return object_disparity.mean()

    def detect_objects_in_zones(self, detections, disparity_map):
        zone_objects = {i: [] for i in range(5)}
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf > 0.5 and int(cls) in [0, 2, 3, 5, 7]:  # person, car, motorcycle, bus, truck
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                for i, (zx1, zy1, zx2, zy2) in enumerate(self.zones):
                    if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                        disparity = self.get_object_disparity(disparity_map, [x1, y1, x2, y2])
                        zone_objects[i].append((det, disparity))
                        break
        return zone_objects

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        if not self.zones:
            self.setup_detection_zones(frame.shape[1], frame.shape[0])

        # YOLO object detection
        results = self.yolo_model(frame, size=320)
        detections = results.xyxy[0]

        # Depth estimation
        disparity_map = self.get_disparity(frame)

        # Zone-based object detection
        zone_objects = self.detect_objects_in_zones(detections, disparity_map)

        # Navigation guidance (every 2 seconds)
        current_time = time.time()
        if current_time - self.last_guidance_time > 2:
            left_zones = [0, 1]
            right_zones = [3, 4]
            center_zone = 2
            near_threshold = 0.7  # Higher disparity means closer

            near_left = any(any(obj[1] > near_threshold for obj in zone_objects[i]) for i in left_zones)
            near_right = any(any(obj[1] > near_threshold for obj in zone_objects[i]) for i in right_zones)
            near_center = any(obj[1] > near_threshold for obj in zone_objects[center_zone])

            if near_left and not near_right:
                guidance = "Obstacle on the left, move right"
            elif near_right and not near_left:
                guidance = "Obstacle on the right, move left"
            elif near_center:
                guidance = "Obstacle ahead, stop or change direction"
            else:
                guidance = "Path is clear"
            self.voice.announce('guidance', guidance)
            self.last_guidance_time = current_time

        # Detailed object announcement (every 5 seconds)
        if current_time - self.last_detail_time > 5:
            for zone_idx, objects in zone_objects.items():
                near_objects = [obj for obj in objects if obj[1] > 0.7]
                if near_objects:
                    labels = [self.yolo_model.names[int(obj[0][5])] for obj in near_objects]
                    message = f"In {self.zone_names[zone_idx]} zone, near objects: {', '.join(labels)}"
                    self.voice.announce(f'zone_{zone_idx}_near', message)
            self.last_detail_time = current_time

        # Return frame for web interface (if needed)
        return True

    def release(self):
        self.cap.release()

# Placeholder for VoiceManager (assumed from earlier logic)
class VoiceManager:
    def announce(self, category, message):
        print(f"Voice Announcement [{category}]: {message}")  # Replace with actual TTS logic

# Example usage (integrates with existing Flask app)
if __name__ == "__main__":
    detector = ISightDetector()
    while True:
        if not detector.process_frame():
            break
    detector.release()