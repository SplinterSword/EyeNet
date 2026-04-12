# src/utils/yolo_uniform_detector.py
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")  # lightweight model

def is_wearing_blue_uniform(frame):
    """
    Uses YOLO to detect people and checks if their clothing is blue.
    Returns True if uniform detected, else False.
    """
    results = model.predict(frame, conf=0.5, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                # Get bounding box of the detected person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]

                # Convert to HSV for color detection
                hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([90, 50, 50])
                upper_blue = np.array([130, 255, 255])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                blue_ratio = np.sum(mask > 0) / mask.size

                if blue_ratio > 0.05:
                    return True  # Wearing blue uniform
    return False  # No blue uniform detected