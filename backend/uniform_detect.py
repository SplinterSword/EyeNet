import cv2
import numpy as np
from typing import Tuple

class UniformDetector:
    def __init__(self):
        # Define blue color range in HSV (Hue, Saturation, Value)
        self.lower_blue = np.array([90, 50, 50])     # Lower range of blue in HSV
        self.upper_blue = np.array([130, 255, 255])  # Upper range of blue in HSV
        self.min_shirt_area = 1000  # Minimum area to be considered as a shirt

    def detect_uniform(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]) -> bool:
        """
        Detect if a blue shirt is present below the face rectangle
        
        Args:
            frame: Input BGR image
            face_rect: Tuple of (x, y, w, h) for the face rectangle
        
        Returns:
            bool: True if uniform (blue shirt) is detected, False otherwise
        """
        x, y, w, h = face_rect
        
        # Define region of interest (ROI) below the face
        shirt_y = y + h  # Start from bottom of face
        shirt_h = int(h * 1.5)  # Look 1.5x face height below the face
        shirt_roi = frame[shirt_y:shirt_y + shirt_h, x:x + w]
        
        if shirt_roi.size == 0:
            return False
            
        # Convert ROI to HSV color space
        hsv = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue color
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contour has significant area
        for contour in contours:
            if cv2.contourArea(contour) > self.min_shirt_area:
                return True
                
        return False

def test_uniform_detection():
    # Initialize detector
    detector = UniformDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Check for uniform
            has_uniform = detector.detect_uniform(frame, (x, y, w, h))
            
            # Display result
            label = "Uniform: Yes" if has_uniform else "Uniform: No"
            color = (0, 255, 0) if has_uniform else (0, 0, 255)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show the frame
        cv2.imshow('Uniform Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_uniform_detection()