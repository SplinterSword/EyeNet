# src/detectors/uniform_detector.py
"""Uniform compliance detector — checks if a person is wearing a light blue button-up shirt.

Uses HSV color thresholding on the torso region below a detected face bounding box.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── HSV range for "light blue" fabric ───────────────────────────────────
# Tight range targeting the specific muted steel-blue / dusty teal uniform:
#   H: 95-115  (blue family, excludes green and purple)
#   S: 30-100  (muted, not vivid — matches fabric texture)
#   V: 100-200 (medium brightness, excludes navy and white)
LIGHT_BLUE_LOWER = np.array([95, 30, 100])
LIGHT_BLUE_UPPER = np.array([115, 100, 200])

# Minimum percentage of the torso ROI that must be light blue to count as "wearing uniform"
UNIFORM_PIXEL_THRESHOLD = 0.25  # 25% of torso region must be light blue


def check_uniform(frame: np.ndarray, face_box: tuple, frame_shape: tuple = None) -> dict:
    """Check if the person below *face_box* is wearing a light blue shirt.

    Args:
        frame: Full BGR frame.
        face_box: (x1, y1, x2, y2) of the face bounding box.
        frame_shape: Optional (h, w) of frame; derived from frame if not given.

    Returns:
        {
            "wearing_uniform": bool,
            "confidence": float,   # fraction of torso pixels that are light blue
            "torso_box": (x1, y1, x2, y2),
        }
    """
    h, w = frame.shape[:2] if frame_shape is None else frame_shape
    fx1, fy1, fx2, fy2 = face_box
    face_w = fx2 - fx1
    face_h = fy2 - fy1

    # Estimate torso region: below face, roughly 2× face height, 1.5× face width (centered)
    torso_x1 = max(0, fx1 - int(face_w * 0.25))
    torso_x2 = min(w, fx2 + int(face_w * 0.25))
    torso_y1 = fy2  # starts right below the face
    torso_y2 = min(h, fy2 + int(face_h * 2.5))

    # Safety: ensure ROI is valid
    if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
        return {"wearing_uniform": False, "confidence": 0.0, "torso_box": (torso_x1, torso_y1, torso_x2, torso_y2)}

    torso_roi = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if torso_roi.size == 0:
        return {"wearing_uniform": False, "confidence": 0.0, "torso_box": (torso_x1, torso_y1, torso_x2, torso_y2)}

    # Convert to HSV and create mask for light blue
    hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LIGHT_BLUE_LOWER, LIGHT_BLUE_UPPER)

    # Apply mild morphological close to fill gaps in the shirt fabric
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = mask.shape[0] * mask.shape[1]
    blue_pixels = cv2.countNonZero(mask)
    ratio = blue_pixels / total_pixels if total_pixels > 0 else 0.0

    wearing = ratio >= UNIFORM_PIXEL_THRESHOLD

    return {
        "wearing_uniform": wearing,
        "confidence": round(ratio, 3),
        "torso_box": (torso_x1, torso_y1, torso_x2, torso_y2),
    }
