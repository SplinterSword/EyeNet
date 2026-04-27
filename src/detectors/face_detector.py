# src/detectors/face_detector.py
"""Face detection and recognition extracted from the monolithic pipeline."""

import logging
import os
import pickle

import cv2
import face_recognition
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detect and recognize faces against enrolled student encodings."""

    def __init__(self, distance_threshold: float | None = None, encodings_path: str = "models/face_encodings.pkl"):
        self._threshold = distance_threshold or Config.FACE_DISTANCE_THRESHOLD
        self._known_rolls: list[str] = []
        self._known_encs: np.ndarray | None = None
        self._load_encodings(encodings_path)

    def _load_encodings(self, path: str):
        if not os.path.exists(path):
            logger.warning("Encodings file not found at %s — face recognition disabled", path)
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._known_rolls = list(data.keys())
        self._known_encs = np.array([data[r] for r in self._known_rolls])
        logger.info("Loaded face encodings for %d students", len(self._known_rolls))

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect faces in *frame* and match against known encodings.

        Returns a list of dicts:
            {'name': str, 'distance': float, 'box': (x1,y1,x2,y2), 'is_known': bool}
        """
        if self._known_encs is None or len(self._known_encs) == 0:
            return []

        # Down-scale for speed
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)

        locations = face_recognition.face_locations(small)
        encodings = face_recognition.face_encodings(small, locations)

        results = []
        for (top, right, bottom, left), enc in zip(locations, encodings):
            distances = np.linalg.norm(self._known_encs - enc, axis=1)
            idx = int(np.argmin(distances))
            min_dist = float(distances[idx])

            # Scale back to original resolution
            box = (left * 2, top * 2, right * 2, bottom * 2)  # (x1, y1, x2, y2)

            if min_dist < self._threshold:
                results.append({
                    "name": self._known_rolls[idx],
                    "distance": min_dist,
                    "box": box,
                    "is_known": True,
                })
            else:
                results.append({
                    "name": "unknown",
                    "distance": min_dist,
                    "box": box,
                    "is_known": False,
                })

        return results
