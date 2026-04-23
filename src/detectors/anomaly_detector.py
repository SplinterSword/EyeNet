# src/detectors/anomaly_detector.py
"""YOLOv8 hazard detection with per-class confidence thresholds and temporal filtering."""

import logging
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import Config

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────
MODEL_WEIGHTS = Config.YOLO_HAZARD_MODEL
MIN_BBOX_AREA_RATIO = 0.01
MIN_CONSECUTIVE_FRAMES = Config.HAZARD_CONSECUTIVE_FRAMES

# Per-class confidence thresholds (replaces flat CONF_THRESHOLD)
CLASS_THRESHOLDS: dict[str, float] = Config.HAZARD_CLASS_THRESHOLDS
DEFAULT_THRESHOLD: float = Config.HAZARD_CONF_THRESHOLD

# Hazard keywords (matched as substrings in YOLO class names)
HAZARD_KEYWORDS = {
    "knife", "gun", "fire", "smoke", "axe", "crowbar", "bat", "sword",
    "explosive", "bomb", "lighter", "chainsaw", "scissors", "hammer",
}

# ── Model loading (singleton) ──────────────────────────────────────────
_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        logger.info("Loading hazard model: %s", MODEL_WEIGHTS)
        _model = YOLO(MODEL_WEIGHTS)
    return _model


# ── Temporal histories ──────────────────────────────────────────────────
_histories: dict[str, deque] = defaultdict(lambda: deque(maxlen=MIN_CONSECUTIVE_FRAMES))


def _is_hazard_label(label: str) -> bool:
    lab = label.lower()
    return any(k in lab for k in HAZARD_KEYWORDS)


def _get_threshold(label: str) -> float:
    """Return the per-class confidence threshold, or the default."""
    lab = label.lower()
    for key, thresh in CLASS_THRESHOLDS.items():
        if key in lab:
            return thresh
    return DEFAULT_THRESHOLD


# ── Public API ──────────────────────────────────────────────────────────
def detect_anomalies(frame: np.ndarray) -> tuple[bool, list[dict]]:
    """Run hazard detection on *frame*.

    Returns ``(confirmed_bool, list_of_detections)`` where each detection
    is ``{'label': str, 'conf': float, 'box': (x1, y1, x2, y2)}``.
    """
    model = _get_model()
    h, w = frame.shape[:2]

    results = model.predict(source=frame, conf=DEFAULT_THRESHOLD * 0.8, verbose=False)
    found: list[dict] = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = model.names[cls_id]

            if not _is_hazard_label(label):
                continue

            # Per-class confidence threshold
            if conf < _get_threshold(label):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = max(0, (x2 - x1) * (y2 - y1))
            if area < MIN_BBOX_AREA_RATIO * (w * h):
                continue

            found.append({"label": label.lower(), "conf": conf, "box": (x1, y1, x2, y2)})

    # ── Temporal filtering ──────────────────────────────────────────────
    current_labels = {d["label"] for d in found}

    for lbl in current_labels:
        _histories[lbl].append(1)
    for lbl in list(_histories.keys()):
        if lbl not in current_labels:
            _histories[lbl].append(0)

    confirmed: list[dict] = []
    for lbl, dq in _histories.items():
        if sum(dq) >= MIN_CONSECUTIVE_FRAMES:
            best = max((d for d in found if d["label"] == lbl), key=lambda x: x["conf"], default=None)
            if best:
                confirmed.append(best)

    # Prune dead histories
    for lbl in list(_histories.keys()):
        if sum(_histories[lbl]) == 0 and len(_histories[lbl]) >= _histories[lbl].maxlen:
            del _histories[lbl]

    return (len(confirmed) > 0), confirmed