# src/detectors/anomaly_detector.py
from ultralytics import YOLO
import numpy as np
import cv2
import time
from collections import defaultdict, deque

# Choose model: 'yolov8n.pt' (fast) or 'yolov8m.pt' (more accurate)
MODEL_WEIGHTS = "yolov8m.pt"   # try yolov8m for higher accuracy
CONF_THRESHOLD = 0.5           # raise to 0.6-0.7 if you still have FP
MIN_BBOX_AREA_RATIO = 0.01     # ignore boxes smaller than 1% of frame area
MIN_CONSECUTIVE_FRAMES = 3     # require N consecutive frames to confirm

model = YOLO(MODEL_WEIGHTS)

# Hazard keywords (will match substrings in class names)
HAZARD_KEYWORDS = {
    "knife", "gun", "fire", "smoke", "axe", "crowbar", "bat", "sword",
    "explosive", "bomb", "lighter", "chainsaw", "scissors", "hammer"
}

# Keep short history per label to require persistence
_histories = defaultdict(lambda: deque(maxlen=MIN_CONSECUTIVE_FRAMES))

def _is_hazard_label(label):
    lab = label.lower()
    for k in HAZARD_KEYWORDS:
        if k in lab:
            return True
    return False

def detect_anomalies(frame, conf_threshold=CONF_THRESHOLD):
    """
    Run model on frame and return confirmed hazards.
    Returns: (confirmed_bool, list_of_detections)
    where each detection is dict: {'label': str, 'conf': float, 'box': (x1,y1,x2,y2)}
    """
    h, w = frame.shape[:2]
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    found = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = model.names[cls_id]
            # simple hazard filter
            if not _is_hazard_label(label):
                continue
            # bbox coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = max(0, (x2 - x1) * (y2 - y1))
            if area < MIN_BBOX_AREA_RATIO * (w * h):
                # ignore too-small boxes
                continue
            found.append({'label': label.lower(), 'conf': conf, 'box': (x1, y1, x2, y2)})

    # Update histories and confirm those with persistence
    confirmed = []
    current_labels = [d['label'] for d in found]
    timestamp = time.time()

    # add counts for detected labels this frame
    for lbl in set(current_labels):
        _histories[lbl].append(1)
    # add zero for labels not seen this frame (so history shows gaps)
    for lbl in list(_histories.keys()):
        if lbl not in current_labels:
            _histories[lbl].append(0)

    # confirm labels that have >= MIN_CONSECUTIVE_FRAMES ones in history
    for lbl, dq in _histories.items():
        if sum(dq) >= MIN_CONSECUTIVE_FRAMES:
            # find the best detection for this label in 'found' list
            best = max((d for d in found if d['label'] == lbl), key=lambda x: x['conf'], default=None)
            if best:
                confirmed.append(best)

    # Optionally remove histories for labels unseen for long
    for lbl in list(_histories.keys()):
        if sum(_histories[lbl]) == 0 and len(_histories[lbl]) >= _histories[lbl].maxlen:
            del _histories[lbl]

    return (len(confirmed) > 0), confirmed

# Standalone test
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ok, detections = detect_anomalies(frame)
        if ok:
            for d in detections:
                lbl = d['label']
                conf = d['conf']
                x1,y1,x2,y2 = d['box']
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, f"{lbl} {int(conf*100)}%", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                print(f"ALERT: {lbl} {conf:.2f}")
        cv2.imshow("Anomaly Detector (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()