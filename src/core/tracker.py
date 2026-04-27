# src/core/tracker.py
"""Lightweight object tracker using the supervision ByteTrack implementation."""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

try:
    import supervision as sv
    _HAS_SUPERVISION = True
except ImportError:
    _HAS_SUPERVISION = False
    logger.warning("supervision not installed — tracker will pass-through detections without tracking")


@dataclass
class TrackedObject:
    """A detection with a persistent track ID across frames."""
    track_id: int
    label: str                 # e.g. "face:22102027", "hazard:knife", "unknown_face"
    bbox: tuple                # (x1, y1, x2, y2)
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    alert_sent: bool = False   # prevents duplicate alerts per track
    metadata: dict = field(default_factory=dict)


class ObjectTracker:
    """Wraps supervision.ByteTrack to give persistent IDs to detections."""

    def __init__(self, frame_rate: int = 30, lost_buffer: int = 30):
        if _HAS_SUPERVISION:
            self._tracker = sv.ByteTrack(
                track_activation_threshold=0.4,
                lost_track_buffer=lost_buffer,
                minimum_matching_threshold=0.8,
                frame_rate=frame_rate,
            )
        else:
            self._tracker = None
        self.tracks: dict[int, TrackedObject] = {}
        self._next_id = 0  # fallback counter when supervision unavailable

    def update(self, detections: list[dict], frame_id: int) -> list[TrackedObject]:
        """Update tracker with new detections and return tracked objects.

        Args:
            detections: [{'label': str, 'conf': float, 'box': (x1,y1,x2,y2)}]
            frame_id: monotonically increasing frame counter
        """
        if not detections:
            if self._tracker:
                self._tracker.update_with_detections(sv.Detections.empty())
            return []

        labels = [d["label"] for d in detections]
        boxes = np.array([d["box"] for d in detections], dtype=np.float32)
        confs = np.array([d["conf"] for d in detections], dtype=np.float32)

        if self._tracker is not None:
            sv_dets = sv.Detections(
                xyxy=boxes,
                confidence=confs,
                class_id=np.array([hash(l) % 10000 for l in labels], dtype=int),
            )
            tracked = self._tracker.update_with_detections(sv_dets)
            track_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        else:
            # Fallback: assign new IDs to each detection (no real tracking)
            track_ids = list(range(self._next_id, self._next_id + len(detections)))
            self._next_id += len(detections)

        results = []
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if tid not in self.tracks:
                self.tracks[tid] = TrackedObject(
                    track_id=tid,
                    label=labels[min(i, len(labels) - 1)],
                    bbox=tuple(boxes[i].astype(int)) if i < len(boxes) else (0, 0, 0, 0),
                    confidence=float(confs[i]) if i < len(confs) else 0.0,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                )
            else:
                t = self.tracks[tid]
                t.last_seen_frame = frame_id
                if i < len(boxes):
                    t.bbox = tuple(boxes[i].astype(int))
                if i < len(confs):
                    t.confidence = float(confs[i])
            results.append(self.tracks[tid])

        # Prune tracks not seen for a long time
        stale = [k for k, v in self.tracks.items()
                 if frame_id - v.last_seen_frame > 300]
        for k in stale:
            del self.tracks[k]

        return results

    def should_alert(self, track: TrackedObject, min_frames: int = 5) -> bool:
        """Only alert once per track, and only after min_frames of persistence."""
        if track.alert_sent:
            return False
        duration = track.last_seen_frame - track.first_seen_frame
        if duration >= min_frames:
            track.alert_sent = True
            return True
        return False
