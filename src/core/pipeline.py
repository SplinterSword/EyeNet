# src/core/pipeline.py
"""Detection pipeline orchestrator — runs in its own thread."""

import logging
import os
import threading
import time

import cv2

from src.config import Config
from src.core.event_bus import DetectionEvent, EventBus, Severity
from src.core.frame_buffer import SharedFrameBuffer
from src.core.tracker import ObjectTracker
from src.core.database import insert_metrics, prune_old_metrics
from src.detectors.face_detector import FaceDetector
from src.detectors.anomaly_detector import detect_anomalies

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Orchestrates face recognition, hazard detection, and object tracking.

    Reads frames from a SharedFrameBuffer, runs detectors, updates the
    tracker, and publishes DetectionEvents to the EventBus.
    """

    def __init__(self, frame_buffer: SharedFrameBuffer, event_bus: EventBus):
        self.frame_buffer = frame_buffer
        self.event_bus = event_bus
        self.tracker = ObjectTracker(frame_rate=Config.TARGET_FPS)
        self.face_detector = FaceDetector()
        self._running = False
        self._last_frame_id = -1
        self._thread: threading.Thread | None = None
        self._metrics_thread: threading.Thread | None = None

        # Live metrics (read by dashboard)
        self.metrics = {
            "fps": 0.0,
            "detections": 0,
            "faces": 0,
            "processing_ms": 0.0,
        }

    # ── public API ──────────────────────────────────────────────────────
    def start(self):
        """Start detection and metrics threads."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="detection-pipeline")
        self._thread.start()
        self._metrics_thread = threading.Thread(target=self._metrics_writer, daemon=True, name="metrics-writer")
        self._metrics_thread.start()
        logger.info("DetectionPipeline started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._metrics_thread:
            self._metrics_thread.join(timeout=3)
        logger.info("DetectionPipeline stopped.")

    # ── main loop ───────────────────────────────────────────────────────
    def _run(self):
        while self._running:
            frame_id, frame = self.frame_buffer.read()
            if frame is None or frame_id == self._last_frame_id:
                time.sleep(0.005)
                continue
            self._last_frame_id = frame_id

            t0 = time.monotonic()
            all_detections: list[dict] = []

            # 1. Face detection + recognition
            try:
                faces = self.face_detector.detect(frame)
                for face in faces:
                    label = f"face:{face['name']}" if face["is_known"] else "unknown_face"
                    all_detections.append({
                        "label": label,
                        "conf": max(0.0, 1.0 - face["distance"]),
                        "box": face["box"],
                        "_face": face,  # carry original data for uniform check etc.
                    })
            except Exception:
                logger.exception("Face detection error")

            # 2. Hazard detection
            try:
                hazard_found, hazards = detect_anomalies(frame)
                for h in hazards:
                    all_detections.append({
                        "label": f"hazard:{h['label']}",
                        "conf": h["conf"],
                        "box": h["box"],
                    })
            except Exception:
                logger.exception("Hazard detection error")

            # 3. Track all detections
            tracked = self.tracker.update(all_detections, frame_id)

            # 4. Emit events for new persistent tracks
            for track in tracked:
                if self.tracker.should_alert(track):
                    severity = self._classify_severity(track.label)
                    duration = track.last_seen_frame - track.first_seen_frame

                    # Save snapshot
                    image_path = ""
                    try:
                        os.makedirs("data/anomalies", exist_ok=True)
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        image_path = f"data/anomalies/{track.label.replace(':', '_')}_{ts}.jpg"
                        cv2.imwrite(image_path, frame)
                    except Exception:
                        logger.exception("Failed to save snapshot")

                    self.event_bus.publish(DetectionEvent(
                        event_type=track.label.split(":")[0],
                        severity=severity,
                        description=f"{track.label} (track {track.track_id})",
                        track_id=track.track_id,
                        image_path=image_path,
                        metadata={
                            "confidence": track.confidence,
                            "duration_frames": duration,
                            "item_name": track.label.split(":")[-1] if ":" in track.label else track.label,
                        },
                    ))

            elapsed_ms = (time.monotonic() - t0) * 1000
            self.metrics.update({
                "fps": self.frame_buffer.fps,
                "detections": len(all_detections),
                "faces": sum(1 for d in all_detections if "face" in d["label"]),
                "processing_ms": round(elapsed_ms, 1),
            })

    # ── severity classification ─────────────────────────────────────────
    @staticmethod
    def _classify_severity(label: str) -> Severity:
        label_lower = label.lower()
        if any(k in label_lower for k in ("gun", "fire", "bomb", "explosive")):
            return Severity.CRITICAL
        if any(k in label_lower for k in ("knife", "smoke", "axe", "sword")):
            return Severity.HIGH
        if "unknown" in label_lower:
            return Severity.MEDIUM
        return Severity.LOW

    # ── periodic metrics writer ─────────────────────────────────────────
    def _metrics_writer(self):
        while self._running:
            time.sleep(5)
            try:
                m = self.metrics
                insert_metrics(m["fps"], m["detections"], m["faces"], m["processing_ms"])
                prune_old_metrics(hours=24)
            except Exception:
                logger.exception("Metrics write error")
