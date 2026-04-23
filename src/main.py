# src/main.py
"""EyeNet entrypoint — wires all components together and starts the system."""

import logging
import os
import signal
import sys
import threading

from src.config import Config
from src.core.logging_config import setup_logging

# Set up logging before any other imports
setup_logging()
logger = logging.getLogger(__name__)

from src.core.database import init_db, insert_alert
from src.core.event_bus import DetectionEvent, EventBus, Severity
from src.core.cooldown import CooldownManager
from src.core.anomaly_scorer import compute_anomaly_score
from src.core.frame_buffer import SharedFrameBuffer
from src.core.pipeline import DetectionPipeline
from src.notifications.sms_sender import send_unknown_face_alert, send_dangerous_item_alert
from src.notifications.email_sender import send_uniform_violation_email


def main():
    logger.info("=" * 60)
    logger.info("  EyeNet Campus Surveillance — Starting")
    logger.info("=" * 60)

    # ── 1. Initialize database ──────────────────────────────────────────
    init_db()

    # ── 2. Shared frame buffer ──────────────────────────────────────────
    frame_buffer = SharedFrameBuffer(
        source=Config.CAMERA_SOURCE,
        width=Config.FRAME_WIDTH,
        height=Config.FRAME_HEIGHT,
    ).start()

    # ── 3. Event bus ────────────────────────────────────────────────────
    event_bus = EventBus(num_workers=2)
    cooldown = CooldownManager()

    # Handler: persist to database
    def db_handler(event: DetectionEvent):
        score = compute_anomaly_score(event)
        insert_alert(
            event_type=event.event_type,
            severity=int(event.severity),
            description=event.description,
            track_id=event.track_id,
            image_path=event.image_path,
            metadata={**event.metadata, "anomaly_score": score},
        )
        logger.info(
            "DB: %s severity=%s score=%.1f track=%s",
            event.event_type, event.severity.name, score, event.track_id,
        )

    # Handler: dispatch notifications (SMS / email)
    def alert_handler(event: DetectionEvent):
        key = event.cooldown_key
        if not cooldown.should_fire(key, event.severity):
            return

        score = compute_anomaly_score(event)
        # Only send external alerts for score >= 25
        if score < 25:
            return

        if event.event_type == "unknown_face":
            send_unknown_face_alert(
                location="Main Campus",
                image_url=event.image_path,
            )
        elif event.event_type == "hazard":
            item = event.metadata.get("item_name", "unknown")
            conf = event.metadata.get("confidence", 0) * 100
            send_dangerous_item_alert(
                item_name=item,
                confidence=conf,
                location="Main Campus",
                image_url=event.image_path,
            )
        elif event.event_type == "face" and "uniform" in event.description.lower():
            # Uniform violations for known students
            student_roll = event.metadata.get("item_name", "unknown")
            if student_roll != "unknown":
                from datetime import datetime
                send_uniform_violation_email(
                    student_roll=student_roll,
                    student_email=f"{student_roll}@mail.jiit.ac.in",
                    violation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    fine_amount=700,
                )

    # Handler: push to dashboard SSE
    def sse_handler(event: DetectionEvent):
        try:
            from src.dashboard.app import push_event
            push_event(event.to_dict())
        except Exception:
            pass  # Dashboard may not be running

    event_bus.register(db_handler)
    event_bus.register(alert_handler)
    event_bus.register(sse_handler)
    event_bus.start()

    # ── 4. Detection pipeline ───────────────────────────────────────────
    pipeline = DetectionPipeline(frame_buffer, event_bus)
    pipeline.start()

    # ── 5. Start Flask dashboard in a separate thread ──────────────────
    # Share the frame buffer so the dashboard reads from the same camera
    from src.dashboard.app import app, set_frame_buffer
    set_frame_buffer(frame_buffer)

    def run_dashboard():
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True, name="flask-dashboard")
    dashboard_thread.start()
    logger.info("Dashboard available at http://localhost:5000")

    # ── 6. Graceful shutdown ────────────────────────────────────────────
    shutdown_event = threading.Event()

    def _signal_handler(signum, frame):
        logger.info("Received signal %s — shutting down...", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("System running. Press Ctrl+C to stop.")
    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        pass

    # Cleanup
    pipeline.stop()
    event_bus.stop()
    frame_buffer.stop()
    logger.info("EyeNet stopped. Goodbye.")


if __name__ == "__main__":
    main()
