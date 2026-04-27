# src/dashboard/app.py
"""Flask dashboard — live video stream, SSE alerts, REST API, and metrics."""

import json
import logging
import os
from datetime import timedelta
from functools import wraps


import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    stream_with_context,
    url_for,
)
from queue import Queue

from src.config import Config
from src.dashboard.auth import verify_password

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.permanent_session_lifetime = timedelta(hours=Config.SESSION_LIFETIME_HOURS)

# ── SSE event queue ─────────────────────────────────────────────────────
event_queue: Queue = Queue()


def push_event(event: dict):
    """Called by detection pipeline to push a new alert to the dashboard."""
    event_queue.put(event)


# ── Auth ────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "logged_in" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        admin_hash = Config.ADMIN_PASSWORD_HASH
        if admin_hash:
            authenticated = username == "admin" and verify_password(password, admin_hash)
        else:
            logger.warning(
                "ADMIN_PASSWORD_HASH not set — using insecure default. "
                "Run: python -m src.dashboard.auth <password> and set the hash in .env"
            )
            authenticated = username == "admin" and password == "admin123"

        if authenticated:
            session.permanent = True
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Dashboard ───────────────────────────────────────────────────────────
@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html")


# ── SSE stream ──────────────────────────────────────────────────────────
@app.route("/live_alerts")
def live_alerts():
    def event_stream():
        while True:
            event = event_queue.get()
            yield f"data: {json.dumps(event)}\n\n"
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


# ── Video feed ──────────────────────────────────────────────────────────
# Shared frame buffer — set by main.py before starting the dashboard.
# When running standalone (python -m src.dashboard.app), a local camera is opened.
_shared_frame_buffer = None


def set_frame_buffer(fb):
    """Called by main.py to share the pipeline's frame buffer with the dashboard."""
    global _shared_frame_buffer
    _shared_frame_buffer = fb
    logger.info("Dashboard using shared frame buffer.")


def generate_frames():
    """Stream MJPEG frames — reads from shared buffer or falls back to direct camera."""
    import time

    # Fallback: open camera directly if no shared buffer (standalone mode)
    fallback_cam = None
    if _shared_frame_buffer is None:
        logger.warning("No shared frame buffer — opening camera directly (standalone mode).")
        fallback_cam = cv2.VideoCapture(Config.CAMERA_SOURCE)
        fallback_cam.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        fallback_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    last_frame_id = -1
    while True:
        if _shared_frame_buffer is not None:
            frame_id, frame = _shared_frame_buffer.read()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.03)  # ~30 FPS cap
                continue
            last_frame_id = frame_id
        else:
            success, frame = fallback_cam.read()
            if not success:
                time.sleep(0.03)
                continue

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── REST API ────────────────────────────────────────────────────────────

@app.route("/api/alerts")
@login_required
def api_alerts():
    """Query alerts with filtering — supports ?type=, ?severity=, ?since=, ?limit=."""
    from src.core.database import query_alerts
    return jsonify(query_alerts(
        event_type=request.args.get("type"),
        min_severity=request.args.get("severity", type=int),
        since=request.args.get("since"),
        limit=request.args.get("limit", 50, type=int),
    ))


@app.route("/api/alerts/<int:alert_id>/ack", methods=["PATCH"])
@login_required
def api_ack_alert(alert_id):
    """Mark an alert as acknowledged."""
    from src.core.database import acknowledge_alert
    acknowledge_alert(alert_id)
    return jsonify({"status": "ok"})


@app.route("/api/metrics/latest")
@login_required
def api_metrics():
    """Return latest pipeline metrics."""
    from src.core.database import get_latest_metrics
    data = get_latest_metrics()
    return jsonify(data or {})


@app.route("/api/stats/hourly")
@login_required
def api_hourly_stats():
    """Alerts per hour for the last 24 hours."""
    from src.core.database import get_hourly_alert_stats
    return jsonify(get_hourly_alert_stats())


# ── Legacy: JSON-based alerts (backward compat) ────────────────────────

@app.route("/alerts_data")
@login_required
def alerts_data():
    """Return latest alerts from DB or JSON fallback."""
    try:
        from src.core.database import query_alerts
        data = query_alerts(limit=15)
        return jsonify(data)
    except Exception:
        log_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/logs/alerts.json")
        )
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                data = json.load(f)
        else:
            data = []
        return jsonify(list(reversed(data))[-15:])


# ── Static anomaly images ──────────────────────────────────────────────
@app.route("/data/anomalies/<path:filename>")
def anomalies(filename):
    return send_from_directory(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/anomalies")),
        filename,
    )


# ── Boot ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data/anomalies", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)