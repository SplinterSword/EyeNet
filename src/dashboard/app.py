from flask import Flask, render_template, Response, jsonify, send_from_directory, request, redirect, url_for, session
from functools import wraps
import cv2
import os
import json
from threading import Lock

app = Flask(__name__)
app.secret_key = "supersecretkey"  # change this later!

print("✅ Unified Flask Dashboard Loading...")

from flask import stream_with_context
from queue import Queue

event_queue = Queue()

def push_event(event):
    """Called by detection script to push a new alert instantly."""
    event_queue.put(event)

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
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin123":  # simple hardcoded creds
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route('/')
@login_required
def dashboard():
    """Main unified dashboard with live feed + alerts."""
    return render_template('dashboard.html')


@app.route('/live_alerts')
def live_alerts():
    """Stream real-time alerts to the dashboard."""
    def event_stream():
        while True:
            event = event_queue.get()
            yield f"data: {json.dumps(event)}\n\n"
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

camera_lock = Lock()
camera = cv2.VideoCapture(0)

def generate_frames():
    """Stream frames from webcam."""
    while True:
        with camera_lock:
            success, frame = camera.read()
        if not success:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Live video stream route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/alerts_data')
def alerts_data():
    """Return latest alerts as JSON (for auto-refresh)."""
    log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/logs/alerts.json"))
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
        data = []
    return jsonify(list(reversed(data))[-15:])


@app.route('/data/anomalies/<path:filename>')
def anomalies(filename):
    """Serve anomaly snapshot images."""
    return send_from_directory(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/anomalies")), filename)


print("📜 Registered Routes:")
for rule in app.url_map.iter_rules():
    print(rule)

if __name__ == "__main__":
    os.makedirs("data/anomalies", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)