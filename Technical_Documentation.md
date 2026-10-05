# EyeNet — Technical Documentation

System: real-time, camera-first surveillance pipeline. Single video source → shared frame buffer → multi-detector pipeline → track-gated events → priority event bus → SQLite persistence + SMS/email fan-out + Flask dashboard (MJPEG + SSE + REST).

Entrypoint: `src/main.py` — initializes DB, starts `SharedFrameBuffer`, `EventBus(num_workers=2)`, `DetectionPipeline`, Flask dashboard thread, OpenCV display loop.

## 1. Tech Stack

- Runtime: Python 3.9 (`python:3.9-slim-bullseye` in Docker), Flask, OpenCV (`opencv-python`), NumPy, Pillow
- CV/ML: Ultralytics YOLOv8 (`ultralytics`, `torch`/`torchvision`/`torchaudio`), `face-recognition` (dlib embeddings), `supervision` (ByteTrack, optional)
- Persistence: SQLite3 stdlib, WAL mode, `busy_timeout=5000`
- Notifications: `twilio` (SMS), `smtplib` stdlib (SMTP email)
- Auth: `bcrypt`, Flask cookie sessions
- Config: `python-dotenv`, centralized in `src/config.py`
- Deploy: `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`

Full dependency list: `requirements.txt` (unpinned).

## 2. Runtime Architecture

### 2.1 Data flow

```text
Camera (/dev/video0 | CAMERA_SOURCE) --> SharedFrameBuffer (capture thread, lock + frame_id)
  --> DetectionPipeline._run (polls read(), skips duplicate frame_id)
    --> FaceDetector.detect + check_uniform + detect_anomalies
    --> ObjectTracker.update --> should_alert(min_frames=5)
    --> EventBus.publish (PriorityQueue, maxsize=200)
      --> db_handler: compute_anomaly_score + insert_alert
      --> alert_handler: CooldownManager.should_fire + SMS/email
      --> sse_handler: push_event(event.to_dict())
  --> Flask dashboard (separate thread): /video_feed reads SharedFrameBuffer, /live_alerts drains Queue
  --> metrics-writer thread: insert_metrics every 5s, prune_old_metrics(24h)
```

### 2.2 Concurrency model

| Thread | Function | Source |
|---|---|---|
| `frame-capture` | `cv2.VideoCapture.read()` loop, lock-protected `frame.copy()`, FPS estimate | `src/core/frame_buffer.py` |
| `detection-pipeline` | Detector orchestration, tracking, annotation, metrics dict update | `src/core/pipeline.py:DetectionPipeline._run` |
| `metrics-writer` | `insert_metrics` + `prune_old_metrics` every 5s | `src/core/pipeline.py:DetectionPipeline._metrics_writer` |
| `event-worker-{0,1}` | Blocking `queue.get(timeout=1.0)`, sequential handler dispatch, exception isolated per handler | `src/core/event_bus.py:EventBus` |
| `flask-dashboard` | `app.run(host=0.0.0.0, port=5000, use_reloader=False)` | `src/main.py` |

- `SharedFrameBuffer.read()` returns `(frame_id, frame.copy())`. Pipeline drops frames with `frame_id == _last_frame_id` with 5ms sleep.
- `EventBus.publish` is non-blocking (`put_nowait`); drops on `queue.Full` with warning. Priority key: `(-int(severity), seq, event)` — min-heap, `CRITICAL` first. `seq` counter under lock breaks ties.
- SQLite uses thread-local connections (`threading.local()`), one connection per thread. WAL enables concurrent readers during pipeline writes.

## 3. Module Map

```text
src/main.py
src/config.py
src/core/frame_buffer.py
src/core/pipeline.py
src/core/tracker.py
src/core/event_bus.py
src/core/cooldown.py
src/core/anomaly_scorer.py
src/core/database.py
src/core/logging_config.py
src/detectors/face_detector.py
src/detectors/uniform_detector.py
src/detectors/anomaly_detector.py
src/dashboard/app.py
src/dashboard/auth.py
src/dashboard/templates/
src/notifications/sms_sender.py
src/notifications/email_sender.py
src/encoders/encode_faces.py
scripts/migrate_json_to_db.py
```

`src/realtime/` and `src/utils/` are legacy and not on the `src/main.py` path.

## 4. Capture Subsystem

Module: `src/core/frame_buffer.py:SharedFrameBuffer`

- Constructor: `(source=0, width=640, height=480)`. `start()` opens `cv2.VideoCapture`, sets `CAP_PROP_FRAME_WIDTH/HEIGHT`, raises `RuntimeError` if not opened.
- Capture loop: continuous `read()`, increments `_frame_id`, stores latest frame under `threading.Lock`. FPS = `1/(now-prev)`.
- `read() -> (frame_id, copy) | (None, None)`. `stop()` joins thread (3s timeout), releases capture.

## 5. Detection Subsystem

Orchestrator: `src/core/pipeline.py:DetectionPipeline`

Per-frame sequence in `_run`:
1. Face recognition → per-face `label = face:{name} | unknown_face`, `conf = 1.0 - distance`.
2. Uniform check (known faces only) → `label = uniform_violation:{roll}`, `conf = 1.0 - blue_ratio`, `box = torso_box`.
3. Hazard detection → `label = hazard:{yolo_label}`, `conf`, `box`.
4. `tracker.update(all_detections, frame_id)` → `should_alert(track)` → snapshot to `data/anomalies/{label}_{%Y%m%d_%H%M%S}.jpg` via `cv2.imwrite` → `event_bus.publish(DetectionEvent(...))`.
5. Annotation: bounding box + `#{track_id} {label} {conf}%`, uniform status text (`NO UNIFORM` / `UNIFORM OK`), metrics HUD overlay.
6. Metrics dict update: `{fps, detections, faces, processing_ms}` where `processing_ms` is monotonic detector time.

### 5.1 Face recognition

Module: `src/detectors/face_detector.py:FaceDetector`

- Artifact: `models/face_encodings.pkl` — `dict[roll, encoding]`, built by `src/encoders/encode_faces.py` from `students/<roll>.jpg`. If missing/empty, `detect()` returns `[]` (recognition disabled, no exception).
- Preprocess: BGR→RGB, 0.5× resize (`fx=0.5, fy=0.5`). `face_recognition.face_locations` + `face_encodings` on downscaled frame. Box scaled ×2 back to `(x1, y1, x2, y2)`.
- Match: L2 norm `np.linalg.norm(known - enc, axis=1)`, `argmin`. `is_known = min_dist < FACE_DISTANCE_THRESHOLD` (default `0.55`). Else `name=unknown`.
- Output: `[{name, distance, box, is_known}]`.

### 5.2 Uniform compliance

Module: `src/detectors/uniform_detector.py:check_uniform`

- Input: full BGR frame + face box `(x1,y1,x2,y2)`. Torso ROI: `x1-0.25*face_w … x2+0.25*face_w`, `y2 … y2+2.5*face_h`, clamped to frame. Invalid/empty ROI → `{wearing_uniform: False, confidence: 0.0}`.
- Color test: BGR→HSV, `cv2.inRange` with `LIGHT_BLUE_LOWER=[95,30,100]`, `LIGHT_BLUE_UPPER=[115,100,200]`, elliptical 5×5 morphological close, `ratio = blue_pixels / total_pixels`.
- Decision: `wearing = ratio >= 0.25`. Returns `{wearing_uniform, confidence: round(ratio,3), torso_box}`.

### 5.3 Hazard detection (YOLOv8)

Module: `src/detectors/anomaly_detector.py:detect_anomalies`

- Model: singleton `YOLO(Config.YOLO_HAZARD_MODEL)` (default `yolov8m.pt`). Inference: `model.predict(source=frame, conf=DEFAULT*0.8, verbose=False)`.
- Keyword filter (substring, case-insensitive): `knife, gun, fire, smoke, axe, crowbar, bat, sword, explosive, bomb, lighter, chainsaw, scissors, hammer`.
- Confidence gate: per-class map from `Config.HAZARD_CLASS_THRESHOLDS` (`knife 0.70, scissors 0.75, gun 0.60, fire 0.50, smoke 0.55`), fallback `HAZARD_CONF_THRESHOLD` (default `0.60`). `conf < threshold` → drop.
- Area gate: `bbox_area >= 0.01 * frame_area`, else drop.
- Temporal persistence: per-label `deque(maxlen=HAZARD_CONSECUTIVE_FRAMES)` (default 5). Current frame appends 1/0. Confirmed iff `sum(dq) >= MIN_CONSECUTIVE_FRAMES`; emits best-confidence box per confirmed label. Dead histories (all-zero, full window) pruned.
- Returns `(bool, [{label (lowercased), conf, box}])`. Only confirmed detections reach the tracker.

## 6. Tracking

Module: `src/core/tracker.py:ObjectTracker`

- Backend: `supervision.ByteTrack(track_activation_threshold=0.4, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=TARGET_FPS)` if `supervision` installed. Else fallback: monotonically increasing IDs, no association.
- `update(detections, frame_id)`: builds `sv.Detections(xyxy, confidence, class_id=hash(label)%10000)`, maps `tracker_id` → `TrackedObject(track_id, label, bbox, confidence, first_seen_frame, last_seen_frame)`. Updates bbox/confidence on re-observation. Prunes tracks unseen for >300 frames. Empty input still pumps `sv.Detections.empty()` to age tracks.
- `should_alert(track, min_frames=5)`: `False` if `alert_sent`; else `True` once `last_seen - first_seen >= min_frames`, sets `alert_sent=True`. Guarantees at most one event per track.

## 7. Event Bus

Module: `src/core/event_bus.py`

```python
class Severity(IntEnum): LOW=1; MEDIUM=2; HIGH=3; CRITICAL=4
@dataclass
class DetectionEvent:
  event_type: str  # unknown_face | uniform_violation | hazard | face
  severity: Severity
  description: str
  timestamp: datetime = now()
  track_id: int | None
  image_path: str
  metadata: dict   # {confidence, duration_frames, item_name, + anomaly_score at DB write}
```

- `cooldown_key`: `{event_type}:track:{track_id}` if tracked, else `{event_type}:{description}`.
- `to_dict()`: `timestamp → "%Y-%m-%d %H:%M:%S"`, `severity → name`. Used for SSE payload.
- `EventBus(num_workers=2, max_queue=200)`: `register(handler)`, `publish(event)`, `start()` spawns daemon workers, `stop()` joins with 2s timeout. Handler exceptions logged, do not block remaining handlers.

Severity assignment (`DetectionPipeline._classify_severity`, substring match on track label, lowercased):
- `CRITICAL`: `gun`, `fire`, `bomb`, `explosive`
- `HIGH`: `knife`, `smoke`, `axe`, `sword`
- `MEDIUM`: `unknown`
- `LOW`: default (uniform violations, known-face tracks)

## 8. Anomaly Scoring

Module: `src/core/anomaly_scorer.py:compute_anomaly_score`

Formula: `score = base_weight × confidence × max(min(duration_frames/10, 2.0), 0.5) × 10`, clamped to `[0, 100]`, rounded to 1 decimal.

Base weights (`_SEVERITY_WEIGHTS`, lookup key = `metadata.item_name` else `event_type`, default `3.0`):

| Key | Weight |
|---|---|
| `gun`, `bomb`, `explosive` | 10.0 |
| `fire` | 9.0 |
| `axe`, `sword` | 8.0 |
| `knife` | 7.0 |
| `smoke` | 6.0 |
| `unknown_face` | 4.0 |
| `uniform_violation` | 1.0 |

- `confidence` defaults to `0.5`, `duration_frames` defaults to `1` (factor 0.5 floor).
- Computed in `db_handler` and `alert_handler`; persisted as `metadata.anomaly_score`.

## 9. Cooldown / Deduplication

Two independent layers:

1. `src/core/cooldown.py:CooldownManager` — gates `alert_handler` in `src/main.py`. Key: `event.cooldown_key`. Policies: `CRITICAL 1min, HIGH 5min, MEDIUM 10min, LOW 24h` (fallback 5min). Thread-safe dict + lock. `reset(key)` clears; `cleanup(max_age=48h)` evicts stale keys.
2. `src/notifications/sms_sender.py` — gates Twilio send. Key: `dangerous_item_{item}` / `unknown_face_{location}` / raw `event_type`. Window: `NOTIFICATION_COOLDOWN_MINUTES=5`, in-memory dict, no persistence across restarts.

Effect: `LOW` uniform violations notify at most once per track per day at bus level; SMS layer additionally rate-limits to one per subtype per 5 minutes.

## 10. Persistence

Module: `src/core/database.py`. Connection: thread-local `sqlite3`, `row_factory=Row`, `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`. DB file auto-created with parent dirs.

### 10.1 Schema (`init_db`)

```sql
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  event_type TEXT NOT NULL,
  severity INTEGER NOT NULL DEFAULT 1,
  description TEXT,
  track_id INTEGER,
  image_path TEXT,
  metadata_json TEXT,
  acknowledged INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(event_type);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

CREATE TABLE IF NOT EXISTS students (
  roll TEXT PRIMARY KEY, name TEXT, email TEXT,
  encoding_path TEXT, registered_at TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  fps REAL, detection_count INTEGER, face_count INTEGER, processing_ms REAL
);
```

Note: runtime face matching reads `models/face_encodings.pkl`, not the `students` table. `students` is reserved metadata only.

### 10.2 Queries

- `insert_alert(event_type, severity:int, description, track_id, image_path, metadata:dict) -> rowid`. `metadata` serialized to `metadata_json`.
- `query_alerts(event_type?, min_severity?, since?, limit=50)`: `WHERE 1=1 [+ event_type = ?] [+ severity >= ?] [+ timestamp >= ?] ORDER BY timestamp DESC LIMIT min(limit,500)`.
- `acknowledge_alert(id)`: `UPDATE alerts SET acknowledged=1 WHERE id=?`.
- `insert_metrics(fps, detection_count, face_count, processing_ms)`.
- `prune_old_metrics(hours=24)`: `DELETE FROM metrics WHERE timestamp < datetime('now', '-N hours', 'localtime')`.
- `get_latest_metrics()`: `SELECT * FROM metrics ORDER BY id DESC LIMIT 1`.
- `get_hourly_alert_stats()`: `SELECT strftime('%H',timestamp) hour, COUNT(*) count FROM alerts WHERE timestamp >= datetime('now','-24 hours','localtime') GROUP BY hour ORDER BY hour`.
- Migration: `scripts/migrate_json_to_db.py` imports legacy `data/logs/alerts.json` into `alerts`.

## 11. Dashboard (Flask)

Module: `src/dashboard/app.py`. `app.secret_key = Config.SECRET_KEY`, `permanent_session_lifetime = SESSION_LIFETIME_HOURS` (default 8h). Templates: `dashboard.html`, `login.html`.

### 11.1 Routes

| Method | Route | Auth | Behavior |
|---|---|---|---|
| `GET,POST` | `/login` | no | Form `username`/`password`. If `ADMIN_PASSWORD_HASH` set: `username=="admin" and bcrypt.checkpw`. Else warn + `admin`/`admin123`. Success: `session.permanent=True`, `session["logged_in"]=True`, redirect `/`. Fail: re-render with error. |
| `GET` | `/logout` | no | `session.clear()`, redirect `/login`. |
| `GET` | `/` | `login_required` | Renders `dashboard.html`. |
| `GET` | `/video_feed` | no | MJPEG `multipart/x-mixed-replace; boundary=frame`. Reads `SharedFrameBuffer` if `set_frame_buffer()` was called, else opens `cv2.VideoCapture(CAMERA_SOURCE)` standalone. JPEG quality 70, ~30fps cap on duplicate frame_id. |
| `GET` | `/live_alerts` | no | SSE `text/event-stream`. Blocks on in-process `Queue`; yields `data: {json}\n\n` per `push_event(dict)`. Single-process only. |
| `GET` | `/api/alerts?type=&severity=&since=&limit=` | `login_required` | Delegates to `query_alerts`. `limit` default 50. Returns JSON array. |
| `PATCH` | `/api/alerts/<int:id>/ack` | `login_required` | `acknowledge_alert(id)` → `{"status":"ok"}`. |
| `GET` | `/api/metrics/latest` | `login_required` | `get_latest_metrics() or {}`. Dashboard polls every 3s. |
| `GET` | `/api/stats/hourly` | `login_required` | `get_hourly_alert_stats()`. |
| `GET` | `/alerts_data` | `login_required` | Legacy compat: `query_alerts(limit=15)`, fallback to `data/logs/alerts.json` last 15 reversed. |
| `GET` | `/data/anomalies/<file>` | no | `send_from_directory(data/anomalies)`. Snapshot serving. |

`login_required`: redirects to `/login` if `"logged_in" not in session`. Note `/video_feed` and `/live_alerts` are intentionally unauthenticated in code.

### 11.2 Wire formats

- MJPEG frame part: `--frame\r\nContent-Type: image/jpeg\r\n\r\n<jpeg bytes>\r\n`.
- SSE message: `data: {"event_type": "...", "severity": "CRITICAL", "description": "...", "timestamp": "YYYY-MM-DD HH:MM:SS", "track_id": N, "image_path": "...", "metadata": {...}}\n\n`.

## 12. Authentication

- Model: single `admin` role, Flask cookie session. No per-route RBAC.
- Hashing: `src/dashboard/auth.py` — `bcrypt.hashpw` / `bcrypt.checkpw` (utf-8). CLI: `python -m src.dashboard.auth "<password>"` prints hash for `ADMIN_PASSWORD_HASH`.
- Failure modes: unset `ADMIN_PASSWORD_HASH` → insecure `admin123` fallback with warning log; `verify_password` returns `False` on any exception. `SECRET_KEY` defaults to ephemeral `os.urandom(32).hex()` if unset (invalidates sessions on restart).

## 13. Notification Dispatch

Wiring: `src/main.py:alert_handler`. Order: cooldown check → score → branch.

| `event_type` | Condition | Action | Destination |
|---|---|---|---|
| `unknown_face` | `score >= 25` | `send_unknown_face_alert(location="Main Campus", image_url)` | All `ADMIN_PHONE_NUMBERS` via Twilio |
| `hazard` | `score >= 25` | `send_dangerous_item_alert(item_name, conf*100, "Main Campus", image_url)` | All `ADMIN_PHONE_NUMBERS` via Twilio |
| `uniform_violation` | always (no score gate) | `send_uniform_violation_email(roll, {roll}@mail.jiit.ac.in, now, fine=700)` | Student email via SMTP |

- SMS: `src/notifications/sms_sender.py:send_sms_alert`. No-op with warning if `TWILIO_ACCOUNT_SID/AUTH_TOKEN/PHONE_NUMBER` or recipient list missing. Body prefix `🚨 SECURITY ALERT — {TYPE}`, appends `Image: {path}` if present. Per-recipient `try/except`, failures logged per number.
- Email: `src/notifications/email_sender.py`, SMTP via `SMTP_HOST` (default `smtp.gmail.com`) / `SMTP_PORT` (default 587), auth `SMTP_EMAIL`/`SMTP_PASSWORD`.

## 14. Metrics Pipeline

- Producer: `DetectionPipeline.metrics` dict updated every frame (`fps` from buffer, `detections=len(all)`, `faces=count "face" in label`, `processing_ms`).
- Writer: `_metrics_writer` sleeps 5s → `insert_metrics(...)` → `prune_old_metrics(24h)`. Exceptions logged, loop continues.
- Consumers: `GET /api/metrics/latest`, `GET /api/stats/hourly`, HUD overlay in annotated frame.

## 15. Configuration Reference

Source: `src/config.py` (`dotenv.load_dotenv()`, `os.getenv` with cast). All values overridable via `.env` / environment.

| Key | Type | Default | Used by |
|---|---|---|---|
| `CAMERA_SOURCE` | int | `0` | `SharedFrameBuffer`, standalone `generate_frames` fallback |
| `FRAME_WIDTH` | int | `640` | capture size |
| `FRAME_HEIGHT` | int | `480` | capture size |
| `TARGET_FPS` | int | `15` | `ObjectTracker(frame_rate=)` |
| `FACE_DISTANCE_THRESHOLD` | float | `0.55` | `FaceDetector` L2 match |
| `HAZARD_CONSECUTIVE_FRAMES` | int | `5` | `detect_anomalies` deque length |
| `HAZARD_CONF_THRESHOLD` | float | `0.6` | YOLO fallback threshold |
| `YOLO_PERSON_MODEL` | str | `yolov8n.pt` | present in config, unused by current pipeline |
| `YOLO_HAZARD_MODEL` | str | `yolov8m.pt` | `detect_anomalies` weights |
| per-class (`HAZARD_CLASS_THRESHOLDS`, hardcoded) | map | `knife 0.70, scissors 0.75, gun 0.60, fire 0.50, smoke 0.55` | `detect_anomalies._get_threshold` |
| `TWILIO_ACCOUNT_SID` | str | — | SMS gate |
| `TWILIO_AUTH_TOKEN` | str | — | SMS gate |
| `TWILIO_PHONE_NUMBER` | str | — | SMS sender |
| `ADMIN_PHONE_NUMBERS` | csv list | `[]` | SMS recipients |
| `SMTP_EMAIL` | str | — | SMTP auth |
| `SMTP_PASSWORD` | str | — | SMTP auth |
| `SMTP_HOST` | str | `smtp.gmail.com` | SMTP |
| `SMTP_PORT` | int | `587` | SMTP |
| `SECRET_KEY` | str | ephemeral random | Flask sessions |
| `ADMIN_PASSWORD_HASH` | bcrypt str | — (fallback `admin123`) | `/login` |
| `SESSION_LIFETIME_HOURS` | int | `8` | session lifetime |
| `DB_PATH` | str | `data/eyenet.db` | `get_db` |
| `LOG_LEVEL` | str | `INFO` | `logging_config` |
| `LOG_FILE` | str | `data/logs/eyenet.log` | `RotatingFileHandler` |

Reference template: `.env.example`.

## 16. Deployment

- `Dockerfile.backend`: full system (`src/main.py`), mounts `./data:/app/data`, `./src:/app/src`, port `5000→5001`, `devices: /dev/video0:/dev/video0`, `privileged: true`, `restart: unless-stopped`, `PYTHONUNBUFFERED=1`.
- `Dockerfile.frontend`: dashboard-only (`src/dashboard`), port `5000:5000`, mount `./src/dashboard:/app`. Still Flask, not a JS SPA.
- Compose: `docker compose up --build`. Requires Linux host for camera passthrough. `.env` is gitignored; inject via environment in production. TLS termination expected at reverse proxy (not in app).

## 17. Operational Limits

- Single-process SSE: `event_queue: Queue` is in-process; multi-worker/multi-instance deployments need an external broker (e.g. Redis pub/sub) to fan out `/live_alerts`.
- Single admin identity; no RBAC, no API tokens; `/video_feed` and `/live_alerts` bypass `login_required`.
- `students` table is write-none on the hot path; enrollment source of truth is the pickle artifact.
- SMS cooldown state is in-memory; restarts reset deduplication windows.
- Snapshot storage is local disk (`data/anomalies/`); no object-store signed URLs, no retention pruning (unlike `metrics`).
