# EyeNet

Watch every camera at once, miss nothing — live detection, structured incidents, and instant escalation.

![EyeNet demo](docs/screenshots/dashboard.png)
<!-- Add a demo gif/screenshot at docs/screenshots/dashboard.png -->

## What is this?

You know that feeling? One guard staring at 8 camera feeds, alerts over phone calls / WhatsApp with zero traceability, and every one-frame false positive becomes a panic. EyeNet fixes that.

EyeNet is three small pieces that work together:

1. **A detection pipeline** (`src/core/pipeline.py`) — shared frame buffer, face + uniform + YOLOv8 hazard detectors, ByteTrack tracking.
2. **An event bus + store** (`src/core/event_bus.py`, `src/core/database.py`) — priority queue, anomaly scoring, cooldowns, SQLite WAL persistence.
3. **A Flask dashboard** (`src/dashboard/app.py`) — MJPEG video, SSE live alerts, history + ACK, hourly analytics, Twilio / SMTP escalation.

The trick is simple: capture once in `SharedFrameBuffer` and share it across threads. Every detection is tracked for 5 frames before it becomes an event, scored 0-100, then gated by per-severity cooldowns before it hits the DB + dashboard + SMS/email. So the dashboard only ever shows persistent, deduplicated incidents with snapshots.

         Think CCTV DVR, but it actually watches back.

## Motivation

Humans can't monitor many cameras continuously — one leak / intruder / fire missed is everything — `grep`-ing a log file finds text, not evidence, and raw YOLO boxes without tracking mean alert spam, so teams fall back to ignoring the monitor wall.

- Shared primitives: one `SharedFrameBuffer`, same frame for pipeline + MJPEG stream, no fighting over `/dev/video0`.
- Faster response: detect once in `pipeline.py`, reuse everywhere via `DetectionEvent` instead of re-running models per consumer.
- Stay in flow: severity-sorted SSE feed, filter by type/severity, snapshot thumbnails, ACK inline without tab-hopping.

## Quick Start

No hosted demo — just Python and a webcam. For local dev, have Twilio / SMTP ready only if you want SMS/email.

### 1. Enroll faces

Drop enrollment photos in `students/` named by roll number, then build encodings:

```bash
python -m src.encoders.encode_faces
# students/22102027.jpg -> models/face_encodings.pkl
```

The runtime matches with L2 distance against `FACE_DISTANCE_THRESHOLD`. Unknown faces become `MEDIUM` events.

### 2. Run the system

```bash
python -m src.main
# Dashboard: http://localhost:5000
# login: admin / admin123 (only if you did NOT set ADMIN_PASSWORD_HASH)
```

`src/main.py` wires it all: `init_db()` → `SharedFrameBuffer.start()` → `EventBus(num_workers=2)` → `DetectionPipeline.start()` → Flask in a thread + local `cv2.imshow` view. Press `q` to stop.

### 3. Review and ACK incidents

- Stay on `/` → live MJPEG + SSE alerts appear with snapshots from `data/anomalies/`
- Filter by type/severity → `GET /api/alerts?type=hazard&severity=3`
- ACK what you handled → `PATCH /api/alerts/<id>/ack`
- Check health → `GET /api/metrics/latest` polls every 3s, `GET /api/stats/hourly` for last 24h

### 4. Notifications, team model, and evidence

- `src/notifications/` — SMS for `unknown_face` / `hazard` (score >= 25), email for `uniform_violation` to `<roll>@mail.jiit.ac.in`
- No orgs / roles yet — one `admin` user, session-cookie auth (`SECRET_KEY` + bcrypt `ADMIN_PASSWORD_HASH`)
- No auto-delete — snapshots stay in `data/anomalies/`, metrics older than 24h are pruned

See `## Usage` below for the daily loop. Want Docker instead? See `## Contributing`.

## Usage

Available pages (auth required via `login_required` + Flask session):

- `/login` — fixed `admin` username, bcrypt verify against `ADMIN_PASSWORD_HASH`, falls back to `admin123`
- `/` — ops dashboard, MJPEG + SSE + filters + ACK + metrics charts
- `/video_feed` — MJPEG stream, shared buffer when run via `src/main.py`, opens camera directly standalone
- `/live_alerts` — SSE stream, pipeline pushes via `push_event(event.to_dict())`
- `/api/alerts` — `GET` list `?type=&severity=&since=&limit=` (default 50, cap 500)
- `/api/alerts/<id>/ack` — `PATCH` acknowledge → `{ "status": "ok" }`
- `/api/metrics/latest` — latest `{ fps, detection_count, face_count, processing_ms }`
- `/api/stats/hourly` — counts per hour, last 24h

Behavior notes:

- Severity by construction — `CRITICAL`: gun/fire/bomb/explosive, `HIGH`: knife/smoke/axe/sword, `MEDIUM`: unknown face, `LOW`: uniform + rest (`DetectionPipeline._classify_severity`).
- Urgency model — `compute_anomaly_score()` 0-100 from base weight + confidence + track persistence, stored in `metadata_json`.
- All `/api/*` routes require login session (`SESSION_LIFETIME_HOURS`, default 8).
- Cooldowns stop spam — `CRITICAL`: 1m, `HIGH`: 5m, `MEDIUM`: 10m, `LOW`: 24h + 5m local SMS cooldown per subtype.

> [!NOTE]
> **Snapshots are evidence.** Every persisted alert stores `image_path` in `data/anomalies/`. ACK doesn't delete — query history via `/api/alerts`.

## Examples

Catch an unknown visitor:

```text
camera -> face_detector (0.5x downscaled) -> no match < FACE_DISTANCE_THRESHOLD
-> track 5 frames -> MEDIUM unknown_face, score 0-100
-> cooldown 10m -> insert_alert + SSE + Twilio SMS if score >= 25
-> appears in dashboard, filter type=unknown_face, ACK when handled
```

Confirm a hazard without false alarms:

```text
YOLO yolov8m.pt -> keyword filter (knife/gun/fire/smoke/...) + area + per-class conf
-> HAZARD_CONSECUTIVE_FRAMES=5 -> track persists -> HIGH/CRITICAL
-> cooldown 5m/1m -> SMS to ADMIN_PHONE_NUMBERS
```

Enforce uniform compliance:

```text
known face -> torso ROI -> HSV light-blue check
-> blue ratio low -> LOW uniform_violation
-> cooldown 24h -> SMTP to <roll>@mail.jiit.ac.in (fine Rs. 700)
```

## What's in the repo?

```text
src/main.py                  -> entrypoint: DB + buffer + bus + pipeline + dashboard
src/config.py                -> central config loaded from .env
src/core/frame_buffer.py     -> SharedFrameBuffer (one writer, many readers)
src/core/pipeline.py         -> orchestrates face/uniform/hazard detection
src/core/tracker.py          -> ByteTrack wrapper (or fallback IDs)
src/core/event_bus.py        -> priority event bus with worker threads
src/core/cooldown.py         -> per-severity cooldown policy
src/core/anomaly_scorer.py   -> 0-100 urgency score
src/core/database.py         -> SQLite schema + queries + WAL
src/core/logging_config.py   -> rotating logs to file + console
src/detectors/face_detector.py    -> face-recognition matching to encodings
src/detectors/uniform_detector.py -> HSV torso check
src/detectors/anomaly_detector.py -> YOLO hazard + temporal filtering
src/dashboard/app.py         -> Flask UI + SSE + REST API
src/dashboard/auth.py        -> bcrypt hash/verify, python -m src.dashboard.auth
src/dashboard/templates/     -> dashboard, login HTML
src/notifications/sms_sender.py   -> Twilio SMS alerts
src/notifications/email_sender.py -> SMTP email alerts
src/encoders/encode_faces.py -> build models/face_encodings.pkl from students/
scripts/migrate_json_to_db.py -> migrate legacy alerts.json into SQLite
docker-compose.yml           -> backend :5001 + frontend :5000 + /dev/video0 passthrough
Dockerfile.backend           -> full app image (python:3.9-slim-bullseye)
Dockerfile.frontend          -> dashboard-only image
requirements.txt             -> Flask, opencv, ultralytics, face-recognition, twilio
.env.example                 -> camera + tuning + Twilio + SMTP + auth + DB + logging
students/                    -> enrollment photos named by roll number
models/face_encodings.pkl    -> generated face encodings artifact
```

## For more technical info

Skipping the deep dive here on purpose. For security model, SQLite schema, session flow, detection tuning, challenges, and deployment — checkout `Technical_Documentation.md`.

## Contributing

### Clone the repo

```bash
git clone https://github.com/SplinterSword/EyeNet.git
cd EyeNet
```

### Local dev

Prereqs: Python 3.9+ recommended, webcam (`CAMERA_SOURCE=0`), optional Twilio account + SMTP creds.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in SECRET_KEY + ADMIN_PASSWORD_HASH, Twilio/SMTP if needed
python -m src.encoders.encode_faces
python -m src.main
```

Condensed env — full table lives in `Technical_Documentation.md`:

```bash
# .env
CAMERA_SOURCE=0 / FRAME_WIDTH=640 / FRAME_HEIGHT=480 / TARGET_FPS=15
FACE_DISTANCE_THRESHOLD=0.55 / HAZARD_CONSECUTIVE_FRAMES=5 / HAZARD_CONF_THRESHOLD=0.6
YOLO_HAZARD_MODEL=yolov8m.pt / YOLO_PERSON_MODEL=yolov8n.pt
TWILIO_ACCOUNT_SID= / TWILIO_AUTH_TOKEN= / TWILIO_PHONE_NUMBER= / ADMIN_PHONE_NUMBERS=
SMTP_EMAIL= / SMTP_PASSWORD= / SMTP_HOST=smtp.gmail.com / SMTP_PORT=587
SECRET_KEY= / ADMIN_PASSWORD_HASH= / SESSION_LIFETIME_HOURS=8
DB_PATH=data/eyenet.db / LOG_LEVEL=INFO / LOG_FILE=data/logs/eyenet.log
```

Generate admin hash with:

```bash
python -m src.dashboard.auth "your-password"
```

Starts Flask on `http://localhost:5000`. Enroll a test face first — unknown-face SMS only fires if encodings exist to compare against.

Docker instead:

```bash
docker compose up --build
# backend -> http://localhost:5001, frontend -> http://localhost:5000
# requires Linux host for /dev/video0 passthrough + privileged: true
```

### Run checks

```bash
python -m compileall src
docker compose up --build # container bundle check
python scripts/migrate_json_to_db.py # only if migrating legacy data/logs/alerts.json
```

### Submit a pull request

Fork the repo and open a PR to `main`. Keep it scoped — one feature / fix per PR.
