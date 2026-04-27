\
# EyeNet — Real-Time Campus Surveillance & Incident Response

EyeNet is a **camera-first, real-time surveillance system** intended for campus / institutional environments. It continuously processes a live video feed and converts computer-vision detections into **actionable incidents**: persisted audit records, real-time dashboards, and automated notifications.

At its core, EyeNet is not “just object detection”—it is an **end-to-end incident pipeline**:

- Capture frames once and share them safely across multiple consumers.
- Run multiple CV detectors per frame (face recognition, uniform compliance, hazard detection).
- Track detections across frames to reduce noise.
- Convert persistent detections into prioritized events.
- Enforce notification cooldown policies to prevent alert spam.
- Persist alerts + operational metrics to SQLite (WAL enabled for concurrent reads).
- Provide a web dashboard with video streaming, live alerts via SSE, historical queries, acknowledgements, and basic analytics.

## Problem this solves

Campus security and discipline teams typically face these pain points:

- **Humans cannot monitor many cameras continuously**.
- Alerts are often **unstructured** (phone calls / WhatsApp) and lack traceability.
- False positives are common; systems must enforce **temporal persistence** and **cooldowns**.
- Admins need **real-time visibility** and a **reviewable history** (with snapshots).

EyeNet addresses this by turning raw video into **structured, queryable incidents**, with optional automated SMS/email escalation.

## Target users

- **Security operations** (unknown visitors, weapons/hazards, fire/smoke detection)
- **Campus administration / discipline** (uniform compliance)
- **Developers / researchers** building real-time CV pipelines with dashboards and event-driven notification delivery

---

## Core Features (as implemented)

### 1) Real-time detection pipeline (multi-signal)

The `DetectionPipeline` (`src/core/pipeline.py`) runs in its own thread and processes frames from a shared buffer:

- **Face recognition** (`src/detectors/face_detector.py`)
  - Loads enrolled face encodings from `models/face_encodings.pkl`.
  - Detects faces and computes encodings per frame.
  - Matches each face against known encodings using L2 distance.
  - Uses `FACE_DISTANCE_THRESHOLD` to classify **known** vs **unknown**.

- **Uniform compliance for known faces** (`src/detectors/uniform_detector.py`)
  - For *known* faces only, estimates a torso ROI below the face bounding box.
  - Converts ROI to HSV and checks for **light-blue fabric** using a fixed HSV threshold.
  - If blue pixel ratio is below the threshold, emits a **uniform violation** detection.

- **Hazard detection (YOLOv8)** (`src/detectors/anomaly_detector.py`)
  - Uses Ultralytics `YOLO` with weights configured via `YOLO_HAZARD_MODEL`.
  - Filters detections by a hazard keyword allow-list (e.g. `knife`, `gun`, `fire`, `smoke`, `sword`, `axe`, …).
  - Enforces minimum bounding-box area ratio to reduce tiny false positives.
  - Applies **per-class confidence thresholds** (`HAZARD_CLASS_THRESHOLDS`) plus a default threshold.
  - Applies **temporal persistence**: a hazard is only “confirmed” after being observed for `HAZARD_CONSECUTIVE_FRAMES`.

### 2) Cross-frame tracking to reduce noise

The pipeline feeds detections into an object tracker (`src/core/tracker.py`):

- Uses `supervision.ByteTrack` if installed.
- Assigns **persistent track IDs** across frames.
- Alerts are emitted only when a track persists for at least `min_frames` (default `5`) and only **once per track**.

This reduces jitter and prevents one-frame anomalies from generating immediate incidents.

### 3) Event-driven system with prioritization

EyeNet uses an in-process asynchronous event bus (`src/core/event_bus.py`):

- Events are represented as immutable `DetectionEvent` objects.
- The bus uses a **priority queue**:
  - Higher severities dispatch first (`CRITICAL` > `HIGH` > `MEDIUM` > `LOW`).
- Multiple background workers dispatch to registered handlers.

### 4) Severity classification + anomaly scoring

- Severity classification is rule-based (`DetectionPipeline._classify_severity`):
  - `CRITICAL`: `gun`, `fire`, `bomb`, `explosive`
  - `HIGH`: `knife`, `smoke`, `axe`, `sword`
  - `MEDIUM`: `unknown` (unknown faces)
  - `LOW`: uniform violations and other low-risk detections

- Anomaly urgency score (`src/core/anomaly_scorer.py`):
  - Produces a **0–100** score from:
    - base weight (by event/item)
    - detection confidence
    - track persistence duration
  - Persisted into alert metadata as `anomaly_score`.

### 5) Cooldown-based notification deduplication

To prevent alert spam, EyeNet implements cooldowns at two levels:

- **System-wide cooldown** (`src/core/cooldown.py`)
  - Per-severity cooldown windows:
    - `CRITICAL`: 1 minute
    - `HIGH`: 5 minutes
    - `MEDIUM`: 10 minutes
    - `LOW`: 24 hours
  - Cooldowns key off event identity (`DetectionEvent.cooldown_key`) and/or track id.

- **SMS sender local cooldown** (`src/notifications/sms_sender.py`)
  - Maintains an in-memory cooldown keyed by event subtype (e.g. `dangerous_item_knife`).
  - Default window: 5 minutes.

### 6) Persistence layer (SQLite + WAL)

EyeNet persists operational history to SQLite (`src/core/database.py`):

- Uses a **thread-local connection** (`threading.local()`).
- Enables `PRAGMA journal_mode=WAL` and `busy_timeout=5000` for better concurrent read behavior.
- Stores:
  - **Alerts** (incidents)
  - **Students** (reserved table for enrollment metadata)
  - **Metrics** (pipeline health snapshots)

### 7) Operations dashboard (Flask)

The dashboard (`src/dashboard/app.py`) is a Flask application providing:

- **Login session auth** (`/login`)
  - Username is fixed to `admin`.
  - Password verification uses `ADMIN_PASSWORD_HASH` (bcrypt) if configured.
  - If not configured, falls back to an insecure dev default (`admin123`).

- **Live video streaming** (`/video_feed`)
  - Streams MJPEG frames.
  - Uses the same shared camera buffer when launched via `src/main.py`.
  - Can run standalone and open camera directly as a fallback.

- **Live alerts via SSE** (`/live_alerts`)
  - Server-Sent Events stream.
  - Pipeline pushes alert events to a queue (`push_event`).

- **Alert management UI** (`templates/dashboard.html`)
  - Filters by type/severity.
  - Shows snapshot thumbnails.
  - Supports ACK button per alert.

- **Metrics polling** (`/api/metrics/latest`)
  - Dashboard polls every 3 seconds.

- **Hourly analytics** (`/api/stats/hourly`)
  - Aggregates alerts per hour for the last 24 hours.

### 8) Automated notifications

- **SMS (Twilio)** (`src/notifications/sms_sender.py`)
  - Sent for:
    - `unknown_face` (if anomaly score >= 25)
    - `hazard` (if anomaly score >= 25)
  - Sent to all numbers in `ADMIN_PHONE_NUMBERS`.

- **Email (SMTP)** (`src/notifications/email_sender.py`)
  - Sent for `uniform_violation`.
  - In `src/main.py`, student email is derived as:
    - `<roll>@mail.jiit.ac.in`
  - Requires SMTP configuration via environment variables.

### 9) Operational metrics (health + performance)

The pipeline maintains a metrics dict (FPS, detection counts, latency) and writes snapshots every 5 seconds:

- `metrics` table gets:
  - `fps`, `detection_count`, `face_count`, `processing_ms`
- Metrics older than 24 hours are pruned automatically.

### 10) One-time migration from legacy JSON logs

`scripts/migrate_json_to_db.py` imports legacy `data/logs/alerts.json` into the SQLite `alerts` table.

---

## Tech Stack (from the repository)

### Backend / runtime

- **Python** (Docker images use `python:3.9-slim-bullseye`)
- **Flask** (dashboard + REST API)
- **OpenCV** (`opencv-python`) for camera capture, encoding, and rendering

### Computer vision / ML

- **Ultralytics YOLOv8** (`ultralytics`) for hazard detection
- **face-recognition** (dlib-based) for face embeddings + matching
- **NumPy** for vector math
- **supervision** (optional) for ByteTrack-based tracking
- `torch`, `torchvision`, `torchaudio` are included as dependencies (required by YOLO/Ultralytics)

### Data & persistence

- **SQLite** (`sqlite3` stdlib) with WAL mode

### Notifications

- **Twilio** SMS (`twilio`)
- **SMTP** for email (`smtplib` stdlib)

### Security

- **bcrypt** for admin password hashing/verification
- Flask sessions (cookie-based) using `SECRET_KEY`

### Deployment

- Docker:
  - `Dockerfile.backend` for the full app
  - `Dockerfile.frontend` for dashboard-only container (note: still Python/Flask; not a JS frontend)
- `docker-compose.yml` for local container orchestration and camera device passthrough

---

## System Architecture

### High-level data flow

```mermaid
flowchart LR
  CAM[Camera / Video Source] --> FB[SharedFrameBuffer]
  FB --> PIPE[DetectionPipeline]

  PIPE -->|detections| TRK[ObjectTracker]
  TRK -->|persistent track| EV[DetectionEvent]
  EV --> BUS[EventBus (priority queue)]

  BUS --> DBH[DB Handler]
  DBH --> SQLITE[(SQLite: alerts, metrics)]

  BUS --> NTFY[Notification Handler]
  NTFY --> TW[Twilio SMS]
  NTFY --> SMTP[SMTP Email]

  BUS --> SSEH[SSE Handler]
  SSEH --> DASH[Flask Dashboard]

  FB --> DASH
```

### Architectural decisions you can see in code

- **Single camera reader**: `SharedFrameBuffer` prevents multiple components from fighting over `/dev/video0`.
- **Threaded isolation**: capture, detection, metrics writer, and dashboard run in separate threads.
- **Async-ish event handling**: `EventBus` decouples detection from persistence/notification latency.
- **WAL-mode SQLite**: designed for concurrent reads from the dashboard while writes happen from the pipeline.

---

## Folder Structure

```text
EyeNet/
  src/
    main.py                    # System entrypoint (pipeline + event bus + dashboard)
    config.py                  # Central config loaded from .env
    core/
      frame_buffer.py          # Shared camera capture (one writer, many readers)
      pipeline.py              # Orchestrates face/uniform/hazard detection
      tracker.py               # ByteTrack wrapper (or fallback IDs)
      event_bus.py             # Priority event bus with worker threads
      cooldown.py              # Per-severity cooldown policy
      anomaly_scorer.py        # 0-100 urgency score
      database.py              # SQLite schema + queries + WAL
      logging_config.py        # Rotating logs to file + console
    detectors/
      face_detector.py         # face-recognition matching to enrolled encodings
      uniform_detector.py      # HSV torso check for uniform compliance
      anomaly_detector.py      # YOLO hazard detector + temporal filtering
    dashboard/
      app.py                   # Flask UI + SSE + REST API
      auth.py                  # bcrypt hash/verify utilities
      templates/               # HTML templates (dashboard, login, etc.)
    notifications/
      sms_sender.py            # Twilio SMS alerts
      email_sender.py          # SMTP email alerts
    encoders/
      encode_faces.py          # Build models/face_encodings.pkl from students/ images
    realtime/                  # Legacy monolithic realtime script (historical)
    utils/                     # Legacy utilities
  scripts/
    migrate_json_to_db.py      # Migrate legacy alerts.json into SQLite
  data/
    eyenet.db                  # Default SQLite DB location
    anomalies/                 # Snapshots saved on events
    logs/                      # Rotating log files / legacy JSON logs
  students/                    # Enrollment photos named by roll number
  models/
    face_encodings.pkl         # Generated face encodings artifact
  docker-compose.yml
  Dockerfile.backend
  Dockerfile.frontend
  requirements.txt
  .env.example
```

---

## Installation & Local Setup

### Prerequisites

- Python 3.9+ recommended
- A video source (default is webcam `CAMERA_SOURCE=0`)
- Optional:
  - Twilio account for SMS
  - SMTP credentials for uniform violation emails

### 1) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Copy `.env.example` to `.env` and fill in values.

```bash
cp .env.example .env
```

### 4) Enroll students (face encodings)

1. Place images in `students/` named by roll number:
   - `students/22102027.jpg`
2. Generate encodings:

```bash
python -m src.encoders.encode_faces
```

This writes `models/face_encodings.pkl`.

### 5) Run the system

```bash
python -m src.main
```

- Dashboard: `http://localhost:5000`
- Default login:
  - username: `admin`
  - password: `admin123` (only if you did **not** set `ADMIN_PASSWORD_HASH`)

### 6) Optional: migrate legacy JSON logs to SQLite

If you previously logged to `data/logs/alerts.json`:

```bash
python scripts/migrate_json_to_db.py
```

---

## Environment Variables

All configuration is loaded via `src/config.py` (`python-dotenv` loads `.env`).

### Camera

- `CAMERA_SOURCE`
  - Integer camera index (default `0`)
- `FRAME_WIDTH`
- `FRAME_HEIGHT`
- `TARGET_FPS`

### Detection tuning

- `FACE_DISTANCE_THRESHOLD`
  - Face recognition match threshold (lower is stricter)
- `HAZARD_CONSECUTIVE_FRAMES`
  - Number of consecutive frames required before confirming a hazard
- `HAZARD_CONF_THRESHOLD`
  - Default YOLO confidence threshold (used as fallback)
- `YOLO_PERSON_MODEL`
  - Present in config but not required by the current pipeline
- `YOLO_HAZARD_MODEL`
  - Path to YOLO hazard weights (default `yolov8m.pt`)

### Twilio (SMS)

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `ADMIN_PHONE_NUMBERS`
  - Comma-separated list, e.g. `+15550001111,+15550002222`

### Email (SMTP)

- `SMTP_EMAIL`
- `SMTP_PASSWORD`
- `SMTP_HOST` (default `smtp.gmail.com`)
- `SMTP_PORT` (default `587`)

### Dashboard authentication

- `SECRET_KEY`
  - Flask session secret key
- `ADMIN_PASSWORD_HASH`
  - bcrypt hash used to validate the admin password
- `SESSION_LIFETIME_HOURS` (default `8`)

Generate a bcrypt hash with:

```bash
python -m src.dashboard.auth "your-password"
```

### Database

- `DB_PATH` (default `data/eyenet.db`)

### Logging

- `LOG_LEVEL` (default `INFO`)
- `LOG_FILE` (default `data/logs/eyenet.log`)

---

## API Documentation (Flask Dashboard)

All API endpoints are served by the dashboard app (`src/dashboard/app.py`).

**Authentication**: all `/api/*` endpoints require a valid login session.

### `GET /api/alerts`

Query persisted alerts.

Query params:

- `type` (string) — filter by `event_type`
- `severity` (int) — minimum severity (`1..4`)
- `since` (string) — filter by timestamp lower bound (passed directly into SQL)
- `limit` (int) — default `50`, capped at `500`

Returns: JSON array of alert rows.

### `PATCH /api/alerts/<alert_id>/ack`

Marks an alert as acknowledged.

Returns:

```json
{ "status": "ok" }
```

### `GET /api/metrics/latest`

Returns the most recent pipeline metrics row.

### `GET /api/stats/hourly`

Returns counts per hour for alerts in the last 24 hours.

### `GET /video_feed`

MJPEG stream of live frames.

### `GET /live_alerts`

Server-Sent Events stream. Each message is JSON encoded.

---

## Database Design

The schema is created by `init_db()` in `src/core/database.py`.

### `alerts`

- `id` (PK)
- `timestamp` (TEXT, default localtime)
- `event_type` (TEXT)
- `severity` (INTEGER)
- `description` (TEXT)
- `track_id` (INTEGER)
- `image_path` (TEXT)
- `metadata_json` (TEXT)
- `acknowledged` (INTEGER boolean)

Indexes:

- `timestamp`
- `event_type`
- `severity`

### `metrics`

- `id` (PK)
- `timestamp`
- `fps`
- `detection_count`
- `face_count`
- `processing_ms`

### `students`

Present in schema for enrollment metadata:

- `roll` (PK)
- `name`, `email`, `encoding_path`, `registered_at`

Note: current runtime face recognition reads from `models/face_encodings.pkl`, not from the `students` table.

---

## Authentication & Authorization

### Dashboard auth model

- Session-based auth via Flask cookies.
- Protected routes use the `login_required` decorator.
- Authentication logic:
  - Username must be `admin`.
  - Password is verified against `ADMIN_PASSWORD_HASH` via bcrypt.
  - If no hash is configured, the dashboard logs a warning and falls back to `admin/admin123`.

### Authorization

Current implementation uses a single admin role. There are no multi-role permissions yet.

---

## AI / Agent Workflows

This repository implements **computer-vision inference workflows**, not LLM/agent orchestration.

- Face recognition uses precomputed embeddings (`face-recognition`).
- Hazard detection uses a YOLOv8 model (`ultralytics`).

There is no prompt orchestration, tool-calling, vector database, or retrieval-augmented generation in the inspected code.

---

## Deployment Guide

### Docker Compose

`docker-compose.yml` defines two services:

- `backend`
  - Builds from `Dockerfile.backend`
  - Mounts `./data` and `./src` into the container
  - Exposes container port `5000` as host `5001`
  - Passes through `/dev/video0` (requires Linux host)
  - Runs in privileged mode for camera access

- `frontend`
  - Builds from `Dockerfile.frontend`
  - Exposes `5000:5000`
  - Note: this is **still Flask** (dashboard-only), not a JS SPA.

Run:

```bash
docker compose up --build
```

### Important notes for production

- Camera passthrough and `privileged: true` are host-specific.
- `.env` is gitignored; inject env vars via your deployment system.
- Consider running behind a reverse proxy and terminating TLS externally.

---

## Performance Optimizations (present in code)

- **SharedFrameBuffer**: one camera reader thread + lock-protected copies.
- **Downscaled face recognition**: face detection/encoding runs on a 0.5× resized frame.
- **Temporal persistence** for hazards (`HAZARD_CONSECUTIVE_FRAMES`).
- **Tracking-based alert emission**: only alert after `min_frames` persistence per track.
- **Cooldown policies**: prevent repeated notifications for the same ongoing incident.
- **SQLite WAL**: improves concurrent read/write behavior when dashboard is querying.
- **Rotating logs**: `RotatingFileHandler` limits log file growth.

---

## Challenges Solved (evident from implementation)

- **Real-time concurrency**: separating capture, detection, notification, persistence, and UI threads.
- **Noise reduction**: combining tracking + temporal hazard filtering.
- **Operational reliability**: WAL-mode DB access, busy timeouts, and cooldown policies.
- **Live UI updates**: SSE stream for alerts + MJPEG for video feed.

---

## Future Improvements

- Replace in-process `Queue` for SSE with a multi-process safe broker (Redis/pubsub) if scaling beyond a single process.
- Add proper user management (roles, per-route permissions).
- Persist student enrollment into `students` table and bind face encodings to DB.
- Add rate limiting on API routes.
- Add structured schema migrations (Alembic is overkill for SQLite but still possible).
- Add an alert review workflow (notes, assignment, escalation state machine).
- Replace snapshot `image_path` with signed URLs if storing in object storage.
- Add GPU/accelerator selection and model warmup to reduce first-inference latency.

---

## Screenshots

Insert screenshots/gifs here:

- `docs/screenshots/dashboard.png`
- `docs/screenshots/live-alerts.png`
- `docs/screenshots/hazard-detection.png`
- `docs/screenshots/uniform-violation.png`

---

## Contributing

1. Create a feature branch.
2. Keep changes focused (detector logic, dashboard, or infra).
3. Prefer adding small, testable modules under `src/core` / `src/detectors`.
4. Avoid committing:
   - `.env`
   - camera snapshots in `data/`
   - large model weights (already present in repo; consider Git LFS for long-term)

---

## License

No license file was found in the repository at the time of inspection. If you intend this project to be open source, add a `LICENSE` file (e.g., MIT/Apache-2.0) and update this section.
