# src/config.py
"""Centralized configuration — all secrets and tunables loaded from environment."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Camera ──────────────────────────────────────────────────────────
    CAMERA_SOURCE = int(os.getenv("CAMERA_SOURCE", "0"))
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
    TARGET_FPS = int(os.getenv("TARGET_FPS", "15"))

    # ── Detection ───────────────────────────────────────────────────────
    FACE_DISTANCE_THRESHOLD = float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.55"))
    HAZARD_CONSECUTIVE_FRAMES = int(os.getenv("HAZARD_CONSECUTIVE_FRAMES", "5"))
    YOLO_PERSON_MODEL = os.getenv("YOLO_PERSON_MODEL", "yolov8n.pt")
    YOLO_HAZARD_MODEL = os.getenv("YOLO_HAZARD_MODEL", "yolov8m.pt")

    # Per-class confidence overrides for hazard detection
    HAZARD_CONF_THRESHOLD = float(os.getenv("HAZARD_CONF_THRESHOLD", "0.6"))
    HAZARD_CLASS_THRESHOLDS = {
        "knife": 0.70,
        "scissors": 0.75,
        "gun": 0.60,
        "fire": 0.50,
        "smoke": 0.55,
    }

    # ── Notifications ───────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    ADMIN_PHONE_NUMBERS = [
        n.strip() for n in os.getenv("ADMIN_PHONE_NUMBERS", "").split(",") if n.strip()
    ]

    SMTP_EMAIL = os.getenv("SMTP_EMAIL")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

    # ── Dashboard ───────────────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(32).hex())
    ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")
    SESSION_LIFETIME_HOURS = int(os.getenv("SESSION_LIFETIME_HOURS", "8"))

    # ── Database ────────────────────────────────────────────────────────
    DB_PATH = os.getenv("DB_PATH", "data/eyenet.db")

    # ── Logging ─────────────────────────────────────────────────────────
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "data/logs/eyenet.log")
