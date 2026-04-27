# src/core/logging_config.py
"""Structured logging setup with file rotation."""

import logging
import logging.handlers
import os

from src.config import Config


def setup_logging():
    """Configure root logger with console + rotating file output."""
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # File handler with rotation (10 MB per file, keep 3 backups)
    os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        Config.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
