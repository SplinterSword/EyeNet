# src/core/cooldown.py
"""Thread-safe cooldown manager with per-severity policies."""

import logging
import threading
from datetime import datetime, timedelta

from src.core.event_bus import Severity

logger = logging.getLogger(__name__)

# Default cooldown windows per severity
_DEFAULT_POLICIES: dict[Severity, timedelta] = {
    Severity.CRITICAL: timedelta(minutes=1),   # Gun / fire — short cooldown
    Severity.HIGH: timedelta(minutes=5),       # Knife / scissors
    Severity.MEDIUM: timedelta(minutes=10),    # Unknown face
    Severity.LOW: timedelta(hours=24),         # Uniform violation
}


class CooldownManager:
    """Determine whether an alert should fire based on key + severity policies."""

    def __init__(self, policies: dict[Severity, timedelta] | None = None):
        self._policies = policies or _DEFAULT_POLICIES
        self._timestamps: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def should_fire(self, key: str, severity: Severity) -> bool:
        """Return True and record the timestamp if the cooldown has expired."""
        cooldown = self._policies.get(severity, timedelta(minutes=5))
        now = datetime.now()
        with self._lock:
            last = self._timestamps.get(key)
            if last is None or (now - last) >= cooldown:
                self._timestamps[key] = now
                return True
        return False

    def reset(self, key: str):
        """Clear cooldown for a specific key."""
        with self._lock:
            self._timestamps.pop(key, None)

    def cleanup(self, max_age: timedelta = timedelta(hours=48)):
        """Remove entries older than *max_age* to prevent unbounded growth."""
        now = datetime.now()
        with self._lock:
            expired = [k for k, t in self._timestamps.items() if (now - t) > max_age]
            for k in expired:
                del self._timestamps[k]
        if expired:
            logger.debug("Cleaned up %d expired cooldown entries", len(expired))
