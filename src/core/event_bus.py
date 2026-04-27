# src/core/event_bus.py
"""Async event bus with priority queue and background workers."""

import json
import logging
import queue
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Severity(IntEnum):
    LOW = 1       # Uniform violation
    MEDIUM = 2    # Unknown face
    HIGH = 3      # Hazardous object (knife, scissors)
    CRITICAL = 4  # Weapon (gun) or fire/smoke


@dataclass(order=False)
class DetectionEvent:
    """Immutable event emitted by the detection pipeline."""
    event_type: str            # "unknown_face", "uniform_violation", "hazard"
    severity: Severity
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    track_id: Optional[int] = None
    image_path: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def cooldown_key(self) -> str:
        """Unique key for cooldown deduplication."""
        if self.track_id is not None:
            return f"{self.event_type}:track:{self.track_id}"
        return f"{self.event_type}:{self.description}"

    def to_dict(self) -> dict:
        """Serialize for JSON / SSE."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        d["severity"] = self.severity.name
        return d


class EventBus:
    """Priority-queue event bus with N background dispatch workers."""

    def __init__(self, num_workers: int = 2, max_queue: int = 200):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue)
        self._handlers: list[Callable[[DetectionEvent], None]] = []
        self._workers: list[threading.Thread] = []
        self._running = False
        self._num_workers = num_workers
        self._counter = 0  # tie-breaker for same-priority events
        self._counter_lock = threading.Lock()

    def register(self, handler: Callable[[DetectionEvent], None]):
        """Register an event handler (called on every event)."""
        self._handlers.append(handler)
        logger.info("Registered event handler: %s", handler.__name__ if hasattr(handler, '__name__') else handler)

    def publish(self, event: DetectionEvent):
        """Non-blocking publish. Drops event if queue is full."""
        with self._counter_lock:
            self._counter += 1
            seq = self._counter
        try:
            # PriorityQueue is a min-heap; negate severity so CRITICAL (4) is dispatched first
            self._queue.put_nowait((-int(event.severity), seq, event))
        except queue.Full:
            logger.warning("Event queue full — dropping %s", event.event_type)

    def start(self):
        """Launch background workers."""
        self._running = True
        for i in range(self._num_workers):
            t = threading.Thread(target=self._worker, daemon=True, name=f"event-worker-{i}")
            t.start()
            self._workers.append(t)
        logger.info("EventBus started with %d workers", self._num_workers)

    def _worker(self):
        while self._running:
            try:
                _, _, event = self._queue.get(timeout=1.0)
                for handler in self._handlers:
                    try:
                        handler(event)
                    except Exception:
                        logger.exception("Event handler error for %s", event.event_type)
                self._queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._running = False
        for t in self._workers:
            t.join(timeout=2)
        logger.info("EventBus stopped.")
