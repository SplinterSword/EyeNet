# src/core/frame_buffer.py
"""Thread-safe shared frame buffer — one writer (capture), many readers."""

import cv2
import logging
import threading
import time

logger = logging.getLogger(__name__)


class SharedFrameBuffer:
    """Decouples camera capture from processing.

    One background thread captures frames continuously. Multiple consumers
    can call ``read()`` to get the latest frame without blocking each other
    or fighting over the camera device.
    """

    def __init__(self, source: int = 0, width: int = 640, height: int = 480):
        self._source = source
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id: int = 0
        self._running = False
        self._fps: float = 0.0
        self._thread: threading.Thread | None = None

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self) -> "SharedFrameBuffer":
        """Open the camera and start the background capture thread."""
        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source {self._source}. "
                "Check /dev/video* access and that no other process is using it."
            )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="frame-capture")
        self._thread.start()
        logger.info("SharedFrameBuffer started (source=%s, %dx%d)", self._source, self._width, self._height)
        return self

    def stop(self):
        """Stop capture and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("SharedFrameBuffer stopped.")

    # ── capture loop ────────────────────────────────────────────────────
    def _capture_loop(self):
        prev = time.monotonic()
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
            now = time.monotonic()
            self._fps = 1.0 / max(now - prev, 1e-6)
            prev = now

    # ── public API ──────────────────────────────────────────────────────
    def read(self) -> tuple[int | None, "cv2.Mat | None"]:
        """Return ``(frame_id, frame_copy)`` or ``(None, None)``."""
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame_id, self._frame.copy()

    @property
    def fps(self) -> float:
        """Approximate capture FPS."""
        return self._fps

    @property
    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id
