# src/core/anomaly_scorer.py
"""Anomaly scoring engine — produces a 0-100 urgency score for each event."""

from src.core.event_bus import DetectionEvent

# Base weights by item/event type
_SEVERITY_WEIGHTS: dict[str, float] = {
    "gun": 10.0,
    "fire": 9.0,
    "knife": 7.0,
    "smoke": 6.0,
    "bomb": 10.0,
    "explosive": 10.0,
    "axe": 8.0,
    "sword": 8.0,
    "unknown_face": 4.0,
    "uniform_violation": 1.0,
}


def compute_anomaly_score(event: DetectionEvent) -> float:
    """Compute an urgency score (0-100) combining type, confidence, and duration.

    ``score = base_weight × confidence × duration_factor × 10``

    Higher score → more urgent.
    """
    # Determine the label for weight lookup
    label = event.metadata.get("item_name", event.event_type)
    base = _SEVERITY_WEIGHTS.get(label, 3.0)

    confidence = event.metadata.get("confidence", 0.5)

    # Tracks that persist longer are more credible
    duration_frames = event.metadata.get("duration_frames", 1)
    duration_factor = min(duration_frames / 10.0, 2.0)  # caps at 2×

    score = base * confidence * max(duration_factor, 0.5) * 10
    return round(min(score, 100.0), 1)
