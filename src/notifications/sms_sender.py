# src/notifications/sms_sender.py
"""SMS alerts via Twilio with smart cooldown deduplication."""

import logging
from datetime import datetime, timedelta

from src.config import Config

logger = logging.getLogger(__name__)

# ── Cooldown tracking ───────────────────────────────────────────────────
_notification_timestamps: dict[str, datetime] = {}
NOTIFICATION_COOLDOWN_MINUTES = 5


def _should_send(event_key: str) -> bool:
    """Return True if no notification for this key was sent within the cooldown window."""
    last = _notification_timestamps.get(event_key)
    if last is None:
        return True
    return (datetime.now() - last) > timedelta(minutes=NOTIFICATION_COOLDOWN_MINUTES)


def _record_sent(event_key: str):
    _notification_timestamps[event_key] = datetime.now()


# ── Core sender ─────────────────────────────────────────────────────────
def send_sms_alert(event_type: str, message: str, image_url: str | None = None,
                   context: dict | None = None) -> bool:
    """Send an SMS alert to all admin phone numbers.

    Args:
        event_type: 'unknown_face' or 'dangerous_item'
        message: Human-readable alert body
        image_url: Optional path/URL to snapshot
        context: Extra context dict for deduplication keys
    """
    try:
        # Build a unique cooldown key
        event_key = event_type
        if context:
            if event_type == "dangerous_item" and "item_name" in context:
                event_key = f"{event_type}_{context['item_name'].lower()}"
            elif event_type == "unknown_face" and "location" in context:
                event_key = f"{event_type}_{context['location'].lower().replace(' ', '_')}"

        if not _should_send(event_key):
            logger.debug("Cooldown active for %s — skipping SMS", event_key)
            return False

        # Validate Twilio config
        sid = Config.TWILIO_ACCOUNT_SID
        token = Config.TWILIO_AUTH_TOKEN
        from_phone = Config.TWILIO_PHONE_NUMBER
        to_numbers = Config.ADMIN_PHONE_NUMBERS

        if not all([sid, token, from_phone]) or not to_numbers:
            logger.warning(
                "Twilio not configured — set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
                "TWILIO_PHONE_NUMBER, ADMIN_PHONE_NUMBERS in .env"
            )
            return False

        from twilio.rest import Client
        client = Client(sid, token)

        full_message = f"🚨 SECURITY ALERT — {event_type.upper()}\n\n{message}"
        if image_url:
            full_message += f"\n\nImage: {image_url}"

        for phone in to_numbers:
            phone = phone.strip()
            if not phone:
                continue
            try:
                client.messages.create(body=full_message, from_=from_phone, to=phone)
                logger.info("SMS sent to %s", phone)
            except Exception as e:
                logger.error("SMS to %s failed: %s", phone, e)

        _record_sent(event_key)
        return True

    except Exception as e:
        logger.error("send_sms_alert error: %s", e)
        return False


# ── Convenience wrappers ────────────────────────────────────────────────
def send_unknown_face_alert(location: str = "main entrance", image_url: str | None = None) -> bool:
    """Alert for unknown face detection."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return send_sms_alert(
        "unknown_face",
        f"⚠️ Unknown person detected at {location} at {ts}",
        image_url,
        {"location": location, "detection_time": ts},
    )


def send_dangerous_item_alert(item_name: str, confidence: float,
                              location: str = "campus", image_url: str | None = None) -> bool:
    """Alert for dangerous item detection."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return send_sms_alert(
        "dangerous_item",
        f"⚠️ THREAT DETECTED!\nItem: {item_name.upper()}\n"
        f"Confidence: {confidence:.1f}%\nLocation: {location}\nTime: {ts}",
        image_url,
        {"item_name": item_name, "confidence": confidence, "location": location},
    )
