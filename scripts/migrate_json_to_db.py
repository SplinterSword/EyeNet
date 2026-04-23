# scripts/migrate_json_to_db.py
"""One-time migration: import existing alerts.json into the SQLite database."""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.database import init_db, get_db


def migrate():
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "logs", "alerts.json")

    if not os.path.exists(json_path):
        print(f"No JSON log found at {json_path} — nothing to migrate.")
        return

    with open(json_path, "r") as f:
        alerts = json.load(f)

    init_db()

    with get_db() as conn:
        for a in alerts:
            conn.execute(
                """INSERT INTO alerts (timestamp, event_type, severity, description, image_path, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    a.get("timestamp", ""),
                    a.get("event_type", "unknown"),
                    1,  # default severity for legacy alerts
                    a.get("description", ""),
                    a.get("image", ""),
                    json.dumps(a),
                ),
            )
        conn.commit()

    print(f"Migrated {len(alerts)} alerts from JSON to SQLite.")


if __name__ == "__main__":
    migrate()
