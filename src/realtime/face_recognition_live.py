# src/realtime/face_recognition_live.py
import cv2
import face_recognition
import pickle
import numpy as np
import os, time
import json
import time
import threading
from datetime import datetime, timedelta
from src.notifications.email_sender import send_uniform_violation_email
from src.notifications.sms_sender import send_unknown_face_alert, send_dangerous_item_alert

# Dictionary to track uniform violations with timestamps
uniform_violations = {}
VIOLATION_COOLDOWN_HOURS = 24  # 24 hours cooldown between notifications

# Dictionary to track last notification times for unknown faces and dangerous items
last_notification = {}
NOTIFICATION_COOLDOWN = 300  # 5 minutes in seconds

def should_notify_violation(student_roll):
    """
    Check if we should notify about a uniform violation.
    Returns True if either:
    1. Student hasn't been recorded before, or
    2. Last violation was more than 24 hours ago
    """
    current_time = datetime.now()
    
    if student_roll not in uniform_violations:
        return True
    
    last_violation = uniform_violations[student_roll]
    time_since_violation = current_time - last_violation
    
    return time_since_violation > timedelta(hours=VIOLATION_COOLDOWN_HOURS)

def record_violation(student_roll):
    """Record a new violation with the current timestamp"""
    uniform_violations[student_roll] = datetime.now()
    
# Background thread to clean up old violations
def cleanup_old_violations():
    """Periodically clean up violations older than the cooldown period"""
    while True:
        current_time = datetime.now()
        expired_rolls = [
            roll for roll, timestamp in uniform_violations.items()
            if (current_time - timestamp) > timedelta(hours=VIOLATION_COOLDOWN_HOURS)
        ]
        
        for roll in expired_rolls:
            uniform_violations.pop(roll, None)
        
        # Check every hour
        time.sleep(3600)  # 3600 seconds = 1 hour

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_violations, daemon=True)
cleanup_thread.start()

def log_alert(event_type, description, image_path=""):
    """Append an alert record to data/logs/alerts.json (keeps last 50)."""
    os.makedirs("data/logs", exist_ok=True)

    # Force absolute base path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    log_file = os.path.join(base_dir, "data/logs/alerts.json")
    print(f"🧾 Logging event to: {log_file}")  # Show actual write path

    # load existing (or empty list)
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                data = json.load(f)
        else:
            data = []
    except Exception as e:
        print(f"⚠️ Error reading log file: {e}")
        data = []

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
        "description": description,
        "image": image_path.replace("\\", "/") if image_path else ""
    }

    data.append(entry)
    # keep last 50
    data = data[-50:]
    try:
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saving log: {event_type} | {description} | {image_path}")
        try:
            from src.dashboard.app import push_event
            push_event(entry)
        except Exception as e:
            print(f"⚠️ Could not push live event: {e}")
    except Exception as e:
        print(f"❌ Error saving log: {e}")
        
# Import detectors
from src.utils.yolo_uniform_detector import is_wearing_blue_uniform
from src.detectors.anomaly_detector import detect_anomalies

# ========== Load known face encodings ==========
with open("models/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_rolls = list(data.keys())
known_encs = np.array([data[r] for r in known_rolls])
print(f"Loaded encodings for {len(known_rolls)} students: {known_rolls}")

# ========== Start webcam ==========
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera not found.")
    exit()

frame_count = 0

print("\n🎥 System Started — Press 'Q' to Quit\n")

# ========== Main Loop ==========
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured. Exiting.")
        break

    frame_count += 1
    if frame_count % 2 != 0:  # Process every 2nd frame (for performance)
        cv2.imshow("Live Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Convert frame to RGB (face_recognition expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    # ========== FACE DETECTION ==========
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    recognized_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Match face
        distances = np.linalg.norm(known_encs - face_encoding, axis=1)
        idx = np.argmin(distances)
        min_dist = distances[idx]

        if min_dist < 0.55:
            name = known_rolls[idx]
            label = f"{name}"
            color = (0, 255, 0)
        else:
            name = "Unknown"
            label = "Unknown Person"
            color = (0, 0, 255)
            
            # Save the frame with detection
            os.makedirs("data/unknown_faces", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            snapshot_path = f"data/unknown_faces/unknown_{ts}.jpg"
            cv2.imwrite(snapshot_path, frame)
            
            # Check if we should send a notification for this unknown face
            current_time = time.time()
            last_notif_time = last_notification.get('unknown_face', 0)
            
            if current_time - last_notif_time >= NOTIFICATION_COOLDOWN:
                # Send SMS alert for unknown face
                send_unknown_face_alert(
                    location="Main Campus Entrance",
                    image_url=snapshot_path  # In a real app, this would be a URL to the image
                )
                
                # Update last notification time
                last_notification['unknown_face'] = current_time
                
                # Log the alert
                log_alert("Unknown Person Detected", 
                         "An unknown person was detected at the entrance", 
                         snapshot_path)
            else:
                print("⏳ Skipping duplicate unknown face notification (cooldown active)")

        recognized_names.append(name)

        # Scale back up (since we resized frame)
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ========== UNIFORM DETECTION ==========
        if name != "Unknown":
            uniform_ok = is_wearing_blue_uniform(frame)
            if uniform_ok:
                uniform_text = "✅ Uniform OK"
                uniform_color = (0, 200, 0)
            else:
                uniform_text = "🚫 Not Wearing Uniform (Fine ₹700)"
                uniform_color = (0, 0, 255)

                # Save snapshot for uniform violation
                os.makedirs("data/anomalies", exist_ok=True)
                ts_uniform = time.strftime("%Y%m%d_%H%M%S")
                snapshot_path = f"data/anomalies/uniform_violation_{name}_{ts_uniform}.jpg"
                cv2.imwrite(snapshot_path, frame)
                log_alert("Uniform Violation", f"{name} not wearing uniform", snapshot_path)
                print("📝 Logged uniform violation to alerts.json")
                

                # Check if we should notify about this violation
                if should_notify_violation(name):
                    # Send email notification
                    violation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    student_email = f"{name}@mail.jiit.ac.in"
                    send_uniform_violation_email(
                        student_roll=name,
                        student_email=student_email,
                        violation_time=violation_time,
                        fine_amount=700
                    )
                    # Record this violation
                    record_violation(name)
                    print(f"📧 Email notification sent to {name}")
                else:
                    print(f"⏳ Skipping email for {name} - already notified within last {VIOLATION_COOLDOWN_HOURS} hours")

            print(f"Recognized: {name}")
            print(uniform_text)

            cv2.putText(
                frame,
                uniform_text,
                (left, bottom + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                uniform_color,
                2,
            )

    # ========== ANOMALY DETECTION ==========
    anomaly_detected, detections = detect_anomalies(frame)
    if anomaly_detected:
        print("\n🚨 SECURITY ALERT — HAZARDOUS OBJECT DETECTED! 🚨")
        os.makedirs("data/anomalies", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        
        # Get the most confident detection
        if detections:
            top_detection = max(detections, key=lambda x: x['conf'])
            item_name = top_detection.get('label', 'unknown object')
            confidence = top_detection.get('conf', 0) * 100
            
            # Create a unique key for this specific dangerous item
            item_key = f"dangerous_item_{item_name.lower()}"
            current_time = time.time()
            last_notif_time = last_notification.get(item_key, 0)
            
            # Check if we should send a notification for this item
            if current_time - last_notif_time >= NOTIFICATION_COOLDOWN:
                # Save the frame with detection
                snapshot_path = f"data/anomalies/dangerous_item_{item_name.lower()}_{ts}.jpg"
                cv2.imwrite(snapshot_path, frame)
                
                # Send SMS alert for dangerous item
                send_dangerous_item_alert(
                    item_name=item_name,
                    confidence=confidence,
                    location="Main Campus",
                    image_url=snapshot_path  # In a real app, this would be a URL to the image
                )
                
                # Update last notification time for this item
                last_notification[item_key] = current_time
                
                # Log the alert
                log_alert("Dangerous Item Detected", 
                         f"{item_name} detected with {confidence:.1f}% confidence", 
                         snapshot_path)
            else:
                print(f"⏳ Skipping duplicate {item_name} notification (cooldown active)")

        for d in detections:
            label = d["label"]
            conf = d["conf"]
            x1, y1, x2, y2 = d.get("box", (0, 0, 0, 0))  # get bounding box if provided

            # Draw red bounding box around detected anomaly
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                frame,
                f"{label.upper()} {conf*100:.1f}%",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            print(f"⚠️  {label.upper()} ({conf*100:.1f}% confidence)")

        # Save snapshot of current frame
        filename = f"data/anomalies/hazard_{ts}.jpg"
        cv2.imwrite(filename, frame)
        print(f"🖼️  Snapshot saved: {filename}")

        for d in detections:
            label = d["label"]
            conf = d["conf"]
            log_alert("Anomaly Detected", f"Hazard: {label} ({conf*100:.1f}%)", filename)
            print("📝 Logged anomaly detection to alerts.json")

    # ========== Display ==========
    cv2.imshow("Live Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n👋 Exiting surveillance system...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()