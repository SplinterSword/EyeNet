import cv2
from threading import Thread, Event, Lock
from dotenv import load_dotenv
from imgbeddings import imgbeddings
from PIL import Image
from database import get_db_connection, get_db_cursor
import time

load_dotenv()

cv2.setUseOptimized(True)

haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if haar_cascade.empty():
    # Fallback to OpenCV's default haarcascades path
    cascade_path = getattr(cv2, 'data', None)
    if cascade_path is not None:
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

"""
Reusable face detection module:
- Initialize the embedding model once.
- Keep recognition off the caller thread using a worker.
- Reuse last computed recognition results for drawing.
Expose: process_frame(frame_bgr) -> display_frame
"""

# Initialize imgbeddings once (heavy). Reused across recognitions.
ibed = imgbeddings()
# Warm up once to avoid first-inference stutter
try:
    _dummy = Image.new('RGB', (32, 32), (0, 0, 0))
    _ = ibed.to_embeddings(_dummy)
except Exception:
    pass

# Shared state for rendering (written by worker, read by UI loop)
face_count = 0
last_results = []  # list of dicts: {rect, recognized, enrollment_no, confidence}
_worker_busy = False
_worker_lock = Lock()

# Rate-limit printing to avoid I/O stutter
_last_alert_ts = 0.0
_last_recog_ts = 0.0


def recognize_face(face_image):
    """
    Recognize a face by comparing it with the database
    Returns: (is_recognized, person_name, confidence)
    """
    try:
        # Convert and generate embedding (model initialized once globally)
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        embedding = ibed.to_embeddings(pil_image)
        string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
        
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cursor:
                # Get the closest match from the database
                cursor.execute("""
                    SELECT enrollment_no, embeddings <-> %s as distance 
                    FROM registed_faces 
                    ORDER BY distance 
                    LIMIT 1;
                """, (string_representation,))
                
                result = cursor.fetchone()
                
                if result and result[1] < 20:  # Threshold for face matching
                    return True, result[0], result[1]
                return False, "Unknown", result[1] if result else 1.0
                
    except Exception as e:
        print(f"Error in face recognition: {str(e)}")
        return False, "Error", 1.0


# Worker function to detect faces in a frame and recognize them.
def _process_faces_worker(frame_bgr):
    global face_count, last_results, _worker_busy
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        results = []
        for (x, y, w, h) in faces:
            cropped_face = frame_bgr[y:y+h, x:x+w]
            is_recognized, name, confidence = recognize_face(cropped_face)
            results.append({
                'rect': (x, y, w, h),
                'recognized': is_recognized,
                'enrollment_no': name,
                'confidence': confidence
            })

        # Safely publish results
        with _worker_lock:
            face_count = len(results)
            last_results = results
    finally:
        # Mark worker as idle
        with _worker_lock:
            _worker_busy = False

# Frame counter and interval
frame_counter = 0
# Process every N frames (tune for your hardware)
PROCESS_INTERVAL = 30

def process_frame(frame_bgr):
    """
    Public API: process one BGR frame and return an annotated frame.
    Heavy recognition is performed asynchronously in a background worker.
    """
    global frame_counter, _worker_busy

    frame_counter += 1

    # Kick off background recognition every N frames if worker idle
    if frame_counter % PROCESS_INTERVAL == 0:
        start_worker = False
        with _worker_lock:
            if not _worker_busy:
                _worker_busy = True
                start_worker = True
        if start_worker:
            Thread(target=_process_faces_worker, args=(frame_bgr.copy(),), daemon=True).start()

    # Prepare display frame
    display_frame = frame_bgr.copy()

    # Draw latest known results (possibly from a previous frame)
    with _worker_lock:
        faces_to_draw = list(last_results)
        faces_count_local = face_count
    for face in faces_to_draw:
        x, y, w, h = face['rect']
        color = (0, 255, 0) if face['recognized'] else (0, 0, 255)
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
        label = f"{face['enrollment_no']} ({face['confidence']:.2f})" if face['recognized'] else "Unknown"
        cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Rate-limit console I/O (max once per second per category)
        if not face['recognized']:
            if time.monotonic() - _last_alert_ts > 1.0:
                print(f"Security Alert: Unknown face detected at {x},{y} with confidence {face['confidence']:.2f}")
                globals()['_last_alert_ts'] = time.monotonic()
        else:
            if time.monotonic() - _last_recog_ts > 1.0:
                print(f"Face recognized: {face['enrollment_no']} with confidence {face['confidence']:.2f}")
                globals()['_last_recog_ts'] = time.monotonic()

    cv2.putText(display_frame, f"Faces: {faces_count_local}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return display_frame

def set_process_interval(n: int):
    global PROCESS_INTERVAL
    PROCESS_INTERVAL = max(1, int(n))

if __name__ == "__main__":
    # Standalone demo runner: open default camera and display annotated frames
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open video device")

    # Optional camera hints
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    # Recognition cadence (every N frames)
    set_process_interval(30)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            display = process_frame(frame)
            cv2.imshow("EyeNet - Face Detection", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nStopping face detection...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
