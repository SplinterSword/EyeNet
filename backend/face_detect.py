import cv2
import os
import psycopg2
from contextlib import contextmanager
from typing import Generator
from dotenv import load_dotenv
from imgbeddings import imgbeddings
from PIL import Image
import time
from threading import Thread, Event

load_dotenv()

haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

face_match = False

if not cap.isOpened():
    raise Exception("Could not open video device")

@contextmanager
def get_db_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(os.getenv("POSTGRES_SERVICE_URI"))
        yield conn
    except psycopg2.Error as e:
        raise RuntimeError(f"Database connection error: {e}")
    finally:
        if conn is not None:
            conn.close()

@contextmanager
def get_db_cursor(conn: psycopg2.extensions.connection) -> Generator[psycopg2.extensions.cursor, None, None]:
    """Context manager for database cursors"""
    cur = None
    try:
        cur = conn.cursor()
        yield cur
    except psycopg2.Error as e:
        conn.rollback()
        raise RuntimeError(f"Database cursor error: {e}")
    finally:
        if cur is not None:
            cur.close()

def detect_face(cropped_image):
    global face_match
    
    # Generate embedding
    try:
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        ibed = imgbeddings()
        embedding = ibed.to_embeddings(pil_image)
        string_representation = "["+ ",".join(str(x) for x in embedding[0].tolist()) +"]"
    except Exception as e:
        raise RuntimeError(f"Failed to generate face embedding: {str(e)}")

    with get_db_connection() as conn:
        with get_db_cursor(conn) as cursor:
            try:
                cursor.execute("SELECT * FROM registed_faces ORDER BY embeddings <-> %s LIMIT 1;", (string_representation,))
                rows = cursor.fetchall()
                for row in rows:
                    face_match = True
            except Exception as e:
                # If face detected but not recognised
                face_match = False
                raise RuntimeError(f"Failed to fetch face embeddings: {str(e)}")


# Event to signal threads to stop
stop_event = Event()

# Frame counter
frame_counter = 0
# Process every 90th frame
PROCESS_INTERVAL = 30

try:
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        
        # Only process every 90th frame
        if frame_counter % PROCESS_INTERVAL == 0:
            # Create a thread for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                face_match = False
                
            for (x, y, w, h) in faces:
                cropped_image = frame[y:y+h, x:x+w]
                
                thread = Thread(
                    target=detect_face, 
                    args=(cropped_image,),
                    daemon=True
                )
                thread.start()
        
        # Always update the display with the latest frame
        display_frame = frame.copy()
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
        if len(faces) == 0:
            face_match = False
                
        for (x, y, w, h) in faces:
            cropped_image = frame[y:y+h, x:x+w]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if face_match:
            cv2.putText(display_frame, "Face Match", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Face Not Match", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", display_frame)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break
        
except KeyboardInterrupt:
    stop_event.set()
    print("\nStopping face detection...")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

