import cv2
from threading import Thread, Event
from dotenv import load_dotenv
from imgbeddings import imgbeddings
from PIL import Image
from threading import Thread, Event
from database import get_db_connection, get_db_cursor

load_dotenv()

haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

face_count = 0

if not cap.isOpened():
    raise Exception("Could not open video device")


def recognize_face(face_image):
    """
    Recognize a face by comparing it with the database
    Returns: (is_recognized, person_name, confidence)
    """
    try:
        # Convert and generate embedding
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        ibed = imgbeddings()
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
                
                if result and result[1] < 0.6:  # Threshold for face matching
                    return True, result[0], result[1]
                return False, "Unknown", result[1] if result else 1.0
                
    except Exception as e:
        print(f"Error in face recognition: {str(e)}")
        return False, "Error", 1.0


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

        face_results = []
        
        # Process faces every N frames
        if frame_counter % PROCESS_INTERVAL == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_count = len(faces)
            
            # Process each face in the current frame
            for (x, y, w, h) in faces:
                cropped_face = frame[y:y+h, x:x+w]
                is_recognized, name, confidence = recognize_face(cropped_face)
                face_results.append({
                    'rect': (x, y, w, h),
                    'recognized': is_recognized,
                    'enrollment_no': name,
                    'confidence': confidence
                })
        
        display_frame = frame.copy()
        
        for face in face_results:
            x, y, w, h = face['rect']
            
            color = (0, 255, 0) if face['recognized'] else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{face['enrollment_no']} ({face['confidence']:.2f})" if face['recognized'] else "Unknown"
            cv2.putText(display_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if not face['recognized']:
                # Send Alert
                print(f"Security Alert: Unknown face detected at {x},{y}")
            else:
                # Send Check for uniform
                print(f"Face recognized: {face['enrollment_no']} with confidence {face['confidence']:.2f}")

        
        cv2.putText(display_frame, f"Faces: {face_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Frame", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break
        
except KeyboardInterrupt:
    stop_event.set()
    print("\nStopping face detection...")
finally:
    cap.release()
    cv2.destroyAllWindows()

