import cv2
import numpy as np
from imgbeddings import imgbeddings
from dotenv import load_dotenv
from PIL import Image
from database import get_db_connection, get_db_cursor

load_dotenv()

# Constants
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Initialize face detection
haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)


def register_face(img: np.ndarray, enrollment_no: str) -> bool:
    """
    Register a face by processing the image and storing the embedding in the database.
    
    Args:
        img: Input image as a numpy array in BGR format
        enrollment_no: Unique identifier for the person
        
    Returns:
        bool: True if registration was successful
        
    Raises:
        ValueError: If no faces are detected in the image or invalid input
        RuntimeError: If there's an error with the database operation
    """
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Expected image to be a numpy array, got {type(img).__name__}")
    
    if not enrollment_no or not isinstance(enrollment_no, str):
        raise ValueError("Valid enrollment number is required")
    
    # Convert image to grayscale for face detection
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")
    
    # Detect faces in the image
    faces = haar_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=1.05, 
        minNeighbors=2, 
        minSize=(100, 100)
    )
    
    if len(faces) == 0:
        raise ValueError("No faces detected in the provided image")
    
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Process each detected face (though we typically expect just one)
                for (x, y, w, h) in faces[:1]:  # Limit to first face found
                    # Extract face region with boundary checks
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x1 + w), min(h, y1 + h)
                    
                    if x1 >= x2 or y1 >= y2:
                        continue  # Skip invalid face regions
                        
                    cropped_image = img[y1:y2, x1:x2]
                    
                    # Generate embedding
                    try:
                        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                        
                        ibed = imgbeddings()
                        embedding = ibed.to_embeddings(pil_image)
                        embedding_list = embedding[0].tolist()
                    except Exception as e:
                        raise RuntimeError(f"Failed to generate face embedding: {str(e)}")
                    
                    # Store in database
                    try:
                        cur.execute(
                            """
                            INSERT INTO registed_faces (enrollment_no, embeddings)
                            VALUES (%s, %s)
                            ON CONFLICT (enrollment_no) 
                            DO UPDATE SET embeddings = EXCLUDED.embeddings
                            """,
                            (enrollment_no, embedding_list)
                        )
                        conn.commit()
                    except psycopg2.Error as e:
                        conn.rollback()
                        raise RuntimeError(f"Database error: {str(e)}")
        
        return True
        
    except Exception as e:
        # Re-raise with appropriate error type
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Unexpected error during face registration: {str(e)}")

