import os
import cv2
import psycopg2
import numpy as np
from typing import Tuple, List
from imgbeddings import imgbeddings
from psycopg2.extensions import cursor, connection

# Constants
HAAR_CASCADE_PATH: str = "haarcascade_frontalface_default.xml"

# Initialize face detection
haar_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Database connection
try:
    conn: connection = psycopg2.connect(os.getenv("POSTGRES_SERVICE_URI"))
except psycopg2.Error as e:
    raise RuntimeError(f"Failed to connect to the database: {e}")

def register_face(img: np.ndarray, enrollment_no: str) -> bool:
    """
    Register a face by processing the image and storing the embedding in the database.
    
    Args:
        img: Input image as a numpy array in BGR format
        enrollment_no: Unique identifier for the person
        
    Returns:
        bool: True if registration was successful, False otherwise
        
    Raises:
        ValueError: If no faces are detected in the image
        psycopg2.Error: If there's an error with the database operation
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected image to be a numpy array, got {type(img).__name__}")
    if not isinstance(enrollment_no, str):
        raise TypeError(f"Expected enrollment_no to be a string, got {type(enrollment_no).__name__}")
        
    # Convert image to grayscale for face detection
    gray_img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Detect faces in the image
    faces: Tuple[Tuple[int, int, int, int], ...] = haar_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=1.05, 
        minNeighbors=2, 
        minSize=(100, 100)
    )
    
    if len(faces) == 0:
        raise ValueError("No faces detected in the provided image")
    
    try:
        cur: cursor = conn.cursor()
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            cropped_image: np.ndarray = img[y:y + h, x:x + w]
            
            # Generate embedding
            ibed: imgbeddings = imgbeddings()
            embedding: List[List[float]] = ibed.to_embeddings(cropped_image)
            
            # Store in database
            cur.execute(
                """
                INSERT INTO registed_faces (enrollment_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (enrollment_id) 
                DO UPDATE SET embedding = EXCLUDED.embedding
                """,
                (enrollment_no, embedding[0].tolist())
            )
        
        # Commit the transaction
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        raise
    finally:
        if 'cur' in locals():
            cur.close()

