# src/encoders/encode_faces.py
import os
import pickle
import face_recognition

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "students")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/face_encodings.pkl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Dictionary to store face encodings with roll numbers as keys
encodings = {}

# Get all jpeg/jpg files in the students directory
image_files = [f for f in os.listdir(DATA_DIR) 
               if f.lower().endswith(('.jpg', '.jpeg'))]

for img_file in sorted(image_files):
    # Extract roll number by removing the file extension
    roll = os.path.splitext(img_file)[0]
    img_path = os.path.join(DATA_DIR, img_file)
    
    try:
        # Load and encode the face
        image = face_recognition.load_image_file(img_path)
        faces = face_recognition.face_encodings(image)
        
        if not faces:
            print(f"No face found in {img_file}. Try a clearer photo.")
            continue
            
        encodings[roll] = faces[0].tolist()
        print(f"Encoded {roll} successfully.")
        
    except Exception as e:
        print(f"Error processing {img_file}: {e}")

# Save the encodings to a file
with open(OUT, "wb") as f:
    pickle.dump(encodings, f)

print(f"\n✅ Encoded {len(encodings)} students. Saved to {OUT}")
print(f"Processed files: {', '.join(encodings.keys())}")