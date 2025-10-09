import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from face_register import register_face
from PIL import Image
import cv2
import base64
import io
from typing import Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class RegisterRequest(BaseModel):
    enrollment_no: str
    image: str  # Base64 encoded image string

@app.post("/register_face")
async def register_face_endpoint(request: Dict[str, Any]):
    try:
        enrollment_no = request.get('enrollment_no')
        base64_image = request.get('image')
        
        if not enrollment_no or not base64_image:
            raise HTTPException(status_code=400, detail="Missing enrollment_no or image in request")
        
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
            
        img_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_data))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        register_face(img, enrollment_no)
        
        return {"status": "success", "message": "Face registered successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
