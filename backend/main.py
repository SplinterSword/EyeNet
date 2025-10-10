import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from face_register import register_face
from pydantic import BaseModel, Field, HttpUrl
import base64
import numpy as np
import cv2
import json
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class RegisterRequest(BaseModel):
    enrollment_no: str = Field(..., min_length=1, description="Unique enrollment identifier")
    image: str = Field(..., min_length=100, description="Base64 encoded image string")

class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    status_code: int

@app.exception_handler(ValueError)
async def value_error_exception_handler(request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "validation_error", "status_code": 400}
    )

@app.exception_handler(RuntimeError)
async def runtime_error_exception_handler(request, exc: RuntimeError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "error_type": "runtime_error", "status_code": 500}
    )

@app.post("/register_face", response_model=Dict[str, str], responses={
    200: {"description": "Face registered successfully"},
    400: {"model": ErrorResponse, "description": "Invalid request or image data"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
})
async def register_face_endpoint(request: RegisterRequest) -> Dict[str, str]:
    """
    Register a face by processing the provided image and storing the face embedding.
    
    - **enrollment_no**: Unique identifier for the person
    - **image**: Base64 encoded image string containing a face
    """
    logger.info(f"Processing face registration for enrollment: {request.enrollment_no}")
    
    try:
        # Decode the base64 image
        try:
            image_bytes = base64.b64decode(request.image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image. Please ensure the image is in a valid format (JPEG, PNG, etc.)")
                
        except Exception as e:
            logger.error(f"Image decoding error: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")
        
        # Register the face
        try:
            register_face(img, request.enrollment_no)
            logger.info(f"Successfully registered face for enrollment: {request.enrollment_no}")
            return {
                "status": "success", 
                "message": "Face registered successfully"
            }
            
        except ValueError as ve:
            logger.warning(f"Validation error during face registration: {str(ve)}")
            raise
            
        except Exception as e:
            logger.error(f"Error during face registration: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to process face registration: {str(e)}")
            
    except Exception as e:
        # This will be handled by the exception handlers above
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
