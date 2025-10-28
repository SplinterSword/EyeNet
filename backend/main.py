import uvicorn
import logging
from fastapi import FastAPI, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from face_register import register_face
from pydantic import BaseModel, Field
import base64
import numpy as np
import cv2
from typing import Dict, List
from threading import Thread, Event
from face_detect import process_frame, set_process_interval

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
async def register_face_endpoint(
    enrollment_no: str = File(..., description="Unique enrollment identifier"),
    image: UploadFile = File(..., description="Image file containing a face")
) -> Dict[str, str]:
    """
    Register a face by processing the provided image and storing the face embedding.
    
    - **enrollment_no**: Unique identifier for the person
    - **image**: Base64 encoded image string containing a face
    """
    logger.info(f"Processing face registration for enrollment: {enrollment_no}")
    
    try:
        # Decode the base64 image
        try:
            image_bytes = await image.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image. Please ensure the image is in a valid format (JPEG, PNG, etc.)")
                
        except Exception as e:
            logger.error(f"Image decoding error: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")
        
        # Register the face
        try:
            register_face(img, enrollment_no)
            logger.info(f"Successfully registered face for enrollment: {enrollment_no}")
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

# ======================
# Multi-camera management
# ======================

_camera_caps: List[cv2.VideoCapture] = []
_camera_threads: List[Thread] = []
_camera_stops: List[Event] = []

def _camera_worker(cap: cv2.VideoCapture, stop_evt: Event, cam_name: str):
    logger.info(f"Camera worker started: {cam_name}")
    try:
        while not stop_evt.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning(f"Camera {cam_name}: frame read failed; stopping worker")
                break
            # Process frame (annotated frame returned but not displayed in server)
            try:
                _ = process_frame(frame)
            except Exception as e:
                logger.exception(f"Error processing frame from {cam_name}: {e}")
                # continue loop; do not crash worker
    finally:
        try:
            cap.release()
        except Exception:
            pass
        logger.info(f"Camera worker stopped: {cam_name}")


def _detect_cameras(max_indices: int = 6) -> List[int]:
    found = []
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            found.append(idx)
        cap.release()
    return found


@app.on_event("startup")
def start_camera_processing():
    # Optional: allow tuning via env later
    set_process_interval(30)  # process every 30 frames by default

    indices = _detect_cameras()
    if not indices:
        logger.warning("No video sources detected at startup")
        return

    logger.info(f"Detected video sources: {indices}")
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            logger.warning(f"Failed to open camera index {idx}")
            continue
        stop_evt = Event()
        th = Thread(target=_camera_worker, args=(cap, stop_evt, f"camera-{idx}"), daemon=True)
        _camera_caps.append(cap)
        _camera_stops.append(stop_evt)
        _camera_threads.append(th)
        th.start()


@app.on_event("shutdown")
def stop_camera_processing():
    for evt in _camera_stops:
        try:
            evt.set()
        except Exception:
            pass
    for th in _camera_threads:
        try:
            th.join(timeout=2.0)
        except Exception:
            pass
    for cap in _camera_caps:
        try:
            cap.release()
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
