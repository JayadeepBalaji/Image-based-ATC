from . import models
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
from datetime import datetime
import uuid
from typing import Optional, Dict, Any
import json

from .models.classifier import AnimalClassifier
from .models.database import Database
from .utils.image_processor import ImageProcessor
from .utils.serialization import to_py

app = FastAPI(title="Animal Type Classification API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
classifier = AnimalClassifier()
db = Database()
image_processor = ImageProcessor()

# Ensure upload directory exists
UPLOAD_DIR = "../uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def startup():
    """Initialize the database and load ML models on startup."""
    await db.init_db()
    classifier.load_model()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Animal Type Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "history": "/history",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": db.is_connected(),
            "classifier": classifier.is_loaded(),
            "image_processor": image_processor.is_ready()
        }
    }

@app.post("/upload")
async def classify_animal(
    file: UploadFile = File(...),
    species: str = Form(...),
    animal_name: Optional[str] = Form(None)
):
    """
    Upload and classify an animal image.
    
    Args:
        file: Image file (jpg, png, jpeg)
        species: Animal species ('cow' or 'buffalo')
        animal_name: Optional name for the animal
    
    Returns:
        Classification results with extracted features and productivity score
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (jpg, png, jpeg)"
            )

        # Validate species
        if species.lower() not in ['cow', 'buffalo']:
            raise HTTPException(
                status_code=400,
                detail="Species must be 'cow' or 'buffalo'"
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{file_id}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process image and extract features
        features = image_processor.extract_features(file_path, species.lower())

        if not features:
            # Clean up file
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail="Could not extract features from image. Please ensure the image shows a clear side view of the animal."
            )

        # Classify the animal
        classification_result = classifier.classify(features, species.lower())

        # Ensure JSON-safe structures (convert numpy types)
        safe_features = to_py(features)
        safe_classification = to_py(classification_result)

        # Prepare response data
        result = {
            "id": file_id,
            "timestamp": datetime.now().isoformat(),
            "animal_name": animal_name or f"Animal_{file_id[:8]}",
            "species": species.lower(),
            "original_filename": file.filename,
            "features": safe_features,
            "classification": safe_classification,
            "success": True
        }

        # Save to database
        await db.save_classification(result)

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except Exception:
            pass  # File cleanup is not critical

        return JSONResponse(content=jsonable_encoder(result))

    except HTTPException:
        raise
    except Exception as e:
        # Clean up file on error
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/history")
async def get_classification_history(limit: int = 10):
    """
    Get recent classification history.
    
    Args:
        limit: Maximum number of records to return
    
    Returns:
        List of recent classifications
    """
    try:
        history = await db.get_history(limit)
        return JSONResponse(content=jsonable_encoder({
            "history": history,
            "count": len(history),
            "limit": limit
        }))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve history: {str(e)}"
        )

@app.get("/stats")
async def get_statistics():
    """
    Get classification statistics.
    
    Returns:
        Statistics about classifications performed
    """
    try:
        stats = await db.get_statistics()
        return JSONResponse(content=jsonable_encoder(stats))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve statistics: {str(e)}"
        )

@app.delete("/history/{classification_id}")
async def delete_classification(classification_id: str):
    """
    Delete a specific classification record.
    
    Args:
        classification_id: ID of the classification to delete
    
    Returns:
        Success confirmation
    """
    try:
        success = await db.delete_classification(classification_id)
        if success:
            return {"message": "Classification deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail="Classification not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete classification: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )