"""Model management API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from typing import List, Optional

from src.core.database import get_db
from src.models.schemas import ModelInfo, ModelStatus, ModelListResponse
from src.models.database import User
from src.services.model import model_service
from src.services.auth import get_current_user

router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    available_only: bool = Query(False, description="Show only available models"),
    current_user: User = Depends(get_current_user)
):
    """List all models with their status."""
    try:
        models_data = await model_service.list_models()
        models = models_data["models"]
        
        # Apply filters
        if model_type:
            models = [m for m in models if m["type"] == model_type]
        
        if available_only:
            models = [m for m in models if m["is_available"]]
        
        return ModelListResponse(
            models=[
                ModelStatus(
                    name=m["name"],
                    type=m["type"],
                    is_loaded=m["is_loaded"],
                    is_downloading=m["is_downloading"],
                    download_progress=m["download_progress"],
                    last_used=m["last_used"],
                    memory_usage_mb=m["memory_usage_mb"]
                )
                for m in models
            ],
            total_memory_usage_mb=models_data["total_memory_usage_mb"],
            available_memory_mb=models_data["available_memory_mb"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/{model_name}")
async def get_model_status(
    model_name: str = Path(..., description="Model name"),
    current_user: User = Depends(get_current_user)
):
    """Get detailed status for a specific model."""
    try:
        status = await model_service.get_model_status(model_name)
        
        if "error" in status:
            raise HTTPException(
                status_code=404,
                detail=status["error"]
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )


@router.post("/{model_name}/download")
async def download_model(
    model_name: str = Path(..., description="Model name"),
    model_type: str = Query(..., description="Model type"),
    download_url: str = Query(..., description="Download URL"),
    description: Optional[str] = Query(None, description="Model description"),
    current_user: User = Depends(get_current_user)
):
    """Download a model from URL."""
    try:
        # Validate model type
        valid_types = ["checkpoint", "lora", "embedding", "vae", "controlnet", "upscaler"]
        if model_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type. Must be one of: {', '.join(valid_types)}"
            )
        
        success = await model_service.download_model(
            model_name=model_name,
            model_type=model_type,
            download_url=download_url,
            description=description
        )
        
        if success:
            return {
                "message": "Model download started",
                "model_name": model_name,
                "model_type": model_type
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to start model download"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download model: {str(e)}"
        )


@router.post("/{model_name}/load")
async def load_model(
    model_name: str = Path(..., description="Model name"),
    model_type: str = Query(..., description="Model type"),
    current_user: User = Depends(get_current_user)
):
    """Load model into memory."""
    try:
        success = await model_service.load_model(model_name, model_type)
        
        if success:
            return {
                "message": "Model loaded successfully",
                "model_name": model_name,
                "model_type": model_type
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post("/{model_name}/unload")
async def unload_model(
    model_name: str = Path(..., description="Model name"),
    current_user: User = Depends(get_current_user)
):
    """Unload model from memory."""
    try:
        success = await model_service.unload_model(model_name)
        
        if success:
            return {
                "message": "Model unloaded successfully",
                "model_name": model_name
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to unload model"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_unused_models(
    max_age_hours: int = Query(1, ge=1, le=24, description="Maximum age in hours"),
    current_user: User = Depends(get_current_user)
):
    """Clean up models that haven't been used recently."""
    try:
        models_unloaded = await model_service.cleanup_unused_models(max_age_hours)
        
        return {
            "message": "Cleanup completed",
            "models_unloaded": models_unloaded
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup models: {str(e)}"
        )


@router.get("/{model_name}/download-progress")
async def get_download_progress(
    model_name: str = Path(..., description="Model name"),
    current_user: User = Depends(get_current_user)
):
    """Get download progress for a model."""
    try:
        status = await model_service.get_model_status(model_name)
        
        if "error" in status:
            raise HTTPException(
                status_code=404,
                detail=status["error"]
            )
        
        return {
            "model_name": model_name,
            "is_downloading": status["is_downloading"],
            "download_progress": status["download_progress"],
            "is_available": status["is_available"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get download progress: {str(e)}"
        )