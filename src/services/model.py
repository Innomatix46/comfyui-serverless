"""Model management service."""
import asyncio
import hashlib
import os
import aiofiles
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import structlog
from sqlalchemy.orm import Session
try:
    import torch
except ImportError:
    torch = None
import psutil
from pathlib import Path

from src.config.settings import settings
from src.core.database import SessionLocal
from src.models.database import Model, ModelType
from src.services.storage import storage_service
from src.utils.gpu import get_gpu_memory_info

logger = structlog.get_logger()


class ModelError(Exception):
    """Custom error for model service operations."""
    pass


class ModelService:
    """Model management and caching service."""
    
    def __init__(self):
        self._loaded_models = {}  # model_name -> model_object
        self._model_memory = {}   # model_name -> memory_usage_mb
        self._download_progress = {}  # model_name -> progress (0.0 to 1.0)
        self._download_tasks = {}  # model_name -> asyncio.Task
        self._access_times = {}   # model_name -> last_access_time
        
        # Model paths
        self.models_path = Path(settings.COMFYUI_MODELS_PATH)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different model types
        for model_type in ModelType:
            (self.models_path / model_type.value).mkdir(parents=True, exist_ok=True)
    
    async def is_model_available(self, model_name: str, model_type: str) -> bool:
        """Check if model is available (downloaded and ready)."""
        try:
            with SessionLocal() as db:
                model = db.query(Model).filter(
                    Model.name == model_name,
                    Model.type == model_type
                ).first()
                
                if not model:
                    return False
                
                # Check if file exists
                if model.file_path and os.path.exists(model.file_path):
                    model.is_available = True
                    model.last_used = datetime.utcnow()
                    db.commit()
                    return True
                else:
                    model.is_available = False
                    db.commit()
                    return False
                    
        except Exception as e:
            logger.error("Error checking model availability", model_name=model_name, error=str(e))
            return False
    
    async def get_model_status(self, model_name: str) -> Dict:
        """Get detailed model status."""
        with SessionLocal() as db:
            model = db.query(Model).filter(Model.name == model_name).first()
            
            if not model:
                return {"error": "Model not found"}
            
            return {
                "name": model.name,
                "type": model.type.value,
                "is_available": model.is_available,
                "is_loaded": model_name in self._loaded_models,
                "is_downloading": model.is_downloading,
                "download_progress": self._download_progress.get(model_name, 0.0),
                "file_size": model.file_size,
                "memory_usage_mb": self._model_memory.get(model_name, 0),
                "last_used": model.last_used.isoformat() if model.last_used else None
            }
    
    async def list_models(self) -> Dict:
        """List all models with their status."""
        with SessionLocal() as db:
            models = db.query(Model).all()
            
            model_statuses = []
            total_memory = 0
            
            for model in models:
                status = {
                    "name": model.name,
                    "type": model.type.value,
                    "is_available": model.is_available,
                    "is_loaded": model.name in self._loaded_models,
                    "is_downloading": model.is_downloading,
                    "download_progress": self._download_progress.get(model.name, 0.0),
                    "memory_usage_mb": self._model_memory.get(model.name, 0),
                    "last_used": model.last_used
                }
                model_statuses.append(status)
                total_memory += status["memory_usage_mb"]
            
            # Get GPU memory info
            gpu_info = get_gpu_memory_info()
            
            return {
                "models": model_statuses,
                "total_memory_usage_mb": total_memory,
                "available_memory_mb": gpu_info.get("free_mb", 0),
                "gpu_memory_total_mb": gpu_info.get("total_mb", 0),
                "loaded_models_count": len(self._loaded_models)
            }
    
    async def download_model(
        self,
        model_name: str,
        model_type: ModelType,
        download_url: str,
        description: str = None
    ) -> bool:
        """Download a model from URL."""
        if model_name in self._download_tasks:
            logger.info("Model download already in progress", model_name=model_name)
            return True
        
        try:
            # Create model record
            with SessionLocal() as db:
                existing_model = db.query(Model).filter(
                    Model.name == model_name,
                    Model.type == model_type
                ).first()
                
                if existing_model:
                    if existing_model.is_available:
                        logger.info("Model already available", model_name=model_name)
                        return True
                    model = existing_model
                else:
                    model = Model(
                        name=model_name,
                        type=model_type,
                        description=description,
                        download_url=download_url
                    )
                    db.add(model)
                
                model.is_downloading = True
                model.download_progress = 0.0
                db.commit()
                db.refresh(model)
            
            # Start download task
            task = asyncio.create_task(
                self._download_model_task(model_name, model_type, download_url)
            )
            self._download_tasks[model_name] = task
            
            return True
            
        except Exception as e:
            logger.error("Failed to start model download", model_name=model_name, error=str(e))
            return False
    
    async def queue_download(self, model_name: str, model_type: str) -> bool:
        """Queue model for download if not available."""
        if await self.is_model_available(model_name, model_type):
            return True
        
        # Get download URL from model registry or default sources
        download_url = await self._get_model_download_url(model_name, model_type)
        if not download_url:
            logger.error("No download URL found for model", model_name=model_name, type=model_type)
            return False
        
        return await self.download_model(model_name, model_type, download_url)
    
    async def load_model(self, model_name: str, model_type: str) -> bool:
        """Load model into memory."""
        if model_name in self._loaded_models:
            self._access_times[model_name] = datetime.utcnow()
            return True
        
        if not await self.is_model_available(model_name, model_type):
            logger.error("Model not available for loading", model_name=model_name)
            return False
        
        try:
            # Check memory constraints
            if not await self._check_memory_constraints(model_name):
                await self._free_memory()
            
            # Load model based on type
            model_path = self._get_model_path(model_name, model_type)
            
            if model_type == ModelType.CHECKPOINT:
                model_obj = await self._load_checkpoint(model_path)
            elif model_type == ModelType.LORA:
                model_obj = await self._load_lora(model_path)
            elif model_type == ModelType.VAE:
                model_obj = await self._load_vae(model_path)
            else:
                # Generic loading
                model_obj = await self._load_generic_model(model_path)
            
            if model_obj:
                self._loaded_models[model_name] = model_obj
                self._access_times[model_name] = datetime.utcnow()
                
                # Estimate memory usage
                memory_usage = await self._estimate_model_memory(model_obj)
                self._model_memory[model_name] = memory_usage
                
                # Update database
                with SessionLocal() as db:
                    model = db.query(Model).filter(Model.name == model_name).first()
                    if model:
                        model.last_used = datetime.utcnow()
                        model.memory_usage_mb = memory_usage
                        model.usage_count += 1
                        db.commit()
                
                logger.info(
                    "Model loaded successfully",
                    model_name=model_name,
                    memory_mb=memory_usage
                )
                return True
            
        except Exception as e:
            logger.error("Failed to load model", model_name=model_name, error=str(e))
            return False
        
        return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload model from memory."""
        if model_name not in self._loaded_models:
            return True
        
        try:
            del self._loaded_models[model_name]
            memory_freed = self._model_memory.pop(model_name, 0)
            self._access_times.pop(model_name, None)
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(
                "Model unloaded",
                model_name=model_name,
                memory_freed_mb=memory_freed
            )
            return True
            
        except Exception as e:
            logger.error("Failed to unload model", model_name=model_name, error=str(e))
            return False
    
    async def cleanup_unused_models(self, max_age_hours: int = 1) -> int:
        """Clean up models that haven't been used recently."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        models_unloaded = 0
        
        models_to_unload = []
        for model_name, last_access in self._access_times.items():
            if last_access < cutoff_time:
                models_to_unload.append(model_name)
        
        for model_name in models_to_unload:
            if await self.unload_model(model_name):
                models_unloaded += 1
        
        logger.info("Cleanup completed", models_unloaded=models_unloaded)
        return models_unloaded
    
    async def _download_model_task(
        self,
        model_name: str,
        model_type: ModelType,
        download_url: str
    ):
        """Background task for downloading models."""
        try:
            # Determine file path
            file_path = self.models_path / model_type.value / f"{model_name}"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress
                            if total_size > 0:
                                progress = downloaded / total_size
                                self._download_progress[model_name] = progress
                                
                                # Update database
                                with SessionLocal() as db:
                                    model = db.query(Model).filter(
                                        Model.name == model_name
                                    ).first()
                                    if model:
                                        model.download_progress = progress
                                        db.commit()
            
            # Verify file integrity
            file_hash = await self._calculate_file_hash(file_path)
            
            # Update model record
            with SessionLocal() as db:
                model = db.query(Model).filter(
                    Model.name == model_name,
                    Model.type == model_type
                ).first()
                
                if model:
                    model.is_downloading = False
                    model.is_available = True
                    model.file_path = str(file_path)
                    model.file_size = os.path.getsize(file_path)
                    model.file_hash = file_hash
                    model.download_progress = 1.0
                    db.commit()
            
            # Clean up
            self._download_progress.pop(model_name, None)
            self._download_tasks.pop(model_name, None)
            
            logger.info(
                "Model download completed",
                model_name=model_name,
                file_size=os.path.getsize(file_path)
            )
            
        except Exception as e:
            logger.error("Model download failed", model_name=model_name, error=str(e))
            
            # Update database with error
            with SessionLocal() as db:
                model = db.query(Model).filter(
                    Model.name == model_name,
                    Model.type == model_type
                ).first()
                
                if model:
                    model.is_downloading = False
                    model.is_available = False
                    db.commit()
            
            # Clean up
            self._download_progress.pop(model_name, None)
            self._download_tasks.pop(model_name, None)
    
    def _get_model_path(self, model_name: str, model_type: str) -> Path:
        """Get the file path for a model."""
        return self.models_path / model_type / model_name
    
    async def _get_model_download_url(self, model_name: str, model_type: str) -> Optional[str]:
        """Get download URL for a model from registry."""
        # This would integrate with model registries like Hugging Face, CivitAI, etc.
        # For now, return None to indicate no URL found
        return None
    
    async def _check_memory_constraints(self, model_name: str) -> bool:
        """Check if loading model would exceed memory limits."""
        gpu_info = get_gpu_memory_info()
        if not gpu_info:
            return True  # Skip check if GPU info unavailable
        
        # Estimate model memory usage (simplified)
        estimated_memory = 2000  # 2GB default estimate
        
        current_usage = sum(self._model_memory.values())
        max_memory = gpu_info.get("total_mb", 0) * settings.GPU_MEMORY_FRACTION
        
        return (current_usage + estimated_memory) <= max_memory
    
    async def _free_memory(self):
        """Free memory by unloading least recently used models."""
        if not self._access_times:
            return
        
        # Sort by access time (oldest first)
        models_by_access = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        # Unload oldest models until we have enough memory
        for model_name, _ in models_by_access:
            await self.unload_model(model_name)
            
            # Check if we have enough memory now
            if await self._check_memory_constraints("dummy"):
                break
    
    async def _load_checkpoint(self, model_path: Path):
        """Load checkpoint model."""
        # Implementation depends on specific checkpoint format
        # This is a placeholder
        return {"type": "checkpoint", "path": str(model_path)}
    
    async def _load_lora(self, model_path: Path):
        """Load LoRA model."""
        # Implementation depends on LoRA format
        return {"type": "lora", "path": str(model_path)}
    
    async def _load_vae(self, model_path: Path):
        """Load VAE model."""
        # Implementation depends on VAE format
        return {"type": "vae", "path": str(model_path)}
    
    async def _load_generic_model(self, model_path: Path):
        """Load generic model."""
        return {"type": "generic", "path": str(model_path)}
    
    async def _estimate_model_memory(self, model_obj) -> float:
        """Estimate model memory usage in MB."""
        # This is a simplified estimation
        # In practice, you'd inspect the actual model structure
        return 1500.0  # Default 1.5GB estimate
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(file_path, "rb") as f:
            async for chunk in f:
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


# Global model service instance
model_service = ModelService()
