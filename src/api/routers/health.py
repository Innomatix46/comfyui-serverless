"""Health check API endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import time
from datetime import datetime

from src.core.database import get_db, db_manager
from src.models.schemas import HealthStatus, DetailedHealthStatus, ServiceHealthStatus
from src.services.monitoring import monitoring_service
from src.config.settings import settings

router = APIRouter()

# Track application start time
app_start_time = time.time()


@router.get("/", response_model=HealthStatus)
async def health_check():
    """Basic health check endpoint."""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.API_VERSION,
        uptime_seconds=time.time() - app_start_time
    )


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check():
    """Detailed health check with service statuses."""
    services = []
    overall_status = "healthy"
    
    # Check database connectivity
    db_start = time.time()
    db_healthy = db_manager.health_check()
    db_response_time = (time.time() - db_start) * 1000  # Convert to ms
    
    services.append(ServiceHealthStatus(
        name="database",
        status="healthy" if db_healthy else "unhealthy",
        response_time_ms=db_response_time,
        error=None if db_healthy else "Database connection failed"
    ))
    
    if not db_healthy:
        overall_status = "unhealthy"
    
    # Check Redis connectivity
    redis_start = time.time()
    try:
        from src.services.workflow import workflow_service
        workflow_service.redis_client.ping()
        redis_healthy = True
        redis_error = None
    except Exception as e:
        redis_healthy = False
        redis_error = str(e)
    
    redis_response_time = (time.time() - redis_start) * 1000
    
    services.append(ServiceHealthStatus(
        name="redis",
        status="healthy" if redis_healthy else "unhealthy",
        response_time_ms=redis_response_time,
        error=redis_error
    ))
    
    if not redis_healthy:
        overall_status = "unhealthy"
    
    # Check ComfyUI connectivity
    comfyui_start = time.time()
    try:
        from src.services.comfyui import ComfyUIClient
        client = ComfyUIClient()
        comfyui_healthy = await client.health_check()
        comfyui_error = None
    except Exception as e:
        comfyui_healthy = False
        comfyui_error = str(e)
    
    comfyui_response_time = (time.time() - comfyui_start) * 1000
    
    services.append(ServiceHealthStatus(
        name="comfyui",
        status="healthy" if comfyui_healthy else "unhealthy",
        response_time_ms=comfyui_response_time,
        error=comfyui_error
    ))
    
    if not comfyui_healthy:
        overall_status = "degraded"  # ComfyUI issues are not critical for API
    
    # Check storage service
    storage_start = time.time()
    try:
        from src.services.storage import storage_service
        # Simple check - try to list a non-existent file
        await storage_service.get_file_info("health-check", 0)
        storage_healthy = True
        storage_error = None
    except Exception as e:
        # Expected to fail, but service is responsive
        if "not found" in str(e).lower():
            storage_healthy = True
            storage_error = None
        else:
            storage_healthy = False
            storage_error = str(e)
    
    storage_response_time = (time.time() - storage_start) * 1000
    
    services.append(ServiceHealthStatus(
        name="storage",
        status="healthy" if storage_healthy else "unhealthy",
        response_time_ms=storage_response_time,
        error=storage_error
    ))
    
    if not storage_healthy:
        overall_status = "unhealthy"
    
    # Get system information
    system_info = await monitoring_service.get_system_info()
    
    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.API_VERSION,
        uptime_seconds=time.time() - app_start_time,
        services=services,
        system=system_info
    )


@router.get("/readiness")
async def readiness_check():
    """Kubernetes readiness probe."""
    # Check if critical services are ready
    try:
        # Check database
        if not db_manager.health_check():
            return {"status": "not ready", "reason": "Database not available"}, 503
        
        # Check Redis
        from src.services.workflow import workflow_service
        workflow_service.redis_client.ping()
        
        return {"status": "ready"}
        
    except Exception as e:
        return {"status": "not ready", "reason": str(e)}, 503


@router.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe."""
    # Simple check - if the application is running, it's alive
    return {"status": "alive", "timestamp": datetime.utcnow()}


@router.get("/metrics/summary")
async def health_metrics():
    """Health-related metrics summary."""
    try:
        metrics = await monitoring_service.get_health_metrics()
        return {
            "status": "healthy",
            "metrics": metrics,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }, 500