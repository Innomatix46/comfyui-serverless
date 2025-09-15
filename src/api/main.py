"""Main FastAPI application."""
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import time
import structlog

from src.config.settings import settings
from src.core.database import create_tables
from src.api.middleware import (
    LoggingMiddleware,
    AuthenticationMiddleware,
    RateLimitMiddleware,
    MetricsMiddleware,
    SecurityMiddleware,
)
from src.api.routers import (
    auth_router,
    workflow_router,
    models_router,
    files_router,
    health_router,
    metrics_router
)
from src.services.monitoring import monitoring_service
from src.services.cleanup import cleanup_service

# Configure structured logging
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting ComfyUI Serverless API", version=settings.API_VERSION)
    # Basic production safety checks
    if not settings.DEBUG and settings.SECRET_KEY == "your-secret-key-change-in-production":
        raise RuntimeError("SECURITY: SECRET_KEY must be set in production")
    
    # Initialize database
    create_tables()
    logger.info("Database tables created")
    
    # Start background services
    monitoring_service.start()
    cleanup_service.start()
    logger.info("Background services started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ComfyUI Serverless API")
    
    # Stop background services
    monitoring_service.stop()
    cleanup_service.stop()
    logger.info("Background services stopped")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Add middleware (order matters - added in reverse order of execution)
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(SecurityMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if settings.DEBUG else ["yourdomain.com", "*.yourdomain.com"]
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(workflow_router, prefix="/workflows", tags=["workflows"])
app.include_router(models_router, prefix="/models", tags=["models"])
app.include_router(files_router, prefix="/files", tags=["files"])
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "docs_url": "/docs" if settings.DEBUG else None,
        "status": "healthy",
        "timestamp": time.time()
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with proper status codes."""
    from starlette.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": request.url.path,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler with proper status code."""
    from starlette.responses import JSONResponse

    logger.error(
        "Unhandled exception",
        exception=str(exc),
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An internal error occurred",
            "timestamp": time.time(),
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4
    )
