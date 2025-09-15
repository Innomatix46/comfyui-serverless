"""Application configuration settings."""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    API_TITLE: str = "ComfyUI Serverless API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Serverless ComfyUI API for AI image generation"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/comfyui_serverless"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_QUEUE_DB: int = 1
    REDIS_CACHE_DB: int = 2
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_ROUTES: dict = {
        "workflow.execute": {"queue": "workflow"},
        "model.download": {"queue": "model"},
        "cleanup.files": {"queue": "cleanup"}
    }
    
    # Storage
    STORAGE_TYPE: str = "s3"  # s3, local, gcs
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET: Optional[str] = None
    AWS_S3_REGION: str = "us-east-1"
    STORAGE_BASE_PATH: str = "/tmp/comfyui"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # ComfyUI
    COMFYUI_PATH: str = "/opt/ComfyUI"
    COMFYUI_MODELS_PATH: str = "/opt/ComfyUI/models"
    COMFYUI_OUTPUT_PATH: str = "/opt/ComfyUI/output"
    COMFYUI_TEMP_PATH: str = "/opt/ComfyUI/temp"
    COMFYUI_API_URL: str = "http://localhost:8188"
    
    # GPU Configuration
    GPU_MEMORY_FRACTION: float = 0.8
    MAX_GPU_MEMORY_GB: int = 24
    ENABLE_MODEL_OFFLOAD: bool = True
    MODEL_CACHE_SIZE_GB: int = 10
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Workflow Execution
    MAX_EXECUTION_TIME_MINUTES: int = 30
    MAX_QUEUE_SIZE: int = 100
    WORKER_CONCURRENCY: int = 2
    PRIORITY_QUEUES: List[str] = ["high", "normal", "low"]
    
    # File Cleanup
    CLEANUP_INTERVAL_HOURS: int = 1
    TEMP_FILE_RETENTION_HOURS: int = 24
    RESULT_RETENTION_DAYS: int = 7
    
    # Webhooks
    WEBHOOK_TIMEOUT_SECONDS: int = 30
    WEBHOOK_RETRY_ATTEMPTS: int = 3
    WEBHOOK_RETRY_DELAY_SECONDS: int = 5
    
    # Health Checks
    HEALTH_CHECK_INTERVAL_SECONDS: int = 30
    HEALTH_CHECK_TIMEOUT_SECONDS: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()