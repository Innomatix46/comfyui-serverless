"""SQLAlchemy database models."""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, Float, ForeignKey, Index, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from src.core.database import Base


class WorkflowStatus(enum.Enum):
    """Workflow execution status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(enum.Enum):
    """Task priority enum."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ModelType(enum.Enum):
    """Model type enum."""
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    EMBEDDING = "embedding"
    VAE = "vae"
    CONTROLNET = "controlnet"
    UPSCALER = "upscaler"


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    workflow_executions = relationship("WorkflowExecution", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")


class APIKey(Base):
    """API key model."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    # Indexes
    __table_args__ = (
        Index("idx_api_keys_user_active", "user_id", "is_active"),
    )


class WorkflowExecution(Base):
    """Workflow execution model."""
    __tablename__ = "workflow_executions"
    
    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    workflow_definition = Column(JSON, nullable=False)
    status = Column(Enum(WorkflowStatus), default=WorkflowStatus.PENDING, index=True)
    priority = Column(Enum(Priority), default=Priority.NORMAL, index=True)
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    timeout_at = Column(DateTime(timezone=True))
    
    # Results
    outputs = Column(JSON)
    error_message = Column(Text)
    logs = Column(JSON)  # Array of log messages
    
    # Metadata
    webhook_url = Column(String(500))
    execution_metadata = Column("metadata", JSON)
    queue_position = Column(Integer)
    worker_id = Column(String(100))
    execution_stats = Column(JSON)  # CPU, memory, GPU usage stats
    
    # Relationships
    user = relationship("User", back_populates="workflow_executions")
    
    # Indexes
    __table_args__ = (
        Index("idx_workflow_status_created", "status", "created_at"),
        Index("idx_workflow_user_status", "user_id", "status"),
        Index("idx_workflow_priority_created", "priority", "created_at"),
    )


class Model(Base):
    """Model information model."""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    type = Column(Enum(ModelType), nullable=False, index=True)
    version = Column(String(50))
    description = Column(Text)
    
    # File information
    file_path = Column(String(500))
    file_size = Column(Integer)  # bytes
    file_hash = Column(String(64))  # SHA-256
    download_url = Column(String(500))
    
    # Status
    is_available = Column(Boolean, default=False, index=True)
    is_loading = Column(Boolean, default=False)
    is_downloading = Column(Boolean, default=False)
    download_progress = Column(Float)  # 0.0 to 1.0
    
    # Usage tracking
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    memory_usage_mb = Column(Float)
    
    # Metadata
    file_metadata = Column("metadata", JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_models_type_available", "type", "is_available"),
        Index("idx_models_last_used", "last_used"),
    )


class FileUpload(Base):
    """File upload model."""
    __tablename__ = "file_uploads"
    
    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Storage
    storage_path = Column(String(500), nullable=False)
    storage_type = Column(String(20), default="s3")  # s3, local, gcs
    
    # Status
    is_uploaded = Column(Boolean, default=False)
    is_processed = Column(Boolean, default=False)
    
    # Lifecycle
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    accessed_at = Column(DateTime(timezone=True))
    
    # Metadata
    file_metadata = Column("metadata", JSON)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index("idx_files_user_created", "user_id", "created_at"),
        Index("idx_files_expires", "expires_at"),
    )


class ExecutionLog(Base):
    """Execution log model."""
    __tablename__ = "execution_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(String(36), ForeignKey("workflow_executions.id"), nullable=False)
    level = Column(String(20), nullable=False, index=True)  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    component = Column(String(100))  # workflow, model, storage, etc.
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    file_metadata = Column("metadata", JSON)
    
    # Indexes
    __table_args__ = (
        Index("idx_logs_execution_time", "execution_id", "timestamp"),
        Index("idx_logs_level_time", "level", "timestamp"),
    )


class SystemMetrics(Base):
    """System metrics model."""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # System metrics
    cpu_usage_percent = Column(Float)
    memory_usage_percent = Column(Float)
    disk_usage_percent = Column(Float)
    
    # GPU metrics
    gpu_usage_percent = Column(Float)
    gpu_memory_usage_percent = Column(Float)
    gpu_temperature = Column(Float)
    
    # Application metrics
    active_executions = Column(Integer, default=0)
    queue_size = Column(Integer, default=0)
    total_executions = Column(Integer, default=0)
    failed_executions = Column(Integer, default=0)
    average_execution_time = Column(Float)  # seconds
    
    # Model metrics
    loaded_models_count = Column(Integer, default=0)
    model_memory_usage_mb = Column(Float, default=0)
    
    # Indexes
    __table_args__ = (
        Index("idx_metrics_timestamp", "timestamp"),
    )


class WebhookLog(Base):
    """Webhook delivery log model."""
    __tablename__ = "webhook_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(String(36), ForeignKey("workflow_executions.id"), nullable=False)
    webhook_url = Column(String(500), nullable=False)
    
    # Request/Response
    request_payload = Column(JSON)
    response_status = Column(Integer)
    response_body = Column(Text)
    response_time_ms = Column(Float)
    
    # Delivery status
    is_successful = Column(Boolean, default=False)
    attempt_number = Column(Integer, default=1)
    error_message = Column(Text)
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    delivered_at = Column(DateTime(timezone=True))
    next_retry_at = Column(DateTime(timezone=True))
    
    # Indexes
    __table_args__ = (
        Index("idx_webhooks_execution", "execution_id"),
        Index("idx_webhooks_retry", "next_retry_at", "is_successful"),
    )