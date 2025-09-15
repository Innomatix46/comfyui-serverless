"""Pydantic models for API request/response schemas."""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum
import json


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ModelType(str, Enum):
    """ComfyUI model types."""
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    EMBEDDING = "embedding"
    VAE = "vae"
    CONTROLNET = "controlnet"
    UPSCALER = "upscaler"


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common fields."""
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# User schemas
class UserBase(BaseSchema):
    """Base user schema."""
    email: str = Field(..., description="User email address")
    username: Optional[str] = Field(None, description="Username")
    is_active: bool = Field(True, description="User active status")


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8, description="User password")


class UserUpdate(BaseSchema):
    """User update schema."""
    email: Optional[str] = None
    username: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """User response schema."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime]


# Authentication schemas
class Token(BaseSchema):
    """Authentication token schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseSchema):
    """Login request payload."""
    email: str
    password: str


class TokenData(BaseSchema):
    """Token data schema."""
    user_id: Optional[int] = None
    username: Optional[str] = None


# Workflow schemas
class WorkflowNodeInput(BaseSchema):
    """Workflow node input schema."""
    name: str = Field(..., description="Input name")
    type: str = Field(..., description="Input type")
    value: Any = Field(..., description="Input value")
    required: bool = Field(True, description="Required input")


class WorkflowNode(BaseSchema):
    """Workflow node schema."""
    id: str = Field(..., description="Node ID")
    class_type: str = Field(..., description="Node class type")
    inputs: List[WorkflowNodeInput] = Field(default=[], description="Node inputs")
    outputs: List[str] = Field(default=[], description="Output names")


class WorkflowDefinition(BaseSchema):
    """Workflow definition schema."""
    nodes: Dict[str, WorkflowNode] = Field(..., description="Workflow nodes")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Workflow metadata")
    
    @validator('nodes')
    def validate_nodes(cls, v):
        """Validate workflow nodes."""
        if not v:
            raise ValueError("Workflow must contain at least one node")
        return v


class WorkflowExecutionRequest(BaseSchema):
    """Workflow execution request schema."""
    workflow: WorkflowDefinition = Field(..., description="Workflow definition")
    priority: Priority = Field(Priority.NORMAL, description="Execution priority")
    webhook_url: Optional[str] = Field(None, description="Webhook callback URL")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Execution metadata")
    timeout_minutes: Optional[int] = Field(30, ge=1, le=120, description="Execution timeout")


class WorkflowExecutionResponse(BaseSchema):
    """Workflow execution response schema."""
    execution_id: str = Field(..., description="Unique execution ID")
    status: WorkflowStatus = Field(..., description="Execution status")
    created_at: datetime = Field(..., description="Creation timestamp")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")
    queue_position: Optional[int] = Field(None, description="Position in queue")


class WorkflowResult(BaseSchema):
    """Workflow execution result schema."""
    execution_id: str
    status: WorkflowStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# Model management schemas
class ModelInfo(BaseSchema):
    """Model information schema."""
    name: str = Field(..., description="Model name")
    type: ModelType = Field(..., description="Model type")
    version: Optional[str] = Field(None, description="Model version")
    description: Optional[str] = Field(None, description="Model description")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    download_url: Optional[str] = Field(None, description="Download URL")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Model metadata")


class ModelStatus(BaseSchema):
    """Model status schema."""
    name: str
    type: ModelType
    is_loaded: bool
    is_downloading: bool
    download_progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_used: Optional[datetime] = None
    memory_usage_mb: Optional[float] = None


class ModelListResponse(BaseSchema):
    """Model list response schema."""
    models: List[ModelStatus] = Field(..., description="List of models")
    total_memory_usage_mb: float = Field(..., description="Total memory usage")
    available_memory_mb: float = Field(..., description="Available memory")


# File upload schemas
class FileUploadResponse(BaseSchema):
    """File upload response schema."""
    file_id: str = Field(..., description="Unique file ID")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    upload_url: Optional[str] = Field(None, description="Upload URL for large files")


class FileInfo(BaseSchema):
    """File information schema."""
    file_id: str
    filename: str
    size: int
    content_type: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    download_url: Optional[str] = None


# Health check schemas
class HealthStatus(BaseSchema):
    """Health check status schema."""
    status: str = Field(..., description="Overall status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")


class ServiceHealthStatus(BaseSchema):
    """Individual service health status."""
    name: str
    status: str
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DetailedHealthStatus(HealthStatus):
    """Detailed health check status."""
    services: List[ServiceHealthStatus] = Field(..., description="Service statuses")
    system: Dict[str, Any] = Field(..., description="System information")


# Error schemas
class ErrorDetail(BaseSchema):
    """Error detail schema."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field causing error")


class ErrorResponse(BaseSchema):
    """API error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# Metrics schemas
class SystemMetrics(BaseSchema):
    """System metrics schema."""
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_usage_percent: Optional[float] = None
    disk_usage_percent: float
    active_executions: int
    queue_size: int
    total_executions: int
    average_execution_time_seconds: float


class ExecutionMetrics(BaseSchema):
    """Execution metrics schema."""
    total_executions: int
    completed_executions: int
    failed_executions: int
    average_duration_seconds: float
    executions_per_minute: float
    queue_wait_time_seconds: float
