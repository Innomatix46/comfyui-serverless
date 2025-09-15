# ComfyUI Serverless Technical Specifications

## Overview

This document provides comprehensive technical specifications for the ComfyUI serverless platform, including infrastructure requirements, implementation guidelines, deployment configurations, and operational procedures.

## 1. System Requirements and Specifications

### 1.1 Infrastructure Requirements

**Minimum System Requirements:**
```yaml
infrastructure_requirements:
  compute:
    api_services:
      cpu: 4 vCPUs (x86_64 or ARM64)
      memory: 8GB RAM
      storage: 50GB SSD
      network: 1Gbps
      
    gpu_containers:
      gpu: NVIDIA RTX3060 (12GB VRAM) minimum
      cpu: 8 vCPUs
      memory: 32GB RAM
      storage: 200GB NVMe SSD
      network: 10Gbps
      
    database:
      cpu: 4 vCPUs
      memory: 16GB RAM
      storage: 500GB SSD (IOPS: 3000+)
      network: 1Gbps
      
  storage:
    model_storage:
      capacity: 10TB minimum
      performance: 1000 IOPS, 100MB/s throughput
      durability: 99.999999999% (11 9's)
      
    result_storage:
      capacity: 5TB minimum
      performance: 500 IOPS, 50MB/s throughput
      retention: 30 days default
      
    cache_storage:
      capacity: 2TB NVMe SSD per GPU node
      performance: 50000 IOPS, 2GB/s throughput
      latency: <1ms

  network:
    bandwidth: 10Gbps inter-service
    latency: <5ms intra-region, <50ms cross-region
    redundancy: Multi-AZ deployment required
```

**Recommended Production Requirements:**
```yaml
production_requirements:
  compute:
    api_services:
      instances: 6 (multi-AZ)
      cpu: 8 vCPUs per instance
      memory: 16GB RAM per instance
      auto_scaling: 3-20 instances
      
    gpu_containers:
      gpu_types: [A100-80GB, H100-80GB, RTX4090-24GB]
      instances: 50 (distributed across regions)
      auto_scaling: 10-200 instances
      spot_instance_ratio: 70%
      
    load_balancers:
      type: Application Load Balancer
      instances: 2 (cross-AZ)
      ssl_termination: true
      waf_enabled: true
      
  storage:
    model_storage:
      capacity: 100TB
      replication: 3x across regions
      cdn: Global distribution
      
    database:
      engine: PostgreSQL 14
      configuration: Multi-AZ, read replicas
      backup: Point-in-time recovery
      
  monitoring:
    metrics_retention: 15 days
    log_retention: 90 days
    alerting: 24/7 coverage
    sla_monitoring: enabled
```

### 1.2 Software Dependencies

**Core Software Stack:**
```yaml
software_dependencies:
  runtime:
    operating_system: Ubuntu 22.04 LTS
    container_runtime: Docker 24.0+
    orchestration: Kubernetes 1.28+ or ECS
    
  application_stack:
    api_framework: FastAPI 0.104+
    async_runtime: Python 3.10+ with asyncio
    queue_system: Redis 7.0+ with Redis Queue
    database: PostgreSQL 14+
    cache: Redis 7.0+ (separate from queue)
    
  ai_ml_stack:
    comfyui: Latest stable release
    pytorch: 2.1+ with CUDA 11.8+
    transformers: 4.35+
    diffusers: 0.21+
    custom_nodes: Vetted extensions only
    
  gpu_stack:
    cuda_driver: 535.0+
    nvidia_container_toolkit: 1.14+
    tensorrt: 8.6+ (optional optimization)
    
  monitoring_stack:
    metrics: Prometheus 2.45+
    logging: ELK Stack 8.0+
    tracing: Jaeger 1.50+
    alerting: AlertManager 0.26+
    
  security_stack:
    authentication: JWT with RSA-256
    encryption: AES-256 at rest, TLS 1.3 in transit
    secrets_management: AWS Secrets Manager / HashiCorp Vault
    vulnerability_scanning: Trivy, Snyk
```

## 2. API Implementation Specifications

### 2.1 FastAPI Application Structure

**Application Architecture:**
```python
# app/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer
import asyncio
import uvloop

# Set high-performance event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

app = FastAPI(
    title="ComfyUI Serverless API",
    description="High-performance serverless API for ComfyUI workflow execution",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

# Dependency injection
from app.dependencies import get_database, get_auth_service, get_queue_manager

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Include routers
from app.routers import workflows, models, queue, admin
app.include_router(workflows.router, prefix="/v1/workflows", tags=["workflows"])
app.include_router(models.router, prefix="/v1/models", tags=["models"])
app.include_router(queue.router, prefix="/v1/queue", tags=["queue"])
app.include_router(admin.router, prefix="/v1/admin", tags=["admin"])

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        workers=4,
        loop="uvloop",
        http="httptools",
        access_log=True,
        server_header=False
    )
```

### 2.2 Database Schema Design

**PostgreSQL Schema:**
```sql
-- Users and Authentication
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    tier VARCHAR(20) NOT NULL DEFAULT 'basic',
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_tier ON users(tier);

-- API Keys
CREATE TABLE api_keys (
    key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_prefix VARCHAR(20) NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]',
    rate_limits JSONB NOT NULL DEFAULT '{}',
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- Workflow Executions
CREATE TABLE workflow_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    workflow_data JSONB NOT NULL,
    input_overrides JSONB DEFAULT '{}',
    output_config JSONB DEFAULT '{}',
    execution_config JSONB DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    priority INTEGER NOT NULL DEFAULT 5,
    queue_position INTEGER,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    error_details JSONB,
    resource_allocation JSONB,
    performance_metrics JSONB,
    total_cost DECIMAL(10,4) DEFAULT 0
);

CREATE INDEX idx_executions_user_id ON workflow_executions(user_id);
CREATE INDEX idx_executions_status ON workflow_executions(status);
CREATE INDEX idx_executions_submitted_at ON workflow_executions(submitted_at);
CREATE INDEX idx_executions_priority ON workflow_executions(priority);

-- Execution Results
CREATE TABLE execution_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES workflow_executions(execution_id),
    output_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    content_type VARCHAR(100),
    storage_provider VARCHAR(50) NOT NULL,
    storage_region VARCHAR(50),
    public_url TEXT,
    signed_url TEXT,
    url_expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_results_execution_id ON execution_results(execution_id);

-- Models
CREATE TABLE models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_hash VARCHAR(128) NOT NULL,
    metadata JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    availability_regions TEXT[] DEFAULT '{}',
    cache_priority INTEGER DEFAULT 5,
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_models_category ON models(category);
CREATE INDEX idx_models_hash ON models(file_hash);
CREATE INDEX idx_models_active ON models(active);

-- Audit Logs
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    event_type VARCHAR(50) NOT NULL,
    event_details JSONB NOT NULL DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);

-- Performance optimization
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## 3. Container Specifications

### 3.1 Production Container Images

**Base Runtime Container:**
```dockerfile
# Dockerfile.base
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    wget \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r comfyui && useradd -r -g comfyui -s /bin/bash comfyui

# Set up Python environment
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create application directory
WORKDIR /app
RUN chown -R comfyui:comfyui /app

# Copy ComfyUI
COPY --chown=comfyui:comfyui ComfyUI/ /app/ComfyUI/
COPY --chown=comfyui:comfyui src/ /app/src/

# Set up entrypoint
COPY --chown=comfyui:comfyui entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

USER comfyui
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "src.server"]
```

**GPU Execution Container:**
```dockerfile
# Dockerfile.gpu
FROM comfyui-base:latest

# Install additional GPU-optimized packages
USER root
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install TensorRT for optimization
RUN pip install --no-cache-dir tensorrt

# Copy GPU-specific configuration
COPY --chown=comfyui:comfyui gpu_config/ /app/gpu_config/

# Set GPU-specific environment
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA="1"

USER comfyui

# Warm-up GPU on startup
RUN python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Devices: {torch.cuda.device_count()}')"
```

### 3.2 Kubernetes Deployment Specifications

**API Service Deployment:**
```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: comfyui-api
  namespace: comfyui
  labels:
    app: comfyui-api
    component: api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 50%
      maxUnavailable: 25%
  selector:
    matchLabels:
      app: comfyui-api
  template:
    metadata:
      labels:
        app: comfyui-api
        component: api
    spec:
      containers:
      - name: api
        image: comfyui-api:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi" 
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
      securityContext:
        fsGroup: 1000
      imagePullPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: comfyui-api-service
  namespace: comfyui
spec:
  selector:
    app: comfyui-api
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  type: ClusterIP
```

**GPU Container Deployment:**
```yaml
# k8s/gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: comfyui-gpu
  namespace: comfyui
  labels:
    app: comfyui-gpu
    component: gpu-worker
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: comfyui-gpu
  template:
    metadata:
      labels:
        app: comfyui-gpu
        component: gpu-worker
    spec:
      nodeSelector:
        accelerator: nvidia-gpu
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: gpu-worker
        image: comfyui-gpu:latest
        ports:
        - containerPort: 8080
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "24Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: temp-storage
          mountPath: /tmp
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: temp-storage
        emptyDir:
          sizeLimit: 50Gi
```

## 4. Monitoring and Observability

### 4.1 Prometheus Metrics Configuration

**Custom Metrics Definition:**
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time

# API Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

# Workflow Metrics
workflow_executions_total = Counter(
    'workflow_executions_total',
    'Total workflow executions',
    ['status', 'priority', 'user_tier']
)

workflow_execution_duration_seconds = Histogram(
    'workflow_execution_duration_seconds',
    'Workflow execution duration',
    ['complexity_level'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
)

workflow_queue_depth = Gauge(
    'workflow_queue_depth',
    'Current workflow queue depth',
    ['priority_level']
)

# GPU Metrics
gpu_utilization_percent = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id', 'node_id']
)

gpu_memory_usage_bytes = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id', 'node_id', 'memory_type']
)

gpu_temperature_celsius = Gauge(
    'gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['gpu_id', 'node_id']
)

# Cache Metrics
cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Cache hit rate',
    ['cache_level', 'cache_type']
)

cache_size_bytes = Gauge(
    'cache_size_bytes',
    'Cache size in bytes',
    ['cache_level', 'cache_type']
)

model_load_duration_seconds = Histogram(
    'model_load_duration_seconds',
    'Model loading duration',
    ['model_type', 'cache_level'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120]
)

# Cost Metrics
execution_cost_dollars = Histogram(
    'execution_cost_dollars',
    'Execution cost in dollars',
    ['resource_type', 'user_tier'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

# Resource Metrics
container_resource_usage = Gauge(
    'container_resource_usage',
    'Container resource usage',
    ['resource_type', 'container_id']
)

# Custom middleware for automatic metrics collection
class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            method = scope["method"]
            path = scope["path"]
            
            start_time = time.time()
            
            # Wrap send to capture response status
            async def wrapped_send(message):
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    duration = time.time() - start_time
                    
                    # Record metrics
                    api_requests_total.labels(
                        method=method,
                        endpoint=path,
                        status_code=status_code
                    ).inc()
                    
                    api_request_duration_seconds.labels(
                        method=method,
                        endpoint=path
                    ).observe(duration)
                
                await send(message)
            
            await self.app(scope, receive, wrapped_send)
        else:
            await self.app(scope, receive, send)
```

### 4.2 Logging Configuration

**Structured Logging Setup:**
```python
# logging_config.py
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'execution_id'):
            log_entry['execution_id'] = record.execution_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'container_id'):
            log_entry['container_id'] = record.container_id
        if hasattr(record, 'gpu_id'):
            log_entry['gpu_id'] = record.gpu_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logging():
    """Configure structured logging for the application"""
    
    # Root logger configuration
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(message)s'
    )
    
    # Set formatter for all handlers
    for handler in logging.root.handlers:
        handler.setFormatter(StructuredFormatter())
    
    # Configure specific loggers
    loggers = {
        'uvicorn.access': logging.WARNING,
        'uvicorn.error': logging.INFO,
        'sqlalchemy.engine': logging.WARNING,
        'redis': logging.WARNING,
        'comfyui.api': logging.INFO,
        'comfyui.workflow': logging.INFO,
        'comfyui.gpu': logging.INFO,
        'comfyui.storage': logging.INFO,
        'comfyui.security': logging.WARNING,
    }
    
    for logger_name, level in loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

# Performance logging decorator
import functools
import time

def log_performance(logger_name: str = None):
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"Function {func.__name__} completed successfully",
                    extra={
                        'function': func.__name__,
                        'duration_seconds': duration,
                        'status': 'success'
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"Function {func.__name__} failed",
                    extra={
                        'function': func.__name__,
                        'duration_seconds': duration,
                        'status': 'error',
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    exc_info=True
                )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"Function {func.__name__} completed successfully",
                    extra={
                        'function': func.__name__,
                        'duration_seconds': duration,
                        'status': 'success'
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"Function {func.__name__} failed",
                    extra={
                        'function': func.__name__,
                        'duration_seconds': duration,
                        'status': 'error',
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    exc_info=True
                )
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

## 5. Security Implementation

### 5.1 Authentication Implementation

**JWT Authentication Service:**
```python
# auth/jwt_handler.py
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets

class JWTHandler:
    def __init__(self, secret_key: str, algorithm: str = "RS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=30)
    
    async def create_tokens(self, user_data: Dict[str, Any]) -> Dict[str, str]:
        """Create access and refresh tokens"""
        
        # Access token payload
        access_payload = {
            "sub": str(user_data["user_id"]),
            "email": user_data["email"],
            "tier": user_data["tier"],
            "permissions": user_data.get("permissions", []),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.access_token_expire,
            "type": "access"
        }
        
        # Refresh token payload
        refresh_payload = {
            "sub": str(user_data["user_id"]),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.refresh_token_expire,
            "type": "refresh",
            "jti": secrets.token_hex(16)  # Unique token ID for blacklisting
        }
        
        # Generate tokens
        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int(self.access_token_expire.total_seconds())
        }
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Create new access token from refresh token"""
        
        try:
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid refresh token"
                )
            
            # Get user data
            user_data = await self.get_user_data(payload["sub"])
            
            # Create new access token
            access_payload = {
                "sub": payload["sub"],
                "email": user_data["email"],
                "tier": user_data["tier"],
                "permissions": user_data.get("permissions", []),
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + self.access_token_expire,
                "type": "access"
            }
            
            access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": int(self.access_token_expire.total_seconds())
            }
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Refresh token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token"
            )
```

### 5.2 API Key Management

**API Key Service:**
```python
# auth/api_key_manager.py
import secrets
import base64
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

class APIKeyManager:
    def __init__(self, database_client):
        self.db = database_client
        self.key_length = 32
        self.hash_algorithm = 'sha256'
    
    async def generate_api_key(self, user_id: str, permissions: List[str], expires_at: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate new API key"""
        
        # Generate cryptographically secure key
        key_bytes = secrets.token_bytes(self.key_length)
        key_b64 = base64.urlsafe_b64encode(key_bytes).decode('ascii').rstrip('=')
        
        # Determine prefix based on permissions
        if any('admin:' in perm for perm in permissions):
            prefix = 'comfyui_admin'
        elif any('pro_' in perm for perm in permissions):
            prefix = 'comfyui_pro'
        else:
            prefix = 'comfyui_basic'
        
        # Create formatted key
        api_key = f"{prefix}_{key_b64}"
        
        # Hash for storage (never store plaintext)
        key_hash = hashlib.pbkdf2_hmac(
            self.hash_algorithm,
            api_key.encode('utf-8'),
            user_id.encode('utf-8'),
            100000  # iterations
        ).hex()
        
        # Store in database
        key_record = {
            'user_id': user_id,
            'key_hash': key_hash,
            'key_prefix': prefix,
            'permissions': permissions,
            'expires_at': expires_at,
            'created_at': datetime.utcnow(),
            'active': True
        }
        
        key_id = await self.db.create_api_key(key_record)
        
        # Return key only once (never stored or logged)
        return {
            'key_id': key_id,
            'api_key': api_key,  # Only returned here
            'permissions': permissions,
            'expires_at': expires_at,
            'created_at': key_record['created_at']
        }
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return key data"""
        
        # Extract prefix and validate format
        if '_' not in api_key:
            return None
        
        prefix = api_key.split('_')[0] + '_' + api_key.split('_')[1]
        
        if not prefix.startswith('comfyui_'):
            return None
        
        # Get all active keys with this prefix
        active_keys = await self.db.get_active_keys_by_prefix(prefix)
        
        # Hash provided key with each user_id to find match
        for key_record in active_keys:
            test_hash = hashlib.pbkdf2_hmac(
                self.hash_algorithm,
                api_key.encode('utf-8'),
                key_record['user_id'].encode('utf-8'),
                100000
            ).hex()
            
            if hmac.compare_digest(test_hash, key_record['key_hash']):
                # Check expiration
                if key_record['expires_at'] and key_record['expires_at'] < datetime.utcnow():
                    return None
                
                # Update usage statistics
                await self.db.update_key_usage(key_record['key_id'])
                
                return {
                    'key_id': key_record['key_id'],
                    'user_id': key_record['user_id'],
                    'permissions': key_record['permissions'],
                    'tier': key_record.get('tier', 'basic')
                }
        
        return None
    
    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke API key"""
        
        result = await self.db.update_api_key(
            key_id,
            {'active': False, 'revoked_at': datetime.utcnow()},
            user_id=user_id
        )
        
        return result.modified_count > 0
```

## 6. Production Deployment Configurations

### 6.1 Terraform Infrastructure

**AWS Infrastructure as Code:**
```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "comfyui-serverless"
      ManagedBy   = "terraform"
    }
  }
}

# VPC Configuration
module "vpc" {
  source = "./modules/vpc"
  
  cidr_block           = var.vpc_cidr
  availability_zones   = var.availability_zones
  environment         = var.environment
  enable_nat_gateway  = true
  enable_vpn_gateway  = false
}

# ECS Cluster for API services
module "ecs_cluster" {
  source = "./modules/ecs"
  
  cluster_name        = "comfyui-${var.environment}"
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  public_subnet_ids  = module.vpc.public_subnet_ids
  
  api_service_config = {
    cpu    = 1024
    memory = 2048
    replicas = var.api_service_replicas
    image = "${var.ecr_repository_url}/comfyui-api:${var.image_tag}"
  }
  
  environment = var.environment
}

# RDS Database
module "database" {
  source = "./modules/rds"
  
  identifier             = "comfyui-${var.environment}"
  engine_version        = "14.9"
  instance_class        = var.db_instance_class
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  
  database_name = var.database_name
  username      = var.database_username
  
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  vpc_security_group_ids = [module.security_groups.database_sg_id]
  
  backup_retention_period = var.db_backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  deletion_protection = var.environment == "production" ? true : false
  skip_final_snapshot = var.environment == "production" ? false : true
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  environment = var.environment
}

# ElastiCache Redis
module "redis" {
  source = "./modules/elasticache"
  
  cluster_id           = "comfyui-${var.environment}"
  node_type           = var.redis_node_type
  num_cache_nodes     = var.redis_num_nodes
  parameter_group_name = "default.redis7"
  port                = 6379
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  security_group_ids = [module.security_groups.redis_sg_id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  environment = var.environment
}

# S3 Buckets
module "storage" {
  source = "./modules/s3"
  
  environment = var.environment
  
  buckets = {
    models = {
      name = "comfyui-models-${var.environment}"
      versioning = true
      lifecycle_rules = [
        {
          id = "model_lifecycle"
          status = "Enabled"
          transitions = [
            {
              days          = 30
              storage_class = "STANDARD_IA"
            },
            {
              days          = 90
              storage_class = "GLACIER"
            }
          ]
        }
      ]
    }
    
    results = {
      name = "comfyui-results-${var.environment}"
      versioning = false
      lifecycle_rules = [
        {
          id = "results_lifecycle"
          status = "Enabled"
          expiration = {
            days = 30
          }
        }
      ]
    }
  }
}

# Application Load Balancer
module "alb" {
  source = "./modules/alb"
  
  name               = "comfyui-${var.environment}"
  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnet_ids
  security_group_ids = [module.security_groups.alb_sg_id]
  
  certificate_arn = var.ssl_certificate_arn
  
  target_groups = {
    api = {
      name        = "comfyui-api-${var.environment}"
      port        = 8080
      protocol    = "HTTP"
      target_type = "ip"
      
      health_check = {
        enabled             = true
        healthy_threshold   = 2
        unhealthy_threshold = 2
        timeout             = 5
        interval            = 30
        path                = "/health"
        matcher             = "200"
      }
    }
  }
  
  environment = var.environment
}

# Auto Scaling
module "autoscaling" {
  source = "./modules/autoscaling"
  
  cluster_name = module.ecs_cluster.cluster_name
  service_name = module.ecs_cluster.api_service_name
  
  min_capacity = var.autoscaling_min_capacity
  max_capacity = var.autoscaling_max_capacity
  
  scale_up_policy = {
    metric_type                = "CPUUtilization"
    target_value              = 70
    scale_out_cooldown        = 300
    scale_in_cooldown         = 300
    disable_scale_in          = false
  }
  
  environment = var.environment
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = can(regex("^(dev|staging|production)$", var.environment))
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = module.database.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.redis.endpoint
  sensitive   = true
}

output "load_balancer_dns_name" {
  description = "Load balancer DNS name"
  value       = module.alb.dns_name
}
```

This comprehensive technical specification provides detailed implementation guidelines, infrastructure requirements, and production deployment configurations for the ComfyUI serverless platform.