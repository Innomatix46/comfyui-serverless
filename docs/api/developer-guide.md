# Developer Guide

This comprehensive guide provides everything developers need to know to work with the ComfyUI Serverless API, from local development setup to production deployment.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Local Development](#local-development)
- [Configuration Management](#configuration-management)
- [Testing Strategies](#testing-strategies)
- [Performance Optimization](#performance-optimization)
- [Deployment Guide](#deployment-guide)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)

## Development Environment Setup

### Prerequisites

Before setting up the development environment, ensure you have the following installed:

- Python 3.9 or higher
- Node.js 16 or higher (for frontend development)
- Docker and Docker Compose
- Git
- PostgreSQL 13+
- Redis 6+

### Local Environment Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/your-org/comfyui-serverless.git
cd comfyui-serverless
```

#### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### 3. Environment Variables

Create a `.env` file in the root directory:

```bash
# .env
# API Configuration
API_TITLE="ComfyUI Serverless API"
API_VERSION="1.0.0"
API_DESCRIPTION="Serverless ComfyUI API for AI image generation"
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-super-secret-development-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
ALGORITHM=HS256

# Database
DATABASE_URL=postgresql://comfyui_user:comfyui_password@localhost:5432/comfyui_dev
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_QUEUE_DB=1
REDIS_CACHE_DB=2

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Storage (Development - Local)
STORAGE_TYPE=local
STORAGE_BASE_PATH=/tmp/comfyui_dev
MAX_FILE_SIZE=104857600  # 100MB

# ComfyUI Configuration
COMFYUI_PATH=/opt/ComfyUI
COMFYUI_MODELS_PATH=/opt/ComfyUI/models
COMFYUI_OUTPUT_PATH=/opt/ComfyUI/output
COMFYUI_TEMP_PATH=/opt/ComfyUI/temp
COMFYUI_API_URL=http://localhost:8188

# GPU Configuration
GPU_MEMORY_FRACTION=0.8
MAX_GPU_MEMORY_GB=24
ENABLE_MODEL_OFFLOAD=true
MODEL_CACHE_SIZE_GB=10

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100  # Higher for development
RATE_LIMIT_PER_HOUR=5000
RATE_LIMIT_PER_DAY=50000

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=DEBUG
LOG_FORMAT=json

# Development-specific
ENABLE_CORS=true
ENABLE_DOCS=true
ENABLE_REDOC=true
```

#### 4. Database Setup

```bash
# Start PostgreSQL (if not running)
brew services start postgresql  # macOS
sudo systemctl start postgresql  # Linux

# Create database and user
psql -c "CREATE USER comfyui_user WITH PASSWORD 'comfyui_password';"
psql -c "CREATE DATABASE comfyui_dev OWNER comfyui_user;"
psql -c "GRANT ALL PRIVILEGES ON DATABASE comfyui_dev TO comfyui_user;"

# Run database migrations
alembic upgrade head
```

#### 5. Redis Setup

```bash
# Start Redis
brew services start redis  # macOS
sudo systemctl start redis  # Linux

# Verify Redis is running
redis-cli ping
# Should return PONG
```

#### 6. Docker Development Environment

For a complete containerized development environment:

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - /tmp/comfyui_dev:/tmp/comfyui_dev
    environment:
      - DEBUG=true
      - DATABASE_URL=postgresql://comfyui_user:comfyui_password@db:5432/comfyui_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    command: ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: comfyui_dev
      POSTGRES_USER: comfyui_user
      POSTGRES_PASSWORD: comfyui_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  celery_worker:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
    environment:
      - DATABASE_URL=postgresql://comfyui_user:comfyui_password@db:5432/comfyui_dev
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
    depends_on:
      - db
      - redis
    command: ["celery", "-A", "services.celery_app", "worker", "--loglevel=info"]

  flower:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/1
    depends_on:
      - redis
    command: ["celery", "-A", "services.celery_app", "flower"]

volumes:
  postgres_data:
  redis_data:
```

Start the development environment:

```bash
docker-compose -f docker-compose.dev.yml up -d
```

## Local Development

### Running the Development Server

#### Option 1: Direct Python Execution

```bash
# Activate virtual environment
source venv/bin/activate

# Start the API server
python -m api.main

# Or using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start Celery worker (in another terminal)
celery -A services.celery_app worker --loglevel=info

# Start Flower for Celery monitoring (optional)
celery -A services.celery_app flower
```

#### Option 2: Using Docker Compose

```bash
docker-compose -f docker-compose.dev.yml up
```

### Development Workflow

#### 1. Code Organization

The project follows a clean architecture pattern:

```
src/
├── api/                 # API layer
│   ├── main.py         # FastAPI application
│   ├── middleware.py   # Custom middleware
│   └── routers/        # API route handlers
├── core/               # Core business logic
│   └── database.py     # Database connections
├── models/             # Data models
│   ├── database.py     # SQLAlchemy models
│   └── schemas.py      # Pydantic schemas
├── services/           # Business services
│   ├── workflow.py     # Workflow management
│   ├── auth.py         # Authentication
│   └── storage.py      # File storage
├── utils/              # Utility functions
└── config/             # Configuration
    └── settings.py     # Application settings
```

#### 2. Adding New Features

Follow this workflow when adding new features:

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-awesome-feature
   ```

2. **Write Tests First (TDD)**
   ```python
   # tests/unit/test_new_feature.py
   import pytest
   from services.new_feature import NewFeatureService

   def test_new_feature_creation():
       service = NewFeatureService()
       result = service.create_feature(name="test")
       assert result.name == "test"
   ```

3. **Implement the Feature**
   ```python
   # services/new_feature.py
   class NewFeatureService:
       def create_feature(self, name: str) -> Feature:
           return Feature(name=name)
   ```

4. **Add API Endpoint**
   ```python
   # api/routers/new_feature.py
   from fastapi import APIRouter, Depends
   from services.new_feature import NewFeatureService

   router = APIRouter()

   @router.post("/features/")
   async def create_feature(
       feature_data: FeatureCreate,
       service: NewFeatureService = Depends()
   ):
       return service.create_feature(feature_data.name)
   ```

5. **Run Tests**
   ```bash
   pytest tests/unit/test_new_feature.py -v
   pytest tests/integration/ -v
   ```

6. **Update Documentation**
   ```bash
   # Add OpenAPI documentation
   # Update README if needed
   # Add examples to docs/
   ```

### Code Quality Tools

The project uses several tools to maintain code quality:

#### 1. Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

#### 2. Testing Configuration

```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
```

#### 3. Code Formatting

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/

# Run all quality checks
pre-commit run --all-files
```

## Configuration Management

### Settings Architecture

The application uses Pydantic for configuration management with environment-based overrides:

```python
# config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    API_TITLE: str = "ComfyUI Serverless API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 10
    
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Feature Flags
    ENABLE_RATE_LIMITING: bool = True
    ENABLE_WEBHOOKS: bool = True
    ENABLE_MODEL_CACHING: bool = True
    
    # Environment-specific settings
    ENVIRONMENT: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() in ["development", "dev"]
    
    @property 
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in ["production", "prod"]
    
    @property
    def database_config(self) -> dict:
        """Database connection configuration."""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "echo": self.DEBUG
        }

# Global settings instance
settings = Settings()
```

### Environment-Specific Configurations

#### Development Configuration
```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://user:pass@localhost/comfyui_dev
RATE_LIMIT_PER_MINUTE=1000
ENABLE_CORS=true
```

#### Testing Configuration
```bash
# .env.testing
DATABASE_URL=postgresql://user:pass@localhost/comfyui_test
REDIS_URL=redis://localhost:6379/15
CELERY_TASK_ALWAYS_EAGER=true
DISABLE_AUTH=true  # For testing
```

#### Production Configuration
```bash
# .env.production
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=${DATABASE_URL}  # From environment
SECRET_KEY=${SECRET_KEY}
CORS_ORIGINS=https://yourdomain.com
TRUSTED_HOSTS=yourdomain.com,*.yourdomain.com
```

## Testing Strategies

### Test Organization

```
tests/
├── conftest.py           # Pytest configuration and fixtures
├── fixtures/             # Test data and factories
│   ├── factories.py      # Model factories
│   └── data_generators.py # Test data generators
├── unit/                 # Unit tests
│   ├── test_services/    # Service layer tests
│   ├── test_models/      # Model tests
│   └── test_utils/       # Utility tests
├── integration/          # Integration tests
│   ├── test_api/         # API endpoint tests
│   └── test_workflows/   # Workflow integration tests
├── e2e/                  # End-to-end tests
└── performance/          # Performance and load tests
```

### Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from core.database import get_db, Base
from config.settings import settings

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with database session override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers(client):
    """Create authentication headers for testing."""
    # Create test user and get token
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "testpassword",
        "username": "testuser"
    })
    
    login_response = client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "testpassword"
    })
    
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def mock_comfyui_client(mocker):
    """Mock ComfyUI client for testing."""
    mock_client = mocker.patch('services.comfyui.ComfyUIClient')
    mock_client.return_value.execute_workflow.return_value = "test-execution-id"
    mock_client.return_value.get_workflow_result.return_value = {
        "execution_id": "test-execution-id",
        "status": "completed",
        "outputs": {"images": [{"filename": "test.png", "url": "https://example.com/test.png"}]}
    }
    return mock_client
```

### Unit Tests Example

```python
# tests/unit/test_workflow_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from services.workflow import WorkflowService
from models.schemas import WorkflowExecutionRequest, WorkflowDefinition

class TestWorkflowService:
    
    @pytest.fixture
    def workflow_service(self, db_session):
        return WorkflowService(db=db_session)
    
    @pytest.fixture
    def sample_workflow(self):
        return WorkflowDefinition(
            nodes={
                "1": {
                    "id": "1",
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": [
                        {"name": "ckpt_name", "type": "STRING", "value": "test.ckpt", "required": True}
                    ]
                }
            }
        )
    
    def test_workflow_validation(self, workflow_service, sample_workflow):
        """Test workflow validation logic."""
        is_valid, errors = workflow_service.validate_workflow(sample_workflow)
        assert is_valid
        assert not errors
    
    def test_invalid_workflow_validation(self, workflow_service):
        """Test validation of invalid workflow."""
        invalid_workflow = WorkflowDefinition(nodes={})
        is_valid, errors = workflow_service.validate_workflow(invalid_workflow)
        assert not is_valid
        assert "Workflow must contain at least one node" in errors
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, workflow_service, sample_workflow, mock_comfyui_client):
        """Test workflow execution."""
        request = WorkflowExecutionRequest(workflow=sample_workflow)
        
        execution_id = await workflow_service.execute_workflow(request, user_id=1)
        
        assert execution_id == "test-execution-id"
        mock_comfyui_client.return_value.execute_workflow.assert_called_once()
    
    def test_workflow_estimation(self, workflow_service, sample_workflow):
        """Test workflow duration estimation."""
        estimated_duration = workflow_service.estimate_duration(sample_workflow)
        assert isinstance(estimated_duration, int)
        assert estimated_duration > 0
```

### Integration Tests Example

```python
# tests/integration/test_workflow_api.py
import pytest
import json

class TestWorkflowAPI:
    
    def test_workflow_execution_endpoint(self, client, auth_headers, mock_comfyui_client):
        """Test the workflow execution API endpoint."""
        workflow_data = {
            "workflow": {
                "nodes": {
                    "1": {
                        "id": "1",
                        "class_type": "CheckpointLoaderSimple",
                        "inputs": [
                            {"name": "ckpt_name", "type": "STRING", "value": "test.ckpt", "required": True}
                        ]
                    }
                }
            },
            "priority": "normal"
        }
        
        response = client.post(
            "/workflows/execute",
            json=workflow_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "execution_id" in data
        assert data["status"] == "pending"
    
    def test_workflow_execution_without_auth(self, client):
        """Test workflow execution without authentication."""
        workflow_data = {"workflow": {"nodes": {}}}
        
        response = client.post("/workflows/execute", json=workflow_data)
        
        assert response.status_code == 401
    
    def test_invalid_workflow_request(self, client, auth_headers):
        """Test invalid workflow request."""
        invalid_data = {"workflow": {"nodes": {}}}  # Empty nodes
        
        response = client.post(
            "/workflows/execute",
            json=invalid_data,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        assert "validation" in response.json()["error"].lower()
    
    def test_workflow_result_endpoint(self, client, auth_headers, mock_comfyui_client):
        """Test getting workflow results."""
        # First execute a workflow
        workflow_data = {
            "workflow": {
                "nodes": {
                    "1": {
                        "id": "1",
                        "class_type": "CheckpointLoaderSimple",
                        "inputs": [{"name": "ckpt_name", "type": "STRING", "value": "test.ckpt", "required": True}]
                    }
                }
            }
        }
        
        execution_response = client.post(
            "/workflows/execute",
            json=workflow_data,
            headers=auth_headers
        )
        execution_id = execution_response.json()["execution_id"]
        
        # Then get the result
        result_response = client.get(
            f"/workflows/{execution_id}",
            headers=auth_headers
        )
        
        assert result_response.status_code == 200
        result_data = result_response.json()
        assert result_data["execution_id"] == execution_id
```

### Performance Tests Example

```python
# tests/performance/test_load.py
import pytest
import time
import concurrent.futures
from fastapi.testclient import TestClient

class TestAPIPerformance:
    
    def test_concurrent_workflow_executions(self, client, auth_headers):
        """Test API performance under concurrent load."""
        
        def submit_workflow():
            workflow_data = {
                "workflow": {
                    "nodes": {
                        "1": {
                            "id": "1",
                            "class_type": "CheckpointLoaderSimple",
                            "inputs": [{"name": "ckpt_name", "type": "STRING", "value": "test.ckpt", "required": True}]
                        }
                    }
                }
            }
            
            start_time = time.time()
            response = client.post("/workflows/execute", json=workflow_data, headers=auth_headers)
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "response": response.json() if response.status_code == 200 else None
            }
        
        # Submit 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(submit_workflow) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_requests = [r for r in results if r["status_code"] == 200]
        average_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
        
        # Assertions
        assert len(successful_requests) >= 8  # At least 80% success rate
        assert average_response_time < 2.0  # Average response time under 2 seconds
        
        print(f"Success rate: {len(successful_requests)}/10")
        print(f"Average response time: {average_response_time:.2f}s")
```

## Performance Optimization

### Database Optimization

#### Connection Pooling

```python
# core/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import os

def create_database_engine():
    """Create optimized database engine."""
    return create_engine(
        os.getenv("DATABASE_URL"),
        poolclass=QueuePool,
        pool_size=20,  # Number of connections to maintain
        max_overflow=30,  # Additional connections when needed
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=3600,  # Recycle connections every hour
        echo=False  # Set to True for SQL debugging
    )

engine = create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

#### Query Optimization

```python
# services/workflow.py
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy import and_, or_, func

class WorkflowService:
    
    def get_user_workflows_optimized(self, user_id: int, limit: int = 20):
        """Get user workflows with optimized queries."""
        return self.db.query(WorkflowExecution)\
            .options(selectinload(WorkflowExecution.files))\
            .filter(WorkflowExecution.user_id == user_id)\
            .filter(WorkflowExecution.status.in_(["completed", "failed"]))\
            .order_by(WorkflowExecution.created_at.desc())\
            .limit(limit)\
            .all()
    
    def get_workflow_statistics(self, user_id: int):
        """Get aggregated workflow statistics."""
        stats = self.db.query(
            func.count(WorkflowExecution.id).label('total'),
            func.count(WorkflowExecution.id).filter(
                WorkflowExecution.status == 'completed'
            ).label('completed'),
            func.avg(WorkflowExecution.duration_seconds).label('avg_duration')
        ).filter(
            WorkflowExecution.user_id == user_id
        ).first()
        
        return {
            'total_workflows': stats.total,
            'completed_workflows': stats.completed,
            'average_duration': float(stats.avg_duration or 0)
        }
```

### Caching Strategies

#### Redis Caching

```python
# services/cache.py
import redis
import json
import pickle
from typing import Any, Optional
from functools import wraps

class CacheService:
    
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache."""
        try:
            serialized = pickle.dumps(value)
            return self.redis_client.setex(
                key, 
                ttl or self.default_ttl, 
                serialized
            )
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return bool(self.redis_client.delete(key))
    
    def cache_result(self, ttl: int = None):
        """Decorator to cache function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # Try to get from cache first
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            return wrapper
        return decorator

# Usage example
cache_service = CacheService(os.getenv("REDIS_URL"))

@cache_service.cache_result(ttl=1800)  # Cache for 30 minutes
def get_model_metadata(model_name: str):
    """Expensive operation to get model metadata."""
    # This would normally query external APIs or file systems
    return {
        "name": model_name,
        "size": "4.2GB",
        "type": "checkpoint",
        "hash": "sha256:abc123..."
    }
```

#### Application-Level Caching

```python
# utils/cache_decorators.py
from functools import wraps, lru_cache
import time
from typing import Dict, Any

class TTLCache:
    """Time-based cache for function results."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check if cached and not expired
            if key in self.cache:
                cached_data = self.cache[key]
                if time.time() - cached_data['timestamp'] < self.ttl_seconds:
                    return cached_data['result']
                else:
                    del self.cache[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.cache[key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            return result
        return wrapper

# Usage examples
@lru_cache(maxsize=100)
def get_workflow_template(template_name: str):
    """Cache workflow templates in memory."""
    # Load template from file system
    with open(f"templates/{template_name}.json") as f:
        return json.load(f)

@TTLCache(ttl_seconds=600)  # Cache for 10 minutes
def get_system_metrics():
    """Cache expensive system metrics."""
    import psutil
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    }
```

### Async Processing Optimization

```python
# services/async_workflow.py
import asyncio
import aiohttp
from typing import List, Dict, Any
import time

class AsyncWorkflowProcessor:
    """Optimized async workflow processing."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_workflow_batch(self, workflows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple workflows concurrently."""
        tasks = [
            self._process_single_workflow(workflow)
            for workflow in workflows
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'workflow_id': workflows[i].get('id', i),
                    'status': 'error',
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single workflow with concurrency control."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Simulate workflow processing
                execution_id = await self._submit_to_comfyui(workflow)
                result = await self._wait_for_completion(execution_id)
                
                return {
                    'workflow_id': workflow.get('id'),
                    'execution_id': execution_id,
                    'status': 'completed',
                    'duration': time.time() - start_time,
                    'result': result
                }
                
            except Exception as e:
                return {
                    'workflow_id': workflow.get('id'),
                    'status': 'error',
                    'duration': time.time() - start_time,
                    'error': str(e)
                }
    
    async def _submit_to_comfyui(self, workflow: Dict[str, Any]) -> str:
        """Submit workflow to ComfyUI API."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://comfyui-api:8188/api/execute',
                json=workflow,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                data = await response.json()
                return data['execution_id']
    
    async def _wait_for_completion(self, execution_id: str, poll_interval: int = 5) -> Dict[str, Any]:
        """Wait for workflow completion with async polling."""
        while True:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://comfyui-api:8188/api/status/{execution_id}') as response:
                    status_data = await response.json()
                    
                    if status_data['status'] in ['completed', 'failed']:
                        return status_data
                    
                    await asyncio.sleep(poll_interval)

# Usage example
async def main():
    processor = AsyncWorkflowProcessor(max_concurrent=5)
    
    workflows = [
        {'id': f'workflow_{i}', 'prompt': f'Image {i}', 'seed': i}
        for i in range(20)
    ]
    
    results = await processor.process_workflow_batch(workflows)
    
    successful = len([r for r in results if r['status'] == 'completed'])
    print(f"Processed {successful}/{len(workflows)} workflows successfully")

# Run async processing
if __name__ == "__main__":
    asyncio.run(main())
```

## Deployment Guide

### Docker Production Setup

#### Multi-Stage Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set PATH
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://comfyui_user:${DB_PASSWORD}@db:5432/comfyui_prod
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - DEBUG=false
    depends_on:
      - db
      - redis
    restart: unless-stopped
    
  celery_worker:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://comfyui_user:${DB_PASSWORD}@db:5432/comfyui_prod
      - CELERY_BROKER_URL=redis://redis:6379/1
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db
      - redis
    restart: unless-stopped
    command: ["celery", "-A", "src.services.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
    
  celery_beat:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://comfyui_user:${DB_PASSWORD}@db:5432/comfyui_prod
      - CELERY_BROKER_URL=redis://redis:6379/1
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db
      - redis
    restart: unless-stopped
    command: ["celery", "-A", "src.services.celery_app", "beat", "--loglevel=info"]
    
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: comfyui_prod
      POSTGRES_USER: comfyui_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_types text/plain application/json application/javascript text/css;
    
    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # API endpoints
        location / {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }
        
        # Health check endpoint (no rate limiting)
        location /health {
            proxy_pass http://api;
            proxy_set_header Host $host;
        }
        
        # File uploads
        location /files/upload {
            client_max_body_size 100M;
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_request_buffering off;
        }
    }
}
```

### Kubernetes Deployment

#### Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: comfyui-serverless
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: comfyui-config
  namespace: comfyui-serverless
data:
  DATABASE_HOST: "postgresql-service"
  REDIS_HOST: "redis-service"
  API_TITLE: "ComfyUI Serverless API"
  API_VERSION: "1.0.0"
  ENVIRONMENT: "production"
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: comfyui-secrets
  namespace: comfyui-serverless
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret>
  DB_PASSWORD: <base64-encoded-password>
  JWT_SECRET: <base64-encoded-jwt-secret>
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: comfyui-api
  namespace: comfyui-serverless
spec:
  replicas: 3
  selector:
    matchLabels:
      app: comfyui-api
  template:
    metadata:
      labels:
        app: comfyui-api
    spec:
      containers:
      - name: api
        image: comfyui-serverless:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://comfyui_user:$(DB_PASSWORD)@$(DATABASE_HOST):5432/comfyui_prod"
        - name: REDIS_URL
          value: "redis://$(REDIS_HOST):6379/0"
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: comfyui-secrets
              key: SECRET_KEY
        envFrom:
        - configMapRef:
            name: comfyui-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: comfyui-api-service
  namespace: comfyui-serverless
spec:
  selector:
    app: comfyui-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: comfyui-api-hpa
  namespace: comfyui-serverless
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: comfyui-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Cloud Deployment Examples

#### AWS ECS Deployment

```json
{
  "family": "comfyui-serverless-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "comfyui-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/comfyui-serverless:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:comfyui/database:url::"
        },
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:comfyui/api:secret_key::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/comfyui-serverless",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health/ || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### Google Cloud Run Deployment

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: comfyui-serverless-api
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "1"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 100
      containers:
      - image: gcr.io/project-id/comfyui-serverless:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: comfyui-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: comfyui-secrets
              key: secret-key
        resources:
          limits:
            cpu: 2
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health/
            port: 8000
          initialDelaySeconds: 30
```

### Deployment Automation

#### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test
        REDIS_URL: redis://localhost:6379/0
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
  
  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/deployment.yaml
          k8s/service.yaml
        images: |
          comfyui-serverless=ghcr.io/${{ github.repository }}:${{ github.sha }}
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    
    - name: Verify deployment
      run: |
        kubectl rollout status deployment/comfyui-api -n comfyui-serverless
        kubectl get pods -n comfyui-serverless
```

This comprehensive developer guide provides everything needed to set up, develop, test, and deploy the ComfyUI Serverless API. From local development to production deployment, it covers all aspects of the development lifecycle.