"""Test configuration and fixtures."""
import pytest
import asyncio
import tempfile
import os
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.main import app
from src.core.database import Base, get_db
from src.config.settings import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db():
    """Create test database."""
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    database_url = f"sqlite:///{db_path}"
    
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield TestingSessionLocal
    
    # Cleanup
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def client(test_db) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_user_data():
    """Test user data."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "username": "testuser"
    }


@pytest.fixture
def test_workflow():
    """Test workflow definition."""
    return {
        "workflow": {
            "nodes": {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": [
                        {
                            "name": "ckpt_name",
                            "value": "model.safetensors",
                            "type": "string",
                            "required": True
                        }
                    ],
                    "outputs": ["MODEL", "CLIP", "VAE"]
                },
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": [
                        {
                            "name": "text",
                            "value": "a beautiful landscape",
                            "type": "string",
                            "required": True
                        },
                        {
                            "name": "clip",
                            "value": ["1", 1],
                            "type": "connection",
                            "required": True
                        }
                    ],
                    "outputs": ["CONDITIONING"]
                },
                "3": {
                    "class_type": "SaveImage",
                    "inputs": [
                        {
                            "name": "images",
                            "value": ["4", 0],
                            "type": "connection",
                            "required": True
                        }
                    ],
                    "outputs": []
                }
            },
            "metadata": {
                "title": "Test Workflow",
                "description": "A simple test workflow"
            }
        },
        "priority": "normal"
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    from unittest.mock import Mock
    
    mock_client = Mock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = 0
    mock_client.incr.return_value = 1
    mock_client.expire.return_value = True
    
    return mock_client


@pytest.fixture
def mock_celery():
    """Mock Celery app."""
    from unittest.mock import Mock
    
    mock_app = Mock()
    mock_task = Mock()
    mock_task.id = "test-task-id"
    mock_app.send_task.return_value = mock_task
    
    return mock_app


@pytest.fixture
def temp_file():
    """Create temporary file for testing."""
    fd, path = tempfile.mkstemp()
    
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(b"test file content")
        
        yield path
    finally:
        os.unlink(path)


@pytest.fixture
def mock_s3():
    """Mock S3 client."""
    from unittest.mock import Mock
    
    mock_client = Mock()
    mock_client.put_object.return_value = {"ETag": '"abc123"'}
    mock_client.get_object.return_value = {"Body": Mock()}
    mock_client.delete_object.return_value = {}
    mock_client.generate_presigned_url.return_value = "https://example.com/presigned-url"
    
    return mock_client


@pytest.fixture(autouse=True)
def mock_gpu():
    """Mock GPU utilities."""
    from unittest.mock import patch
    
    with patch('src.utils.gpu.get_gpu_memory_info') as mock_memory, \
         patch('src.utils.gpu.get_gpu_utilization') as mock_util:
        
        mock_memory.return_value = {
            'total_mb': 24000,
            'used_mb': 8000,
            'free_mb': 16000,
            'utilization_percent': 33.3
        }
        
        mock_util.return_value = {
            'utilization': 45,
            'temperature': 65,
            'memory_utilization': 33
        }
        
        yield


@pytest.fixture
def sample_image():
    """Create sample image for testing."""
    from PIL import Image
    import io
    
    # Create a small test image
    img = Image.new('RGB', (64, 64), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes


class MockAsyncIterator:
    """Mock async iterator for testing."""
    
    def __init__(self, items):
        self.items = items
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def mock_comfyui_response():
    """Mock ComfyUI API response."""
    return {
        "prompt_id": "test-prompt-id",
        "number": 1,
        "node_errors": {}
    }


@pytest.fixture
def mock_workflow_result():
    """Mock workflow execution result."""
    return {
        "outputs": {
            "images": [
                {
                    "filename": "test_output.png",
                    "subfolder": "",
                    "type": "output"
                }
            ]
        },
        "logs": [
            {
                "timestamp": "2024-08-31T00:00:00Z",
                "level": "INFO",
                "message": "Workflow started",
                "component": "workflow"
            },
            {
                "timestamp": "2024-08-31T00:01:00Z", 
                "level": "INFO",
                "message": "Workflow completed",
                "component": "workflow"
            }
        ],
        "execution_time": 60.0,
        "status": "completed"
    }