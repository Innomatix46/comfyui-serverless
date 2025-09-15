"""Test utilities and helper functions."""
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import Mock, AsyncMock, MagicMock
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


class DatabaseTestHelper:
    """Helper class for database testing operations."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_test_user(self, email: str = None, **kwargs) -> Dict[str, Any]:
        """Create a test user in the database."""
        from src.models.database import User
        
        if email is None:
            email = f"test_{uuid.uuid4().hex[:8]}@example.com"
        
        user_data = {
            "email": email,
            "username": email.split("@")[0],
            "hashed_password": "$2b$12$test_hash",
            "is_active": True,
            **kwargs
        }
        
        user = User(**user_data)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "is_active": user.is_active
        }
    
    def create_test_execution(self, user_id: int = None, **kwargs) -> Dict[str, Any]:
        """Create a test workflow execution in the database."""
        from src.models.database import WorkflowExecution, WorkflowStatus, Priority
        
        if user_id is None:
            user = self.create_test_user()
            user_id = user["id"]
        
        execution_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "workflow_definition": {"nodes": {"1": {"class_type": "TestNode"}}},
            "status": WorkflowStatus.PENDING,
            "priority": Priority.NORMAL,
            "created_at": datetime.utcnow(),
            "queue_position": 1,
            "estimated_duration": 300,
            **kwargs
        }
        
        execution = WorkflowExecution(**execution_data)
        self.db.add(execution)
        self.db.commit()
        self.db.refresh(execution)
        
        return {
            "id": execution.id,
            "user_id": execution.user_id,
            "status": execution.status.value,
            "created_at": execution.created_at.isoformat()
        }
    
    def cleanup_test_data(self):
        """Clean up test data from database."""
        from src.models.database import WorkflowExecution, User
        
        # Delete test executions
        self.db.query(WorkflowExecution).filter(
            WorkflowExecution.id.like("test-%")
        ).delete(synchronize_session=False)
        
        # Delete test users
        self.db.query(User).filter(
            User.email.like("test_%@example.com")
        ).delete(synchronize_session=False)
        
        self.db.commit()


class APITestHelper:
    """Helper class for API testing operations."""
    
    def __init__(self, client: TestClient):
        self.client = client
        self.auth_token = None
    
    def authenticate_user(self, email: str = "test@example.com", password: str = "testpass123") -> str:
        """Authenticate a user and return auth token."""
        # Register user first
        register_response = self.client.post("/auth/register", json={
            "email": email,
            "password": password,
            "username": email.split("@")[0]
        })
        
        if register_response.status_code != 201:
            # User might already exist, try to login
            pass
        
        # Login
        login_response = self.client.post("/auth/login", json={
            "email": email,
            "password": password
        })
        
        assert login_response.status_code == 200
        token_data = login_response.json()
        self.auth_token = token_data["access_token"]
        return self.auth_token
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if not self.auth_token:
            self.authenticate_user()
        
        return {"Authorization": f"Bearer {self.auth_token}"}
    
    def submit_test_workflow(self, workflow_data: Dict = None) -> Dict[str, Any]:
        """Submit a test workflow."""
        if workflow_data is None:
            workflow_data = {
                "workflow": {
                    "nodes": {
                        "1": {
                            "class_type": "CheckpointLoaderSimple",
                            "inputs": [{"name": "ckpt_name", "value": "test.safetensors", "type": "string", "required": True}],
                            "outputs": ["MODEL", "CLIP", "VAE"]
                        }
                    }
                },
                "priority": "normal"
            }
        
        response = self.client.post(
            "/workflows/submit",
            json=workflow_data,
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == 201
        return response.json()
    
    def wait_for_workflow_status(
        self, 
        execution_id: str, 
        target_status: str, 
        timeout: int = 30,
        poll_interval: float = 0.5
    ) -> Dict[str, Any]:
        """Wait for workflow to reach target status."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.client.get(
                f"/workflows/{execution_id}/status",
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                status_data = response.json()
                if status_data["status"] == target_status:
                    return status_data
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Workflow {execution_id} did not reach status {target_status} within {timeout} seconds")


class MockServiceHelper:
    """Helper class for mocking external services."""
    
    @staticmethod
    def mock_redis_client() -> Mock:
        """Create a mock Redis client."""
        mock = Mock()
        mock.ping.return_value = True
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = 1
        mock.exists.return_value = 0
        mock.incr.return_value = 1
        mock.expire.return_value = True
        mock.hset.return_value = 1
        mock.hget.return_value = None
        mock.hdel.return_value = 1
        mock.lpush.return_value = 1
        mock.rpop.return_value = None
        mock.lrange.return_value = []
        mock.llen.return_value = 0
        return mock
    
    @staticmethod
    def mock_celery_app() -> Mock:
        """Create a mock Celery app."""
        mock = Mock()
        mock_task = Mock()
        mock_task.id = "test-task-id"
        mock_task.status = "PENDING"
        mock_task.result = None
        
        mock.send_task.return_value = mock_task
        mock.control.inspect.return_value.active.return_value = {}
        mock.control.inspect.return_value.reserved.return_value = {}
        mock.control.revoke.return_value = None
        
        return mock
    
    @staticmethod
    def mock_comfyui_client() -> AsyncMock:
        """Create a mock ComfyUI client."""
        mock = AsyncMock()
        mock.health_check.return_value = True
        mock.execute_workflow.return_value = {
            "status": "completed",
            "outputs": {"images": [{"filename": "output.png", "type": "output"}]},
            "logs": ["Workflow started", "Workflow completed"]
        }
        mock.cancel_execution.return_value = True
        mock.get_queue_status.return_value = {
            "queue_remaining": 0,
            "queue_pending": []
        }
        mock.get_system_stats.return_value = {
            "system": {"ram": {"total": 32000, "used": 8000}},
            "devices": [{"name": "cuda:0", "type": "gpu"}]
        }
        return mock
    
    @staticmethod
    def mock_s3_client() -> Mock:
        """Create a mock S3 client."""
        mock = Mock()
        mock.put_object.return_value = {"ETag": '"test-etag"'}
        mock.get_object.return_value = {
            "Body": Mock(read=lambda: b"test file content")
        }
        mock.delete_object.return_value = {}
        mock.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "test-file.txt", "Size": 100, "LastModified": datetime.utcnow()}
            ]
        }
        mock.generate_presigned_url.return_value = "https://example.com/presigned-url"
        mock.head_bucket.return_value = {}
        return mock
    
    @staticmethod
    def mock_webhook_service() -> AsyncMock:
        """Create a mock webhook service."""
        mock = AsyncMock()
        mock.send_webhook.return_value = True
        mock.send_completion_webhook.return_value = True
        mock.send_progress_webhook.return_value = True
        mock.validate_webhook_url.return_value = True
        mock.validate_payload.return_value = True
        return mock


class FileTestHelper:
    """Helper class for file testing operations."""
    
    @staticmethod
    def create_temp_image(width: int = 64, height: int = 64, format: str = "JPEG") -> tempfile.NamedTemporaryFile:
        """Create a temporary test image."""
        from PIL import Image
        import io
        
        # Create test image
        img = Image.new('RGB', (width, height), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format=format)
        img_buffer.seek(0)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{format.lower()}", delete=False)
        temp_file.write(img_buffer.getvalue())
        temp_file.seek(0)
        
        return temp_file
    
    @staticmethod
    def create_temp_json(data: Dict[str, Any]) -> tempfile.NamedTemporaryFile:
        """Create a temporary JSON file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix=".json", delete=False)
        json.dump(data, temp_file)
        temp_file.seek(0)
        return temp_file
    
    @staticmethod
    def create_temp_text(content: str) -> tempfile.NamedTemporaryFile:
        """Create a temporary text file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix=".txt", delete=False)
        temp_file.write(content)
        temp_file.seek(0)
        return temp_file
    
    @staticmethod
    def cleanup_temp_files(files: List[str]):
        """Clean up temporary files."""
        import os
        for file_path in files:
            try:
                os.unlink(file_path)
            except (OSError, FileNotFoundError):
                pass


class PerformanceTestHelper:
    """Helper class for performance testing."""
    
    @staticmethod
    def measure_execution_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Measure function execution time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    async def measure_async_execution_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Measure async function execution time."""
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def run_load_test(
        func: Callable,
        num_iterations: int,
        num_threads: int = 1
    ) -> Dict[str, Any]:
        """Run a load test on a function."""
        import threading
        import statistics
        
        results = []
        errors = []
        
        def worker():
            for _ in range(num_iterations // num_threads):
                try:
                    start_time = time.time()
                    func()
                    end_time = time.time()
                    results.append(end_time - start_time)
                except Exception as e:
                    errors.append(str(e))
        
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        if results:
            return {
                "total_iterations": len(results),
                "errors": len(errors),
                "avg_time": statistics.mean(results),
                "min_time": min(results),
                "max_time": max(results),
                "p95_time": statistics.quantiles(results, n=20)[18] if len(results) >= 20 else max(results),
                "success_rate": len(results) / (len(results) + len(errors))
            }
        else:
            return {
                "total_iterations": 0,
                "errors": len(errors),
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "p95_time": 0,
                "success_rate": 0
            }
    
    @staticmethod
    def assert_performance_requirements(
        results: Dict[str, Any],
        max_avg_time: float = None,
        max_p95_time: float = None,
        min_success_rate: float = None
    ):
        """Assert performance requirements are met."""
        if max_avg_time is not None:
            assert results["avg_time"] <= max_avg_time, f"Average time {results['avg_time']:.3f}s exceeds limit {max_avg_time}s"
        
        if max_p95_time is not None:
            assert results["p95_time"] <= max_p95_time, f"P95 time {results['p95_time']:.3f}s exceeds limit {max_p95_time}s"
        
        if min_success_rate is not None:
            assert results["success_rate"] >= min_success_rate, f"Success rate {results['success_rate']:.2%} below minimum {min_success_rate:.2%}"


class AsyncTestHelper:
    """Helper class for async testing."""
    
    @staticmethod
    async def run_concurrently(tasks: List[Callable], max_concurrent: int = 10) -> List[Any]:
        """Run async tasks concurrently with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await task()
        
        return await asyncio.gather(*[run_with_semaphore(task) for task in tasks])
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 30.0,
        poll_interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if condition():
                return True
            await asyncio.sleep(poll_interval)
        
        return False
    
    @staticmethod
    def create_async_mock(return_value: Any = None) -> AsyncMock:
        """Create an async mock with optional return value."""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        return mock


class ValidationTestHelper:
    """Helper class for validation testing."""
    
    @staticmethod
    def assert_valid_uuid(value: str):
        """Assert value is a valid UUID."""
        try:
            uuid.UUID(value)
        except ValueError:
            pytest.fail(f"'{value}' is not a valid UUID")
    
    @staticmethod
    def assert_valid_datetime(value: str):
        """Assert value is a valid ISO datetime."""
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"'{value}' is not a valid ISO datetime")
    
    @staticmethod
    def assert_valid_email(value: str):
        """Assert value is a valid email."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            pytest.fail(f"'{value}' is not a valid email address")
    
    @staticmethod
    def assert_response_structure(response_data: Dict, expected_keys: List[str], optional_keys: List[str] = None):
        """Assert response has expected structure."""
        optional_keys = optional_keys or []
        
        # Check required keys
        for key in expected_keys:
            assert key in response_data, f"Required key '{key}' missing from response"
        
        # Check for unexpected keys
        all_expected_keys = set(expected_keys + optional_keys)
        unexpected_keys = set(response_data.keys()) - all_expected_keys
        
        if unexpected_keys:
            pytest.fail(f"Unexpected keys in response: {unexpected_keys}")
    
    @staticmethod
    def assert_error_response(response_data: Dict, expected_error_type: str = None):
        """Assert response is a valid error response."""
        required_keys = ["error", "message", "timestamp"]
        ValidationTestHelper.assert_response_structure(response_data, required_keys)
        
        if expected_error_type:
            assert response_data["error"] == expected_error_type, f"Expected error type '{expected_error_type}', got '{response_data['error']}'"


# Global test helpers instance
test_helpers = {
    "database": DatabaseTestHelper,
    "api": APITestHelper, 
    "mock": MockServiceHelper,
    "file": FileTestHelper,
    "performance": PerformanceTestHelper,
    "async": AsyncTestHelper,
    "validation": ValidationTestHelper
}