"""Integration tests for the API."""
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os

from src.api.main import app
from src.core.database import Base, get_db
from src.config.settings import settings


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Set up test database."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")


@pytest.fixture
def test_user():
    """Create test user."""
    user_data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "username": "testuser"
    }
    
    response = client.post("/auth/register", json=user_data)
    assert response.status_code == 200
    
    return user_data


@pytest.fixture
def auth_token(test_user):
    """Get authentication token for test user."""
    login_data = {
        "email": test_user["email"],
        "password": test_user["password"]
    }
    
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 200
    
    token_data = response.json()
    return token_data["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    """Get authorization headers."""
    return {"Authorization": f"Bearer {auth_token}"}


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == settings.API_TITLE
        assert data["version"] == settings.API_VERSION
        assert data["status"] == "healthy"
    
    def test_health_check(self):
        """Test basic health check."""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
    
    def test_liveness_check(self):
        """Test liveness probe."""
        response = client.get("/health/liveness")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
    
    def test_readiness_check(self):
        """Test readiness probe."""
        response = client.get("/health/readiness")
        # May return 503 if Redis/DB not available in test environment
        assert response.status_code in [200, 503]


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    def test_register_user(self):
        """Test user registration."""
        user_data = {
            "email": "newuser@example.com",
            "password": "newpassword123",
            "username": "newuser"
        }
        
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert data["is_active"] is True
    
    def test_register_duplicate_email(self, test_user):
        """Test registration with duplicate email."""
        response = client.post("/auth/register", json=test_user)
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    
    def test_login_success(self, test_user):
        """Test successful login."""
        login_data = {
            "email": test_user["email"],
            "password": test_user["password"]
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        login_data = {
            "email": "invalid@example.com",
            "password": "wrongpassword"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401
    
    def test_get_current_user(self, auth_headers, test_user):
        """Test getting current user info."""
        response = client.get("/auth/me", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["email"] == test_user["email"]
        assert data["username"] == test_user["username"]
    
    def test_unauthorized_access(self):
        """Test access without authentication."""
        response = client.get("/auth/me")
        assert response.status_code == 401


class TestWorkflowEndpoints:
    """Test workflow management endpoints."""
    
    def test_list_workflows_empty(self, auth_headers):
        """Test listing workflows when none exist."""
        response = client.get("/workflows/", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_execute_workflow_invalid(self, auth_headers):
        """Test executing invalid workflow."""
        workflow_data = {
            "workflow": {
                "nodes": {}  # Empty workflow
            },
            "priority": "normal"
        }
        
        response = client.post("/workflows/execute", headers=auth_headers, json=workflow_data)
        # Should fail validation
        assert response.status_code in [400, 422, 500]  # Depending on validation implementation
    
    def test_get_nonexistent_workflow(self, auth_headers):
        """Test getting non-existent workflow."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/workflows/{fake_id}", headers=auth_headers)
        assert response.status_code == 404
    
    def test_workflow_without_auth(self):
        """Test workflow endpoints without authentication."""
        response = client.get("/workflows/")
        assert response.status_code == 401


class TestModelsEndpoints:
    """Test model management endpoints."""
    
    def test_list_models(self, auth_headers):
        """Test listing models."""
        response = client.get("/models/", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "total_memory_usage_mb" in data
        assert "available_memory_mb" in data
    
    def test_get_nonexistent_model(self, auth_headers):
        """Test getting non-existent model."""
        response = client.get("/models/nonexistent_model", headers=auth_headers)
        assert response.status_code == 404
    
    def test_models_without_auth(self):
        """Test models endpoints without authentication."""
        response = client.get("/models/")
        assert response.status_code == 401


class TestFilesEndpoints:
    """Test file management endpoints."""
    
    def test_upload_file(self, auth_headers):
        """Test file upload."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.txt') as tmp_file:
            test_content = b"This is a test file"
            tmp_file.write(test_content)
            tmp_file.flush()
            
            try:
                with open(tmp_file.name, 'rb') as upload_file:
                    files = {"file": ("test.txt", upload_file, "text/plain")}
                    response = client.post("/files/upload", headers=auth_headers, files=files)
                
                assert response.status_code == 200
                
                data = response.json()
                assert "file_id" in data
                assert data["filename"] == "test.txt"
                assert data["size"] == len(test_content)
                
                return data["file_id"]
            finally:
                os.unlink(tmp_file.name)
    
    def test_upload_oversized_file(self, auth_headers):
        """Test upload of oversized file."""
        # Create a large temporary file
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp_file:
            # Write more than 100MB
            large_content = b"x" * (101 * 1024 * 1024)
            tmp_file.write(large_content)
            tmp_file.flush()
            
            try:
                with open(tmp_file.name, 'rb') as upload_file:
                    files = {"file": ("large.txt", upload_file, "text/plain")}
                    response = client.post("/files/upload", headers=auth_headers, files=files)
                
                assert response.status_code == 413  # Request Entity Too Large
            finally:
                os.unlink(tmp_file.name)
    
    def test_list_files_empty(self, auth_headers):
        """Test listing files when none exist."""
        response = client.get("/files/", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_nonexistent_file(self, auth_headers):
        """Test getting non-existent file."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/files/{fake_id}", headers=auth_headers)
        assert response.status_code == 404
    
    def test_files_without_auth(self):
        """Test file endpoints without authentication."""
        response = client.get("/files/")
        assert response.status_code == 401


class TestMetricsEndpoints:
    """Test metrics endpoints."""
    
    def test_prometheus_metrics(self):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        
        # Should return text/plain content
        assert "text/plain" in response.headers.get("content-type", "")
    
    def test_system_metrics(self, auth_headers):
        """Test system metrics endpoint."""
        response = client.get("/metrics/system", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "cpu_usage_percent" in data
        assert "memory_usage_percent" in data
        assert "active_executions" in data
    
    def test_execution_metrics(self, auth_headers):
        """Test execution metrics endpoint."""
        response = client.get("/metrics/executions", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_executions" in data
        assert "completed_executions" in data
        assert "failed_executions" in data


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_health_endpoint(self):
        """Test that health endpoints are not rate limited."""
        # Make many requests to health endpoint
        for _ in range(10):
            response = client.get("/health/")
            assert response.status_code == 200
    
    @pytest.mark.skip(reason="Rate limiting requires Redis in test environment")
    def test_rate_limit_api_endpoints(self, auth_headers):
        """Test rate limiting on API endpoints."""
        # This test would require Redis to be available
        # and proper rate limiting configuration
        pass


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_not_found(self):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self):
        """Test 405 error handling."""
        response = client.post("/health/")  # GET only endpoint
        assert response.status_code == 405
    
    def test_422_validation_error(self):
        """Test validation error handling."""
        # Send invalid JSON to registration endpoint
        response = client.post("/auth/register", json={"invalid": "data"})
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_async_operations():
    """Test async operations."""
    # This is a placeholder for testing async functionality
    # In a real implementation, you would test async database operations,
    # background tasks, etc.
    
    await asyncio.sleep(0.01)  # Minimal async operation
    assert True