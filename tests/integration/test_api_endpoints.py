"""Integration tests for API endpoints."""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock

from src.api.main import app


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    def test_user_registration(self, client: TestClient):
        """Test user registration endpoint."""
        user_data = {
            "email": "newuser@example.com",
            "password": "SecurePass123!",
            "username": "newuser"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["email"] == user_data["email"]
        assert "password" not in data  # Password should not be returned
    
    def test_user_login(self, client: TestClient, test_user_data):
        """Test user login endpoint."""
        # First register a user
        client.post("/auth/register", json=test_user_data)
        
        # Then login
        login_data = {
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_token_refresh(self, client: TestClient, test_user_data):
        """Test token refresh endpoint."""
        # Register and login to get tokens
        client.post("/auth/register", json=test_user_data)
        login_response = client.post("/auth/login", json={
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        })
        
        tokens = login_response.json()
        refresh_token = tokens["refresh_token"]
        
        # Use refresh token to get new access token
        refresh_response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        assert refresh_response.status_code == 200
        new_tokens = refresh_response.json()
        assert "access_token" in new_tokens
        assert new_tokens["access_token"] != tokens["access_token"]
    
    def test_protected_endpoint_without_token(self, client: TestClient):
        """Test accessing protected endpoint without token."""
        response = client.get("/workflows/list")
        
        assert response.status_code == 401
        assert "unauthorized" in response.json()["error"].lower()
    
    def test_protected_endpoint_with_invalid_token(self, client: TestClient):
        """Test accessing protected endpoint with invalid token."""
        response = client.get(
            "/workflows/list",
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401


class TestWorkflowEndpoints:
    """Test workflow management endpoints."""
    
    def test_submit_workflow(self, client: TestClient, test_workflow):
        """Test workflow submission endpoint."""
        response = client.post(
            "/workflows/submit",
            json=test_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "execution_id" in data
        assert data["status"] == "pending"
        assert "estimated_duration" in data
    
    def test_get_workflow_status(self, client: TestClient, test_workflow):
        """Test getting workflow status."""
        # Submit workflow first
        submit_response = client.post(
            "/workflows/submit",
            json=test_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        execution_id = submit_response.json()["execution_id"]
        
        # Get status
        status_response = client.get(
            f"/workflows/{execution_id}/status",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["execution_id"] == execution_id
        assert "status" in status_data
        assert "created_at" in status_data
    
    def test_cancel_workflow(self, client: TestClient, test_workflow):
        """Test workflow cancellation."""
        # Submit workflow
        submit_response = client.post(
            "/workflows/submit", 
            json=test_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        execution_id = submit_response.json()["execution_id"]
        
        # Cancel workflow
        cancel_response = client.post(
            f"/workflows/{execution_id}/cancel",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert cancel_response.status_code == 200
        
        # Verify status is cancelled
        status_response = client.get(
            f"/workflows/{execution_id}/status",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert status_response.json()["status"] == "cancelled"
    
    def test_list_workflows(self, client: TestClient):
        """Test listing user workflows."""
        response = client.get(
            "/workflows/list",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "executions" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
    
    def test_workflow_validation(self, client: TestClient):
        """Test workflow validation."""
        # Test invalid workflow (empty nodes)
        invalid_workflow = {
            "workflow": {
                "nodes": {},
                "metadata": {}
            },
            "priority": "normal"
        }
        
        response = client.post(
            "/workflows/submit",
            json=invalid_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 422  # Validation error


class TestModelEndpoints:
    """Test model management endpoints."""
    
    def test_list_models(self, client: TestClient):
        """Test listing available models."""
        response = client.get(
            "/models/list",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total_memory_usage_mb" in data
        assert "available_memory_mb" in data
    
    def test_download_model(self, client: TestClient):
        """Test model download endpoint."""
        model_data = {
            "name": "test_model.safetensors",
            "type": "checkpoint",
            "download_url": "https://example.com/model.safetensors"
        }
        
        with patch('src.services.model.model_service.download_model') as mock_download:
            mock_download.return_value = True
            
            response = client.post(
                "/models/download",
                json=model_data,
                headers={"Authorization": "Bearer test-token"}
            )
        
        assert response.status_code == 202  # Accepted
        data = response.json()
        assert "message" in data
        assert "downloading" in data["message"].lower()
    
    def test_model_status(self, client: TestClient):
        """Test getting model status."""
        response = client.get(
            "/models/test_model.safetensors/status?type=checkpoint",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "type" in data
        assert "is_loaded" in data
        assert "is_downloading" in data
    
    def test_delete_model(self, client: TestClient):
        """Test model deletion."""
        with patch('src.services.model.model_service.delete_model') as mock_delete:
            mock_delete.return_value = True
            
            response = client.delete(
                "/models/test_model.safetensors?type=checkpoint",
                headers={"Authorization": "Bearer test-token"}
            )
        
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()


class TestFileEndpoints:
    """Test file management endpoints."""
    
    def test_upload_file(self, client: TestClient, sample_image):
        """Test file upload endpoint."""
        response = client.post(
            "/files/upload",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "file_id" in data
        assert "filename" in data
        assert "size" in data
        assert data["filename"] == "test.jpg"
    
    def test_download_file(self, client: TestClient, sample_image):
        """Test file download endpoint."""
        # First upload a file
        upload_response = client.post(
            "/files/upload",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
            headers={"Authorization": "Bearer test-token"}
        )
        
        file_id = upload_response.json()["file_id"]
        
        # Then download it
        download_response = client.get(
            f"/files/{file_id}/download",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert download_response.status_code == 200
        assert download_response.headers["content-type"] == "image/jpeg"
    
    def test_list_files(self, client: TestClient):
        """Test listing user files."""
        response = client.get(
            "/files/list",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert "total" in data
    
    def test_delete_file(self, client: TestClient, sample_image):
        """Test file deletion."""
        # Upload file first
        upload_response = client.post(
            "/files/upload",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
            headers={"Authorization": "Bearer test-token"}
        )
        
        file_id = upload_response.json()["file_id"]
        
        # Delete file
        delete_response = client.delete(
            f"/files/{file_id}",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert delete_response.status_code == 200
        
        # Verify file is deleted
        download_response = client.get(
            f"/files/{file_id}/download",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert download_response.status_code == 404


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, client: TestClient):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
    
    def test_detailed_health_check(self, client: TestClient):
        """Test detailed health check endpoint."""
        with patch('src.services.monitoring.monitoring_service.check_health') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "services": [
                    {"name": "redis", "status": "healthy", "response_time_ms": 2.3},
                    {"name": "database", "status": "healthy", "response_time_ms": 5.1},
                    {"name": "comfyui", "status": "healthy", "response_time_ms": 12.4}
                ],
                "system": {
                    "cpu_usage": 45.2,
                    "memory_usage": 60.1,
                    "disk_usage": 35.8
                }
            }
            
            response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "system" in data
        assert len(data["services"]) == 3
    
    def test_readiness_check(self, client: TestClient):
        """Test readiness check endpoint."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
    
    def test_liveness_check(self, client: TestClient):
        """Test liveness check endpoint."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert "alive" in data


class TestMetricsEndpoints:
    """Test metrics endpoints."""
    
    def test_system_metrics(self, client: TestClient):
        """Test system metrics endpoint."""
        with patch('src.services.monitoring.monitoring_service.collect_system_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "cpu_usage_percent": 45.2,
                "memory_usage_percent": 60.1,
                "disk_usage_percent": 35.8,
                "gpu_usage_percent": 25.5,
                "timestamp": "2024-08-31T12:00:00Z"
            }
            
            response = client.get(
                "/metrics/system",
                headers={"Authorization": "Bearer test-token"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "cpu_usage_percent" in data
        assert "memory_usage_percent" in data
        assert "disk_usage_percent" in data
        assert "timestamp" in data
    
    def test_execution_metrics(self, client: TestClient):
        """Test execution metrics endpoint."""
        response = client.get(
            "/metrics/executions",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_executions" in data
        assert "completed_executions" in data
        assert "average_duration_seconds" in data
    
    def test_prometheus_metrics(self, client: TestClient):
        """Test Prometheus-compatible metrics endpoint."""
        response = client.get("/metrics/prometheus")
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        
        # Check for common Prometheus metrics format
        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content


class TestErrorHandling:
    """Test API error handling."""
    
    def test_404_error(self, client: TestClient):
        """Test 404 error handling."""
        response = client.get("/nonexistent/endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "timestamp" in data
    
    def test_validation_error(self, client: TestClient):
        """Test validation error handling."""
        # Send invalid JSON to an endpoint that expects specific structure
        response = client.post(
            "/workflows/submit",
            json={"invalid": "data"},
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "validation" in data["error"].lower()
    
    def test_rate_limiting(self, client: TestClient):
        """Test rate limiting."""
        # Make multiple rapid requests
        responses = []
        for _ in range(100):  # Exceed rate limit
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Should eventually get rate limited
        assert any(status == 429 for status in responses)
    
    def test_request_timeout(self, client: TestClient):
        """Test request timeout handling."""
        with patch('asyncio.sleep', side_effect=lambda x: None if x < 60 else Exception("Timeout")):
            # This would normally timeout in a real scenario
            response = client.get("/health")
            # In test environment, this should still work
            assert response.status_code in [200, 408, 504]


class TestMiddleware:
    """Test API middleware functionality."""
    
    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are present."""
        response = client.options("/health")
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_request_logging(self, client: TestClient):
        """Test request logging middleware."""
        with patch('structlog.get_logger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            client.get("/health")
            
            # Should have logged the request (exact call depends on implementation)
            assert mock_log.info.called or mock_log.debug.called
    
    def test_metrics_collection(self, client: TestClient):
        """Test metrics collection middleware."""
        # Make a request
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check that request was tracked (this would need access to metrics store)
        # In a real test, you'd check that metrics were incremented
    
    def test_authentication_middleware(self, client: TestClient):
        """Test authentication middleware."""
        # Test with valid token
        response = client.get(
            "/workflows/list",
            headers={"Authorization": "Bearer valid-test-token"}
        )
        # Response depends on token validation implementation
        
        # Test with invalid token format
        response = client.get(
            "/workflows/list",
            headers={"Authorization": "InvalidFormat"}
        )
        assert response.status_code == 401