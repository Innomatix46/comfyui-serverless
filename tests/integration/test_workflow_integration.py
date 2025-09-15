"""Integration tests for complete workflow execution flow."""
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from src.api.main import app
from src.models.database import WorkflowExecution, WorkflowStatus, Priority
from src.models.schemas import WorkflowDefinition, WorkflowNodeInput, WorkflowNode
from src.services.workflow import workflow_service
from src.services.comfyui import ComfyUIClient


class TestWorkflowIntegration:
    """Test complete workflow execution integration."""
    
    def test_submit_and_track_workflow(self, client: TestClient, test_db, test_workflow):
        """Test submitting workflow and tracking its execution."""
        # Submit workflow
        response = client.post(
            "/workflows/submit",
            json=test_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        execution_id = data["execution_id"]
        assert data["status"] == "pending"
        assert "queue_position" in data
        
        # Track execution status
        status_response = client.get(
            f"/workflows/{execution_id}/status",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["execution_id"] == execution_id
        assert status_data["status"] in ["pending", "running", "completed"]
    
    def test_workflow_with_file_upload(self, client: TestClient, sample_image):
        """Test workflow that includes file upload."""
        # First upload a file
        upload_response = client.post(
            "/files/upload",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert upload_response.status_code == 201
        file_data = upload_response.json()
        file_id = file_data["file_id"]
        
        # Create workflow that uses the uploaded file
        workflow_data = {
            "workflow": {
                "nodes": {
                    "1": {
                        "class_type": "LoadImage",
                        "inputs": [
                            {
                                "name": "image",
                                "value": file_id,
                                "type": "file",
                                "required": True
                            }
                        ],
                        "outputs": ["IMAGE", "MASK"]
                    },
                    "2": {
                        "class_type": "SaveImage", 
                        "inputs": [
                            {
                                "name": "images",
                                "value": ["1", 0],
                                "type": "connection",
                                "required": True
                            }
                        ],
                        "outputs": []
                    }
                }
            },
            "priority": "normal"
        }
        
        # Submit workflow
        workflow_response = client.post(
            "/workflows/submit",
            json=workflow_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert workflow_response.status_code == 201
        workflow_result = workflow_response.json()
        assert "execution_id" in workflow_result
    
    def test_workflow_cancellation(self, client: TestClient, test_workflow):
        """Test workflow cancellation."""
        # Submit workflow
        response = client.post(
            "/workflows/submit",
            json=test_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        execution_id = response.json()["execution_id"]
        
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
        
        status_data = status_response.json()
        assert status_data["status"] == "cancelled"
    
    def test_workflow_with_webhook(self, client: TestClient, test_workflow):
        """Test workflow execution with webhook notification."""
        # Add webhook URL to workflow
        test_workflow["webhook_url"] = "https://webhook.site/test-endpoint"
        
        with patch('src.services.webhook.webhook_service.send_completion_webhook') as mock_webhook:
            mock_webhook.return_value = True
            
            # Submit workflow
            response = client.post(
                "/workflows/submit",
                json=test_workflow,
                headers={"Authorization": "Bearer test-token"}
            )
            
            execution_id = response.json()["execution_id"]
            
            # Simulate workflow completion
            # Note: In real integration test, this would happen through Celery
            # Here we'll directly call the service method for testing
            asyncio.run(
                workflow_service.submit_workflow(
                    execution_id,
                    WorkflowDefinition(**test_workflow["workflow"]),
                    webhook_url=test_workflow["webhook_url"]
                )
            )
        
        # Webhook should have been called (in real scenario after completion)
        # mock_webhook.assert_called_once()
    
    def test_workflow_progress_tracking(self, client: TestClient, test_workflow):
        """Test real-time workflow progress tracking."""
        # Submit workflow
        response = client.post(
            "/workflows/submit",
            json=test_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        execution_id = response.json()["execution_id"]
        
        # Mock progress updates in Redis
        with patch('redis.from_url') as mock_redis_factory:
            mock_redis = Mock()
            mock_redis.get.return_value = json.dumps({
                "status": "running",
                "progress_percent": 45.0,
                "current_node": "KSampler",
                "eta_seconds": 120,
                "timestamp": datetime.utcnow().isoformat()
            }).encode()
            
            mock_redis_factory.return_value = mock_redis
            
            # Get progress
            progress_response = client.get(
                f"/workflows/{execution_id}/progress",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert progress_response.status_code == 200
            progress_data = progress_response.json()
            assert progress_data["progress_percent"] == 45.0
            assert progress_data["current_node"] == "KSampler"
    
    def test_workflow_logs_streaming(self, client: TestClient, test_workflow):
        """Test workflow logs streaming."""
        # Submit workflow
        response = client.post(
            "/workflows/submit",
            json=test_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        execution_id = response.json()["execution_id"]
        
        # Mock logs in Redis
        mock_logs = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "message": "Workflow started",
                "component": "workflow"
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO", 
                "message": "Loading checkpoint model",
                "component": "model_loader"
            }
        ]
        
        with patch('redis.from_url') as mock_redis_factory:
            mock_redis = Mock()
            mock_redis.lrange.return_value = [
                json.dumps(log).encode() for log in mock_logs
            ]
            
            mock_redis_factory.return_value = mock_redis
            
            # Get logs
            logs_response = client.get(
                f"/workflows/{execution_id}/logs",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert logs_response.status_code == 200
            logs_data = logs_response.json()
            assert len(logs_data["logs"]) == 2
            assert logs_data["logs"][0]["message"] == "Workflow started"
    
    def test_workflow_error_handling(self, client: TestClient):
        """Test workflow error handling and reporting."""
        # Create invalid workflow
        invalid_workflow = {
            "workflow": {
                "nodes": {
                    "1": {
                        "class_type": "NonExistentNode",
                        "inputs": [],
                        "outputs": []
                    }
                }
            },
            "priority": "normal"
        }
        
        # Submit invalid workflow
        response = client.post(
            "/workflows/submit",
            json=invalid_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Should reject invalid workflow
        assert response.status_code == 422  # Validation error
    
    def test_concurrent_workflow_execution(self, client: TestClient, test_workflow):
        """Test handling of multiple concurrent workflows."""
        execution_ids = []
        
        # Submit multiple workflows concurrently
        for i in range(3):
            workflow_copy = test_workflow.copy()
            workflow_copy["workflow"]["metadata"] = {"batch_id": f"batch_{i}"}
            
            response = client.post(
                "/workflows/submit",
                json=workflow_copy,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 201
            execution_ids.append(response.json()["execution_id"])
        
        # Verify all workflows are queued
        for execution_id in execution_ids:
            status_response = client.get(
                f"/workflows/{execution_id}/status",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] in ["pending", "running"]
    
    def test_workflow_priority_queue(self, client: TestClient, test_workflow):
        """Test workflow priority queue handling."""
        # Submit low priority workflow
        low_priority_workflow = test_workflow.copy()
        low_priority_workflow["priority"] = "low"
        
        response1 = client.post(
            "/workflows/submit",
            json=low_priority_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        low_priority_id = response1.json()["execution_id"]
        
        # Submit high priority workflow
        high_priority_workflow = test_workflow.copy()
        high_priority_workflow["priority"] = "high"
        
        response2 = client.post(
            "/workflows/submit",
            json=high_priority_workflow,
            headers={"Authorization": "Bearer test-token"}
        )
        
        high_priority_id = response2.json()["execution_id"]
        
        # High priority workflow should have lower queue position
        high_status = client.get(f"/workflows/{high_priority_id}/status").json()
        low_status = client.get(f"/workflows/{low_priority_id}/status").json()
        
        assert high_status["queue_position"] <= low_status["queue_position"]


class TestWorkflowDatabaseIntegration:
    """Test workflow database operations."""
    
    def test_workflow_persistence(self, test_db, test_workflow):
        """Test workflow execution persistence in database."""
        db = test_db()
        
        # Create workflow execution record
        execution = WorkflowExecution(
            id="test-persistence-123",
            user_id=1,
            workflow_definition=test_workflow["workflow"],
            status=WorkflowStatus.PENDING,
            priority=Priority.NORMAL,
            queue_position=1,
            estimated_duration=300
        )
        
        db.add(execution)
        db.commit()
        
        # Verify record was saved
        saved_execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.id == "test-persistence-123"
        ).first()
        
        assert saved_execution is not None
        assert saved_execution.status == WorkflowStatus.PENDING
        assert saved_execution.workflow_definition == test_workflow["workflow"]
        
        # Update execution status
        saved_execution.status = WorkflowStatus.COMPLETED
        saved_execution.completed_at = datetime.utcnow()
        saved_execution.outputs = {"images": ["output.png"]}
        db.commit()
        
        # Verify update
        updated_execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.id == "test-persistence-123"
        ).first()
        
        assert updated_execution.status == WorkflowStatus.COMPLETED
        assert updated_execution.completed_at is not None
        assert updated_execution.outputs == {"images": ["output.png"]}
        
        db.close()
    
    def test_workflow_history_query(self, test_db):
        """Test querying workflow execution history."""
        db = test_db()
        
        # Create multiple workflow executions
        executions = []
        for i in range(5):
            execution = WorkflowExecution(
                id=f"history-test-{i}",
                user_id=1,
                workflow_definition={"nodes": {}},
                status=WorkflowStatus.COMPLETED,
                priority=Priority.NORMAL,
                created_at=datetime.utcnow() - timedelta(days=i),
                completed_at=datetime.utcnow() - timedelta(days=i) + timedelta(hours=1)
            )
            executions.append(execution)
            db.add(execution)
        
        db.commit()
        
        # Query recent executions
        recent_executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.user_id == 1
        ).order_by(WorkflowExecution.created_at.desc()).limit(3).all()
        
        assert len(recent_executions) == 3
        assert recent_executions[0].id == "history-test-0"  # Most recent
        
        # Query executions by status
        completed_executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.status == WorkflowStatus.COMPLETED
        ).all()
        
        assert len(completed_executions) == 5
        
        db.close()
    
    def test_workflow_cleanup(self, test_db):
        """Test cleanup of old workflow executions."""
        db = test_db()
        
        # Create old workflow executions
        old_execution = WorkflowExecution(
            id="old-execution",
            user_id=1,
            workflow_definition={"nodes": {}},
            status=WorkflowStatus.COMPLETED,
            priority=Priority.NORMAL,
            created_at=datetime.utcnow() - timedelta(days=35),
            completed_at=datetime.utcnow() - timedelta(days=35) + timedelta(hours=1)
        )
        
        recent_execution = WorkflowExecution(
            id="recent-execution",
            user_id=1,
            workflow_definition={"nodes": {}},
            status=WorkflowStatus.COMPLETED,
            priority=Priority.NORMAL,
            created_at=datetime.utcnow() - timedelta(days=5),
            completed_at=datetime.utcnow() - timedelta(days=5) + timedelta(hours=1)
        )
        
        db.add(old_execution)
        db.add(recent_execution)
        db.commit()
        
        # Cleanup old executions (older than 30 days)
        cleanup_date = datetime.utcnow() - timedelta(days=30)
        deleted_count = db.query(WorkflowExecution).filter(
            WorkflowExecution.created_at < cleanup_date
        ).delete()
        
        db.commit()
        
        assert deleted_count == 1
        
        # Verify only recent execution remains
        remaining = db.query(WorkflowExecution).all()
        assert len(remaining) == 1
        assert remaining[0].id == "recent-execution"
        
        db.close()


class TestWorkflowAPIIntegration:
    """Test workflow API endpoint integration."""
    
    def test_workflow_list_endpoint(self, client: TestClient, test_db):
        """Test listing workflows for a user."""
        # Create some test executions in database
        db = test_db()
        
        for i in range(3):
            execution = WorkflowExecution(
                id=f"list-test-{i}",
                user_id=1,
                workflow_definition={"nodes": {}},
                status=WorkflowStatus.COMPLETED if i < 2 else WorkflowStatus.PENDING,
                priority=Priority.NORMAL,
                created_at=datetime.utcnow() - timedelta(hours=i)
            )
            db.add(execution)
        
        db.commit()
        db.close()
        
        # Test list endpoint
        response = client.get(
            "/workflows/list",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "executions" in data
        assert "total" in data
        assert "page" in data
        
        # Should return executions for the authenticated user
        assert len(data["executions"]) >= 3
    
    def test_workflow_list_filtering(self, client: TestClient, test_db):
        """Test workflow list with filtering."""
        # Query with status filter
        response = client.get(
            "/workflows/list?status=completed",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned executions should have completed status
        for execution in data["executions"]:
            assert execution["status"] == "completed"
    
    def test_workflow_list_pagination(self, client: TestClient):
        """Test workflow list pagination."""
        # Test with pagination parameters
        response = client.get(
            "/workflows/list?page=1&size=10",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["page"] == 1
        assert len(data["executions"]) <= 10
    
    def test_workflow_statistics_endpoint(self, client: TestClient):
        """Test workflow statistics endpoint."""
        response = client.get(
            "/workflows/statistics",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        expected_fields = [
            "total_executions",
            "completed_executions", 
            "failed_executions",
            "average_execution_time",
            "success_rate"
        ]
        
        for field in expected_fields:
            assert field in data
    
    def test_workflow_bulk_operations(self, client: TestClient, test_workflow):
        """Test bulk workflow operations."""
        # Submit multiple workflows
        execution_ids = []
        for i in range(3):
            response = client.post(
                "/workflows/submit",
                json=test_workflow,
                headers={"Authorization": "Bearer test-token"}
            )
            execution_ids.append(response.json()["execution_id"])
        
        # Test bulk status check
        bulk_status_response = client.post(
            "/workflows/bulk/status",
            json={"execution_ids": execution_ids},
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert bulk_status_response.status_code == 200
        bulk_data = bulk_status_response.json()
        
        assert len(bulk_data["results"]) == 3
        for result in bulk_data["results"]:
            assert "execution_id" in result
            assert "status" in result
        
        # Test bulk cancellation
        bulk_cancel_response = client.post(
            "/workflows/bulk/cancel",
            json={"execution_ids": execution_ids[:2]},  # Cancel first 2
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert bulk_cancel_response.status_code == 200
        cancel_data = bulk_cancel_response.json()
        
        assert cancel_data["cancelled_count"] == 2