"""Unit tests for workflow service."""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import fakeredis

try:
    from src.services.workflow import WorkflowService, execute_workflow_task
    from src.models.schemas import WorkflowDefinition, WorkflowStatus, Priority
    from src.models.database import WorkflowExecution
    from src.config.settings import settings
except ImportError:
    pytest.skip("Source modules not available", allow_module_level=True)


class TestWorkflowService:
    """Test cases for WorkflowService."""
    
    @pytest.fixture
    def workflow_service(self, mock_redis, mock_comfyui_client):
        """Create WorkflowService instance with mocked dependencies."""
        with patch('redis.from_url', return_value=mock_redis), \
             patch('src.services.workflow.ComfyUIClient', return_value=mock_comfyui_client), \
             patch('src.services.workflow.model_service') as mock_model_service, \
             patch('src.services.workflow.celery_app') as mock_celery:
            
            mock_model_service.is_model_available = AsyncMock(return_value=True)
            mock_model_service.queue_download = AsyncMock()
            
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_celery.apply_async.return_value = mock_task
            
            service = WorkflowService()
            return service
    
    @pytest.fixture
    def sample_workflow(self):
        """Sample workflow definition."""
        return WorkflowDefinition(
            nodes={
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
        )
    
    @pytest.mark.asyncio
    async def test_submit_workflow_success(self, workflow_service, sample_workflow):
        """Test successful workflow submission."""
        execution_id = str(uuid.uuid4())
        
        with patch('src.utils.validation.validate_workflow') as mock_validate:
            mock_validate.return_value.is_valid = True
            mock_validate.return_value.errors = []
            
            with patch('src.core.database.SessionLocal') as mock_session:
                mock_db = Mock()
                mock_session.return_value.__enter__.return_value = mock_db
                mock_execution = Mock(spec=WorkflowExecution)
                mock_db.query.return_value.filter.return_value.first.return_value = mock_execution
                
                queue_position = await workflow_service.submit_workflow(
                    execution_id=execution_id,
                    workflow=sample_workflow,
                    priority=Priority.NORMAL,
                    user_id=1,
                    webhook_url="https://example.com/webhook",
                    timeout_minutes=30
                )
                
                assert isinstance(queue_position, int)
                assert queue_position >= 1
                mock_db.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_submit_workflow_invalid(self, workflow_service, sample_workflow):
        """Test workflow submission with invalid workflow."""
        execution_id = str(uuid.uuid4())
        
        with patch('src.utils.validation.validate_workflow') as mock_validate:
            mock_validate.return_value.is_valid = False
            mock_validate.return_value.errors = ["Missing required node"]
            
            with pytest.raises(ValueError, match="Invalid workflow"):
                await workflow_service.submit_workflow(
                    execution_id=execution_id,
                    workflow=sample_workflow,
                    priority=Priority.NORMAL,
                    user_id=1
                )
    
    @pytest.mark.asyncio
    async def test_submit_workflow_model_download(self, workflow_service):
        """Test workflow submission triggers model download."""
        execution_id = str(uuid.uuid4())
        
        # Workflow with LoRA model
        workflow = WorkflowDefinition(
            nodes={
                "1": {
                    "class_type": "LoraLoader",
                    "inputs": [
                        {
                            "name": "lora_name",
                            "value": "style_lora.safetensors",
                            "type": "string",
                            "required": True
                        }
                    ],
                    "outputs": ["MODEL", "CLIP"]
                }
            }
        )
        
        with patch('src.utils.validation.validate_workflow') as mock_validate:
            mock_validate.return_value.is_valid = True
            mock_validate.return_value.errors = []
            
            with patch('src.services.workflow.model_service') as mock_model_service:
                mock_model_service.is_model_available = AsyncMock(return_value=False)
                mock_model_service.queue_download = AsyncMock()
                
                with patch('src.core.database.SessionLocal') as mock_session:
                    mock_db = Mock()
                    mock_session.return_value.__enter__.return_value = mock_db
                    mock_execution = Mock(spec=WorkflowExecution)
                    mock_db.query.return_value.filter.return_value.first.return_value = mock_execution
                    
                    await workflow_service.submit_workflow(
                        execution_id=execution_id,
                        workflow=workflow,
                        priority=Priority.HIGH
                    )
                    
                    mock_model_service.queue_download.assert_called_once_with(
                        "style_lora.safetensors", "lora"
                    )
    
    @pytest.mark.asyncio
    async def test_cancel_workflow_success(self, workflow_service):
        """Test successful workflow cancellation."""
        execution_id = str(uuid.uuid4())
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_execution = Mock(spec=WorkflowExecution)
            mock_execution.status = WorkflowStatus.RUNNING
            mock_execution.metadata = {"celery_task_id": "task-123"}
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_execution
            
            with patch('src.services.workflow.celery_app') as mock_celery:
                with patch.object(workflow_service.comfyui_client, 'cancel_execution', return_value=True):
                    result = await workflow_service.cancel_workflow(execution_id)
                    
                    assert result is True
                    mock_celery.control.revoke.assert_called_once_with("task-123", terminate=True)
                    assert mock_execution.status == WorkflowStatus.CANCELLED
                    mock_db.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_cancel_workflow_not_found(self, workflow_service):
        """Test cancelling non-existent workflow."""
        execution_id = str(uuid.uuid4())
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            result = await workflow_service.cancel_workflow(execution_id)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_progress_from_redis(self, workflow_service):
        """Test getting progress from Redis cache."""
        execution_id = str(uuid.uuid4())
        progress_data = {
            "status": "running",
            "progress_percent": 75,
            "elapsed_seconds": 180,
            "eta_seconds": 60
        }
        
        workflow_service.redis_client.set(
            f"workflow:progress:{execution_id}",
            json.dumps(progress_data)
        )
        
        result = await workflow_service.get_progress(execution_id)
        assert result == progress_data
    
    @pytest.mark.asyncio
    async def test_get_progress_from_database(self, workflow_service):
        """Test getting progress from database when Redis is empty."""
        execution_id = str(uuid.uuid4())
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_execution = Mock(spec=WorkflowExecution)
            mock_execution.status = WorkflowStatus.PENDING
            mock_execution.queue_position = 3
            mock_execution.created_at = datetime.utcnow() - timedelta(minutes=2)
            mock_execution.started_at = None
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_execution
            
            result = await workflow_service.get_progress(execution_id)
            
            assert result["status"] == WorkflowStatus.PENDING
            assert result["queue_position"] == 3
            assert "estimated_wait_seconds" in result
    
    @pytest.mark.asyncio
    async def test_get_progress_running_workflow(self, workflow_service):
        """Test getting progress for running workflow."""
        execution_id = str(uuid.uuid4())
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_execution = Mock(spec=WorkflowExecution)
            mock_execution.status = WorkflowStatus.RUNNING
            mock_execution.started_at = datetime.utcnow() - timedelta(minutes=2)
            mock_execution.workflow_definition = {"nodes": {"1": {"class_type": "TestNode"}}}
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_execution
            
            with patch.object(workflow_service, 'estimate_duration', return_value=300):
                result = await workflow_service.get_progress(execution_id)
                
                assert result["status"] == WorkflowStatus.RUNNING
                assert "elapsed_seconds" in result
                assert "estimated_total_seconds" in result
                assert "progress_percent" in result
                assert "eta_seconds" in result
    
    @pytest.mark.asyncio
    async def test_get_live_logs(self, workflow_service):
        """Test getting live logs from Redis."""
        execution_id = str(uuid.uuid4())
        logs_key = f"workflow:logs:{execution_id}"
        
        # Add sample logs to Redis
        log_entries = [
            json.dumps({
                "timestamp": "2024-08-31T10:00:00Z",
                "level": "INFO",
                "message": "Workflow started",
                "component": "workflow"
            }),
            json.dumps({
                "timestamp": "2024-08-31T10:01:00Z", 
                "level": "INFO",
                "message": "Processing node 1",
                "component": "comfyui"
            }),
            json.dumps({
                "timestamp": "2024-08-31T10:01:30Z",
                "level": "ERROR",
                "message": "Node processing failed",
                "component": "comfyui"
            })
        ]
        
        for entry in log_entries:
            workflow_service.redis_client.lpush(logs_key, entry)
        
        # Test getting all logs
        logs = await workflow_service.get_live_logs(execution_id)
        assert len(logs) == 3
        
        # Test filtering by level
        error_logs = await workflow_service.get_live_logs(execution_id, level="ERROR")
        assert len(error_logs) == 1
        assert error_logs[0]["level"] == "ERROR"
        
        # Test tail limit
        limited_logs = await workflow_service.get_live_logs(execution_id, tail=2)
        assert len(limited_logs) == 2
    
    @pytest.mark.asyncio
    async def test_estimate_duration_simple(self, workflow_service):
        """Test duration estimation for simple workflow."""
        workflow = {
            "nodes": {
                "1": {"class_type": "CheckpointLoaderSimple"},
                "2": {"class_type": "SaveImage"}
            }
        }
        
        duration = await workflow_service.estimate_duration(workflow)
        
        # 2 nodes * 2 seconds each = 4 seconds minimum
        assert duration >= 4
        assert duration <= 1800  # Maximum 30 minutes
    
    @pytest.mark.asyncio
    async def test_estimate_duration_heavy_nodes(self, workflow_service):
        """Test duration estimation with heavy processing nodes."""
        workflow = {
            "nodes": {
                "1": {"class_type": "KSampler"},
                "2": {"class_type": "ESRGAN_UPSCALER"},
                "3": {
                    "class_type": "KSampler",
                    "inputs": {
                        "width": 1024,
                        "height": 1024
                    }
                }
            }
        }
        
        duration = await workflow_service.estimate_duration(workflow)
        
        # Should account for heavy nodes and higher resolution
        assert duration > 60  # More than 1 minute for heavy processing
    
    def test_extract_required_models_checkpoint(self, workflow_service):
        """Test extracting checkpoint models from workflow."""
        workflow = WorkflowDefinition(
            nodes={
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": [
                        {
                            "name": "ckpt_name",
                            "value": "realistic_model_v2.safetensors",
                            "type": "string",
                            "required": True
                        }
                    ],
                    "outputs": ["MODEL", "CLIP", "VAE"]
                }
            }
        )
        
        models = workflow_service._extract_required_models(workflow)
        
        assert "realistic_model_v2.safetensors" in models
        assert models["realistic_model_v2.safetensors"] == "checkpoint"
    
    def test_extract_required_models_lora(self, workflow_service):
        """Test extracting LoRA models from workflow."""
        workflow = WorkflowDefinition(
            nodes={
                "1": {
                    "class_type": "LoraLoader",
                    "inputs": [
                        {
                            "name": "lora_name",
                            "value": "style_enhance.safetensors",
                            "type": "string",
                            "required": True
                        }
                    ],
                    "outputs": ["MODEL", "CLIP"]
                }
            }
        )
        
        models = workflow_service._extract_required_models(workflow)
        
        assert "style_enhance.safetensors" in models
        assert models["style_enhance.safetensors"] == "lora"
    
    def test_extract_required_models_vae(self, workflow_service):
        """Test extracting VAE models from workflow."""
        workflow = WorkflowDefinition(
            nodes={
                "1": {
                    "class_type": "VAELoader",
                    "inputs": [
                        {
                            "name": "vae_name",
                            "value": "vae-ft-mse-840000-ema-pruned.ckpt",
                            "type": "string",
                            "required": True
                        }
                    ],
                    "outputs": ["VAE"]
                }
            }
        )
        
        models = workflow_service._extract_required_models(workflow)
        
        assert "vae-ft-mse-840000-ema-pruned.ckpt" in models
        assert models["vae-ft-mse-840000-ema-pruned.ckpt"] == "vae"
    
    def test_get_queue_name(self, workflow_service):
        """Test queue name selection based on priority."""
        assert workflow_service._get_queue_name(Priority.HIGH) == "workflow_high"
        assert workflow_service._get_queue_name(Priority.NORMAL) == "workflow"
        assert workflow_service._get_queue_name(Priority.LOW) == "workflow_low"
    
    @pytest.mark.asyncio
    async def test_get_queue_position(self, workflow_service):
        """Test queue position calculation."""
        execution_id = str(uuid.uuid4())
        
        with patch('src.services.workflow.celery_app') as mock_celery:
            # Mock active and reserved tasks
            mock_celery.control.inspect.return_value.active.return_value = {
                "worker1": [
                    {"queue": "workflow", "id": "task1"},
                    {"queue": "workflow", "id": "task2"}
                ]
            }
            mock_celery.control.inspect.return_value.reserved.return_value = {
                "worker1": [
                    {"queue": "workflow", "id": "task3"}
                ]
            }
            
            position = await workflow_service._get_queue_position(execution_id, Priority.NORMAL)
            
            # 2 active + 1 reserved + 1 for current = 4
            assert position == 4
    
    @pytest.mark.asyncio
    async def test_estimate_queue_wait_time(self, workflow_service):
        """Test queue wait time estimation."""
        queue_position = 5
        
        wait_time = await workflow_service._estimate_queue_wait_time(queue_position)
        
        # 5 positions * 300 seconds (5 minutes) each = 1500 seconds
        assert wait_time == 1500


class TestExecuteWorkflowTask:
    """Test the Celery task for workflow execution."""
    
    @pytest.fixture
    def mock_celery_request(self):
        """Mock Celery task request."""
        mock_request = Mock()
        mock_request.id = "test-task-id"
        mock_request.hostname = "test-worker"
        return mock_request
    
    @pytest.mark.asyncio
    async def test_execute_workflow_task_success(self, mock_celery_request):
        """Test successful workflow execution task."""
        execution_id = str(uuid.uuid4())
        workflow_data = {
            "nodes": {
                "1": {"class_type": "TestNode", "inputs": []}
            }
        }
        
        with patch('src.services.workflow.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_execution = Mock(spec=WorkflowExecution)
            mock_db.query.return_value.filter.return_value.first.return_value = mock_execution
            
            with patch('src.services.workflow.ComfyUIClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.execute_workflow.return_value = {
                    "outputs": {"images": []},
                    "logs": [],
                    "execution_time": 60.0,
                    "status": "completed"
                }
                mock_client_class.return_value = mock_client
                
                with patch('src.services.workflow.webhook_service') as mock_webhook:
                    mock_webhook.send_completion_webhook = AsyncMock()
                    
                    # Mock the task's request attribute
                    with patch('src.services.workflow.execute_workflow_task.request', mock_celery_request):
                        execute_workflow_task(
                            execution_id,
                            workflow_data,
                            user_id=1,
                            webhook_url="https://example.com/webhook"
                        )
                        
                        # Verify execution was updated to running then completed
                        assert mock_execution.status == WorkflowStatus.COMPLETED
                        assert mock_execution.worker_id == "test-worker"
                        assert mock_db.commit.call_count >= 2  # Once for running, once for completed
    
    @pytest.mark.asyncio 
    async def test_execute_workflow_task_failure(self, mock_celery_request):
        """Test failed workflow execution task."""
        execution_id = str(uuid.uuid4())
        workflow_data = {"nodes": {"1": {"class_type": "TestNode"}}}
        
        with patch('src.services.workflow.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_execution = Mock(spec=WorkflowExecution)
            mock_db.query.return_value.filter.return_value.first.return_value = mock_execution
            
            with patch('src.services.workflow.ComfyUIClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.execute_workflow.side_effect = Exception("ComfyUI execution failed")
                mock_client_class.return_value = mock_client
                
                with patch('src.services.workflow.webhook_service') as mock_webhook:
                    mock_webhook.send_completion_webhook = AsyncMock()
                    
                    with patch('src.services.workflow.execute_workflow_task.request', mock_celery_request):
                        with pytest.raises(Exception, match="ComfyUI execution failed"):
                            execute_workflow_task(
                                execution_id,
                                workflow_data,
                                user_id=1,
                                webhook_url="https://example.com/webhook"
                            )
                        
                        # Verify execution was marked as failed
                        assert mock_execution.status == WorkflowStatus.FAILED
                        assert mock_execution.error_message == "ComfyUI execution failed"
    
    def test_execute_workflow_task_execution_not_found(self, mock_celery_request):
        """Test workflow execution task when execution record not found."""
        execution_id = str(uuid.uuid4())
        workflow_data = {"nodes": {"1": {"class_type": "TestNode"}}}
        
        with patch('src.services.workflow.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            with patch('src.services.workflow.execute_workflow_task.request', mock_celery_request):
                # Should return early without raising exception
                execute_workflow_task(execution_id, workflow_data, user_id=1)
                
                # No further processing should occur
                mock_db.commit.assert_not_called()


class TestWorkflowServiceEdgeCases:
    """Edge case tests for WorkflowService."""
    
    @pytest.fixture
    def workflow_service(self):
        """Create WorkflowService instance."""
        with patch('redis.from_url', return_value=fakeredis.FakeRedis()), \
             patch('src.services.workflow.ComfyUIClient'):
            return WorkflowService()
    
    @pytest.mark.asyncio
    async def test_submit_workflow_redis_error(self, workflow_service, sample_workflow):
        """Test workflow submission with Redis connection error."""
        execution_id = str(uuid.uuid4())
        
        with patch.object(workflow_service.redis_client, 'set', side_effect=Exception("Redis connection failed")):
            with patch('src.utils.validation.validate_workflow') as mock_validate:
                mock_validate.return_value.is_valid = True
                mock_validate.return_value.errors = []
                
                with patch('src.core.database.SessionLocal') as mock_session:
                    mock_db = Mock()
                    mock_session.return_value.__enter__.return_value = mock_db
                    mock_execution = Mock(spec=WorkflowExecution)
                    mock_db.query.return_value.filter.return_value.first.return_value = mock_execution
                    
                    # Should still succeed despite Redis error
                    queue_position = await workflow_service.submit_workflow(
                        execution_id=execution_id,
                        workflow=sample_workflow,
                        priority=Priority.NORMAL,
                        user_id=1
                    )
                    
                    assert isinstance(queue_position, int)
    
    @pytest.mark.asyncio
    async def test_get_progress_malformed_redis_data(self, workflow_service):
        """Test getting progress with malformed Redis data."""
        execution_id = str(uuid.uuid4())
        
        # Set malformed JSON in Redis
        workflow_service.redis_client.set(
            f"workflow:progress:{execution_id}",
            "invalid json data"
        )
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            result = await workflow_service.get_progress(execution_id)
            
            # Should fall back to empty result when both Redis and DB fail
            assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_live_logs_redis_error(self, workflow_service):
        """Test getting live logs with Redis error."""
        execution_id = str(uuid.uuid4())
        
        with patch.object(workflow_service.redis_client, 'lrange', side_effect=Exception("Redis error")):
            logs = await workflow_service.get_live_logs(execution_id)
            
            # Should return empty list on error
            assert logs == []
    
    def test_extract_required_models_empty_workflow(self, workflow_service):
        """Test extracting models from empty workflow."""
        workflow = WorkflowDefinition(nodes={})
        
        models = workflow_service._extract_required_models(workflow)
        
        assert models == {}
    
    @pytest.mark.asyncio
    async def test_estimate_duration_edge_cases(self, workflow_service):
        """Test duration estimation edge cases."""
        # Empty workflow
        empty_workflow = {"nodes": {}}
        duration = await workflow_service.estimate_duration(empty_workflow)
        assert duration == 30  # Minimum duration
        
        # Workflow with very large image dimensions
        large_workflow = {
            "nodes": {
                "1": {
                    "class_type": "KSampler",
                    "inputs": {
                        "width": 4096,
                        "height": 4096
                    }
                }
            }
        }
        duration = await workflow_service.estimate_duration(large_workflow)
        assert duration >= 100  # Should account for large resolution
        
        # Workflow with invalid node structure
        invalid_workflow = {"nodes": {"1": "invalid"}}
        duration = await workflow_service.estimate_duration(invalid_workflow)
        assert duration == 300  # Default fallback