"""Unit tests for ComfyUI service."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json
import aiohttp
from typing import Dict, Any

from src.services.comfyui import ComfyUIClient, ComfyUIError


@pytest.fixture
def comfyui_client():
    """Create ComfyUI client instance."""
    return ComfyUIClient(base_url="http://localhost:8188")


@pytest.fixture
def mock_session():
    """Mock aiohttp session."""
    mock = AsyncMock(spec=aiohttp.ClientSession)
    return mock


@pytest.fixture
def sample_workflow():
    """Sample workflow data."""
    return {
        "nodes": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "beautiful landscape",
                    "clip": ["1", 1]
                }
            }
        }
    }


class TestComfyUIClient:
    """Test ComfyUI client functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, comfyui_client, mock_session):
        """Test successful health check."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            result = await comfyui_client.health_check()
        
        assert result is True
        mock_session.get.assert_called_once_with(
            f"{comfyui_client.base_url}/health"
        )
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, comfyui_client, mock_session):
        """Test health check failure."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            result = await comfyui_client.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, comfyui_client, mock_session, sample_workflow):
        """Test successful workflow execution."""
        execution_id = "test-execution-123"
        
        # Mock prompt submission
        submit_response = AsyncMock()
        submit_response.status = 200
        submit_response.json.return_value = {
            "prompt_id": "prompt-123",
            "number": 1,
            "node_errors": {}
        }
        
        # Mock history check
        history_response = AsyncMock()
        history_response.status = 200
        history_response.json.return_value = {
            "prompt-123": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "3": {"images": [{"filename": "output.png", "type": "output"}]}
                }
            }
        }
        
        mock_session.post.return_value.__aenter__.return_value = submit_response
        mock_session.get.return_value.__aenter__.return_value = history_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            result = await comfyui_client.execute_workflow(execution_id, sample_workflow)
        
        assert result["status"] == "completed"
        assert "outputs" in result
        assert len(result["outputs"]["3"]["images"]) == 1
        
        # Verify API calls
        mock_session.post.assert_called_once()
        mock_session.get.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_errors(self, comfyui_client, mock_session, sample_workflow):
        """Test workflow execution with node errors."""
        execution_id = "test-execution-456"
        
        # Mock response with errors
        submit_response = AsyncMock()
        submit_response.status = 200
        submit_response.json.return_value = {
            "node_errors": {
                "1": {
                    "errors": [{"message": "Model not found", "type": "value_error"}],
                    "dependent_outputs": ["2", "3"]
                }
            }
        }
        
        mock_session.post.return_value.__aenter__.return_value = submit_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            with pytest.raises(ComfyUIError, match="Node errors in workflow"):
                await comfyui_client.execute_workflow(execution_id, sample_workflow)
    
    @pytest.mark.asyncio
    async def test_cancel_execution_success(self, comfyui_client, mock_session):
        """Test successful execution cancellation."""
        execution_id = "test-execution-789"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            result = await comfyui_client.cancel_execution(execution_id)
        
        assert result is True
        mock_session.post.assert_called_once_with(
            f"{comfyui_client.base_url}/interrupt",
            json={"execution_id": execution_id}
        )
    
    @pytest.mark.asyncio
    async def test_get_queue_status(self, comfyui_client, mock_session):
        """Test getting queue status."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "exec_info": {
                "queue_remaining": 3,
                "queue_pending": [
                    ["prompt-1", {"execution_id": "exec-1"}],
                    ["prompt-2", {"execution_id": "exec-2"}]
                ]
            }
        }
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            result = await comfyui_client.get_queue_status()
        
        assert result["queue_remaining"] == 3
        assert len(result["queue_pending"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_system_stats(self, comfyui_client, mock_session):
        """Test getting system statistics."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "system": {
                "ram": {"total": 32000, "used": 8000},
                "vram": {"total": 24000, "used": 12000}
            },
            "devices": [
                {"name": "cuda:0", "type": "gpu", "memory_total": 24000}
            ]
        }
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            result = await comfyui_client.get_system_stats()
        
        assert result["system"]["ram"]["total"] == 32000
        assert len(result["devices"]) == 1
        assert result["devices"][0]["type"] == "gpu"
    
    @pytest.mark.asyncio
    async def test_upload_image(self, comfyui_client, mock_session, sample_image):
        """Test image upload."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"name": "uploaded_image.jpg"}
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            result = await comfyui_client.upload_image(sample_image, "test.jpg")
        
        assert result["name"] == "uploaded_image.jpg"
        mock_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_download_output(self, comfyui_client, mock_session):
        """Test downloading output file."""
        filename = "output.png"
        file_content = b"fake image data"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = file_content
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            result = await comfyui_client.download_output(filename)
        
        assert result == file_content
        mock_session.get.assert_called_once_with(
            f"{comfyui_client.base_url}/view",
            params={"filename": filename}
        )
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, comfyui_client, mock_session):
        """Test timeout handling in requests."""
        mock_session.get.side_effect = asyncio.TimeoutError()
        
        with patch.object(comfyui_client, '_session', mock_session):
            with pytest.raises(ComfyUIError, match="Request timeout"):
                await comfyui_client.health_check()
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, comfyui_client, mock_session):
        """Test connection error handling."""
        mock_session.get.side_effect = aiohttp.ClientConnectionError()
        
        with patch.object(comfyui_client, '_session', mock_session):
            with pytest.raises(ComfyUIError, match="Connection failed"):
                await comfyui_client.health_check()
    
    def test_workflow_validation(self, comfyui_client):
        """Test workflow validation."""
        # Valid workflow
        valid_workflow = {
            "nodes": {
                "1": {"class_type": "LoadImage", "inputs": {"image": "test.jpg"}}
            }
        }
        assert comfyui_client._validate_workflow(valid_workflow) is True
        
        # Invalid workflow - no nodes
        invalid_workflow = {"nodes": {}}
        assert comfyui_client._validate_workflow(invalid_workflow) is False
        
        # Invalid workflow - missing class_type
        invalid_workflow2 = {
            "nodes": {
                "1": {"inputs": {"image": "test.jpg"}}
            }
        }
        assert comfyui_client._validate_workflow(invalid_workflow2) is False
    
    def test_progress_calculation(self, comfyui_client):
        """Test progress calculation from ComfyUI status."""
        # Test pending status
        pending_status = {"status_str": "pending", "completed": False}
        assert comfyui_client._calculate_progress(pending_status) == 0
        
        # Test in progress status
        running_status = {
            "status_str": "running", 
            "completed": False,
            "messages": [["progress", {"value": 50, "max": 100}]]
        }
        assert comfyui_client._calculate_progress(running_status) == 50
        
        # Test completed status
        completed_status = {"status_str": "success", "completed": True}
        assert comfyui_client._calculate_progress(completed_status) == 100
    
    @pytest.mark.asyncio
    async def test_websocket_progress_monitoring(self, comfyui_client):
        """Test WebSocket progress monitoring."""
        execution_id = "test-exec-ws"
        
        # Mock WebSocket messages
        ws_messages = [
            {"type": "progress", "data": {"value": 25, "max": 100}},
            {"type": "executing", "data": {"node": "1", "prompt_id": execution_id}},
            {"type": "progress", "data": {"value": 75, "max": 100}},
            {"type": "executed", "data": {"prompt_id": execution_id, "output": {}}}
        ]
        
        with patch('aiohttp.ClientSession.ws_connect') as mock_ws:
            mock_ws_conn = AsyncMock()
            mock_ws_conn.__aiter__.return_value = [
                Mock(data=json.dumps(msg)) for msg in ws_messages
            ]
            mock_ws.return_value.__aenter__.return_value = mock_ws_conn
            
            progress_updates = []
            async for update in comfyui_client.monitor_progress(execution_id):
                progress_updates.append(update)
                if update.get("type") == "executed":
                    break
            
            assert len(progress_updates) == 4
            assert progress_updates[0]["data"]["value"] == 25
            assert progress_updates[-1]["type"] == "executed"


class TestComfyUIErrorHandling:
    """Test ComfyUI error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, comfyui_client, mock_session):
        """Test handling of malformed JSON responses."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        mock_response.text.return_value = "Invalid response"
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            with pytest.raises(ComfyUIError, match="Invalid response"):
                await comfyui_client.health_check()
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, comfyui_client, mock_session):
        """Test rate limit handling."""
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "60"}
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            with pytest.raises(ComfyUIError, match="Rate limited"):
                await comfyui_client.execute_workflow("test", {})
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self, comfyui_client, mock_session):
        """Test server error handling."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            with pytest.raises(ComfyUIError, match="Server error"):
                await comfyui_client.get_system_stats()
    
    @pytest.mark.asyncio
    async def test_execution_timeout(self, comfyui_client, mock_session, sample_workflow):
        """Test execution timeout handling."""
        execution_id = "timeout-test"
        
        # Mock infinite pending status
        submit_response = AsyncMock()
        submit_response.status = 200
        submit_response.json.return_value = {
            "prompt_id": "prompt-timeout",
            "number": 1,
            "node_errors": {}
        }
        
        history_response = AsyncMock()
        history_response.status = 200
        history_response.json.return_value = {
            "prompt-timeout": {
                "status": {"status_str": "pending", "completed": False}
            }
        }
        
        mock_session.post.return_value.__aenter__.return_value = submit_response
        mock_session.get.return_value.__aenter__.return_value = history_response
        
        with patch.object(comfyui_client, '_session', mock_session):
            with patch('asyncio.sleep', side_effect=lambda _: None):  # Speed up the test
                with pytest.raises(ComfyUIError, match="Execution timeout"):
                    await comfyui_client.execute_workflow(
                        execution_id, sample_workflow, timeout_seconds=1
                    )


class TestComfyUIUtilities:
    """Test ComfyUI utility functions."""
    
    def test_node_dependency_analysis(self, comfyui_client):
        """Test node dependency analysis."""
        workflow = {
            "nodes": {
                "1": {
                    "class_type": "CheckpointLoader",
                    "inputs": {"ckpt_name": "model.safetensors"}
                },
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["1", 1], "text": "test"}
                },
                "3": {
                    "class_type": "KSampler",
                    "inputs": {"model": ["1", 0], "positive": ["2", 0]}
                }
            }
        }
        
        dependencies = comfyui_client._analyze_dependencies(workflow)
        
        assert "1" in dependencies  # Node 1 has no dependencies
        assert "2" in dependencies["1"]  # Node 2 depends on node 1
        assert "3" in dependencies["1"]  # Node 3 depends on node 1
        assert "3" in dependencies["2"]  # Node 3 depends on node 2
    
    def test_model_extraction(self, comfyui_client):
        """Test model extraction from workflow."""
        workflow = {
            "nodes": {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
                },
                "2": {
                    "class_type": "LoraLoader",
                    "inputs": {"lora_name": "character_lora.safetensors"}
                }
            }
        }
        
        models = comfyui_client._extract_required_models(workflow)
        
        assert "sd_xl_base_1.0.safetensors" in models
        assert models["sd_xl_base_1.0.safetensors"] == "checkpoint"
        assert "character_lora.safetensors" in models
        assert models["character_lora.safetensors"] == "lora"
    
    def test_execution_time_estimation(self, comfyui_client):
        """Test execution time estimation."""
        simple_workflow = {
            "nodes": {
                "1": {"class_type": "LoadImage", "inputs": {}}
            }
        }
        
        complex_workflow = {
            "nodes": {
                "1": {"class_type": "CheckpointLoader", "inputs": {}},
                "2": {"class_type": "KSampler", "inputs": {"steps": 50}},
                "3": {"class_type": "VAEDecode", "inputs": {}},
                "4": {"class_type": "UpscaleModelLoader", "inputs": {}},
                "5": {"class_type": "ImageUpscaleWithModel", "inputs": {}}
            }
        }
        
        simple_time = comfyui_client._estimate_execution_time(simple_workflow)
        complex_time = comfyui_client._estimate_execution_time(complex_workflow)
        
        assert simple_time < complex_time
        assert simple_time > 0
        assert complex_time > 0