"""Unit tests for model management service."""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime, timedelta

from src.services.model import ModelService, ModelError
from src.models.schemas import ModelType


@pytest.fixture
def model_service():
    """Create model service instance."""
    return ModelService(models_dir="/tmp/test_models")


@pytest.fixture
def temp_models_dir():
    """Create temporary models directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_redis():
    """Mock Redis client for model caching."""
    mock = Mock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = 1
    mock.exists.return_value = 0
    mock.hget.return_value = None
    mock.hset.return_value = 1
    mock.hdel.return_value = 1
    mock.expire.return_value = True
    return mock


@pytest.fixture
def sample_model_info():
    """Sample model information."""
    return {
        "name": "test_model.safetensors",
        "type": ModelType.CHECKPOINT,
        "version": "1.0",
        "description": "Test checkpoint model",
        "file_size": 7000000000,  # 7GB
        "download_url": "https://example.com/model.safetensors",
        "metadata": {
            "base_model": "SD 1.5",
            "training_steps": 50000,
            "resolution": "512x512"
        }
    }


class TestModelService:
    """Test model service functionality."""
    
    def test_model_service_initialization(self, temp_models_dir):
        """Test model service initialization."""
        service = ModelService(models_dir=temp_models_dir)
        
        assert service.models_dir == Path(temp_models_dir)
        assert service.models_dir.exists()
        
        # Check subdirectories are created
        for model_type in ModelType:
            assert (service.models_dir / model_type.value).exists()
    
    @pytest.mark.asyncio
    async def test_list_available_models(self, model_service, temp_models_dir):
        """Test listing available models."""
        # Create test model files
        checkpoints_dir = Path(temp_models_dir) / "checkpoint"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        (checkpoints_dir / "model1.safetensors").write_text("fake model data")
        (checkpoints_dir / "model2.ckpt").write_text("fake model data")
        
        lora_dir = Path(temp_models_dir) / "lora"
        lora_dir.mkdir(parents=True, exist_ok=True)
        (lora_dir / "lora1.safetensors").write_text("fake lora data")
        
        with patch.object(model_service, 'models_dir', Path(temp_models_dir)):
            models = await model_service.list_available_models()
        
        assert len(models) == 3
        
        # Check model types are correctly identified
        checkpoint_models = [m for m in models if m["type"] == ModelType.CHECKPOINT]
        lora_models = [m for m in models if m["type"] == ModelType.LORA]
        
        assert len(checkpoint_models) == 2
        assert len(lora_models) == 1
    
    @pytest.mark.asyncio
    async def test_is_model_available(self, model_service, temp_models_dir):
        """Test checking if model is available."""
        # Create test model file
        checkpoints_dir = Path(temp_models_dir) / "checkpoint"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / "available_model.safetensors").write_text("fake data")
        
        with patch.object(model_service, 'models_dir', Path(temp_models_dir)):
            # Test existing model
            assert await model_service.is_model_available(
                "available_model.safetensors", ModelType.CHECKPOINT
            ) is True
            
            # Test non-existing model
            assert await model_service.is_model_available(
                "missing_model.safetensors", ModelType.CHECKPOINT
            ) is False
    
    @pytest.mark.asyncio
    async def test_download_model_success(self, model_service, mock_redis, temp_models_dir):
        """Test successful model download."""
        model_name = "test_model.safetensors"
        model_type = ModelType.CHECKPOINT
        download_url = "https://example.com/model.safetensors"
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-length": "1000"}
        mock_response.content.iter_chunked.return_value = [b"chunk1", b"chunk2", b"chunk3"]
        
        with patch('aiohttp.ClientSession.get') as mock_get, \
             patch.object(model_service, 'redis_client', mock_redis), \
             patch.object(model_service, 'models_dir', Path(temp_models_dir)):
            
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await model_service.download_model(
                model_name, model_type, download_url
            )
        
        assert result is True
        
        # Verify file was created
        model_path = Path(temp_models_dir) / model_type.value / model_name
        assert model_path.exists()
        assert model_path.read_text() == "chunk1chunk2chunk3"
        
        # Verify Redis calls for progress tracking
        mock_redis.hset.assert_called()
        mock_redis.expire.assert_called()
    
    @pytest.mark.asyncio
    async def test_download_model_failure(self, model_service):
        """Test model download failure."""
        model_name = "failing_model.safetensors"
        model_type = ModelType.CHECKPOINT
        download_url = "https://example.com/nonexistent.safetensors"
        
        # Mock HTTP 404 response
        mock_response = AsyncMock()
        mock_response.status = 404
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await model_service.download_model(
                model_name, model_type, download_url
            )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_download_progress(self, model_service, mock_redis):
        """Test getting download progress."""
        model_name = "downloading_model.safetensors"
        
        # Mock Redis data
        mock_redis.hget.side_effect = lambda key, field: {
            "status": "downloading",
            "downloaded": "500000000",  # 500MB
            "total": "1000000000",     # 1GB
            "speed_mbps": "10.5"
        }.get(field)
        
        with patch.object(model_service, 'redis_client', mock_redis):
            progress = await model_service.get_download_progress(model_name)
        
        assert progress["status"] == "downloading"
        assert progress["progress_percent"] == 50.0
        assert progress["speed_mbps"] == 10.5
        assert progress["downloaded_mb"] == 500.0
        assert progress["total_mb"] == 1000.0
    
    @pytest.mark.asyncio
    async def test_delete_model(self, model_service, temp_models_dir):
        """Test model deletion."""
        model_name = "to_delete.safetensors"
        model_type = ModelType.CHECKPOINT
        
        # Create test model file
        checkpoints_dir = Path(temp_models_dir) / "checkpoint"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoints_dir / model_name
        model_path.write_text("fake model data")
        
        with patch.object(model_service, 'models_dir', Path(temp_models_dir)):
            result = await model_service.delete_model(model_name, model_type)
        
        assert result is True
        assert not model_path.exists()
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, model_service, temp_models_dir):
        """Test getting model information."""
        model_name = "info_model.safetensors"
        model_type = ModelType.CHECKPOINT
        
        # Create test model file
        checkpoints_dir = Path(temp_models_dir) / "checkpoint"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoints_dir / model_name
        model_path.write_text("fake model data")
        
        # Create metadata file
        metadata = {
            "name": model_name,
            "type": model_type.value,
            "description": "Test model",
            "base_model": "SD 1.5"
        }
        metadata_path = checkpoints_dir / f"{model_name}.json"
        metadata_path.write_text(json.dumps(metadata))
        
        with patch.object(model_service, 'models_dir', Path(temp_models_dir)):
            info = await model_service.get_model_info(model_name, model_type)
        
        assert info["name"] == model_name
        assert info["type"] == model_type
        assert info["file_size"] > 0
        assert "description" in info
    
    @pytest.mark.asyncio
    async def test_load_model_memory_management(self, model_service, temp_models_dir):
        """Test model loading with memory management."""
        model_name = "memory_test.safetensors"
        model_type = ModelType.CHECKPOINT
        
        # Create test model file
        checkpoints_dir = Path(temp_models_dir) / "checkpoint"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / model_name).write_text("fake model data")
        
        with patch.object(model_service, 'models_dir', Path(temp_models_dir)), \
             patch('src.utils.gpu.get_gpu_memory_info') as mock_gpu_info:
            
            mock_gpu_info.return_value = {
                'total_mb': 24000,
                'used_mb': 8000,
                'free_mb': 16000,
                'utilization_percent': 33.3
            }
            
            result = await model_service.load_model(model_name, model_type)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_unload_model(self, model_service):
        """Test model unloading."""
        model_name = "loaded_model.safetensors"
        model_type = ModelType.CHECKPOINT
        
        # Mock loaded models tracking
        model_service._loaded_models = {
            f"{model_type.value}/{model_name}": {
                "loaded_at": datetime.now(),
                "memory_usage_mb": 3500.0
            }
        }
        
        result = await model_service.unload_model(model_name, model_type)
        
        assert result is True
        assert f"{model_type.value}/{model_name}" not in model_service._loaded_models
    
    @pytest.mark.asyncio
    async def test_cleanup_old_models(self, model_service, temp_models_dir, mock_redis):
        """Test cleanup of old/unused models."""
        # Create test model files with different ages
        checkpoints_dir = Path(temp_models_dir) / "checkpoint"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Old model (should be cleaned up)
        old_model = checkpoints_dir / "old_model.safetensors"
        old_model.write_text("fake old model")
        old_time = datetime.now() - timedelta(days=40)
        os.utime(old_model, (old_time.timestamp(), old_time.timestamp()))
        
        # Recent model (should be kept)
        recent_model = checkpoints_dir / "recent_model.safetensors"
        recent_model.write_text("fake recent model")
        
        with patch.object(model_service, 'models_dir', Path(temp_models_dir)), \
             patch.object(model_service, 'redis_client', mock_redis):
            
            # Mock Redis to show old model as unused
            mock_redis.hget.return_value = None  # No usage data
            
            cleaned = await model_service.cleanup_old_models(max_age_days=30)
        
        assert cleaned > 0
        assert not old_model.exists()
        assert recent_model.exists()
    
    @pytest.mark.asyncio
    async def test_get_memory_usage(self, model_service):
        """Test getting memory usage statistics."""
        # Mock loaded models
        model_service._loaded_models = {
            "checkpoint/model1.safetensors": {
                "loaded_at": datetime.now(),
                "memory_usage_mb": 3500.0
            },
            "lora/lora1.safetensors": {
                "loaded_at": datetime.now(),
                "memory_usage_mb": 500.0
            }
        }
        
        with patch('src.utils.gpu.get_gpu_memory_info') as mock_gpu_info:
            mock_gpu_info.return_value = {
                'total_mb': 24000,
                'used_mb': 12000,
                'free_mb': 12000,
                'utilization_percent': 50.0
            }
            
            usage = await model_service.get_memory_usage()
        
        assert usage["total_memory_mb"] == 24000
        assert usage["used_memory_mb"] == 12000
        assert usage["model_memory_mb"] == 4000.0  # 3500 + 500
        assert len(usage["loaded_models"]) == 2
    
    @pytest.mark.asyncio
    async def test_model_search(self, model_service, temp_models_dir):
        """Test model search functionality."""
        # Create test models with metadata
        checkpoints_dir = Path(temp_models_dir) / "checkpoint"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Model 1 - realistic style
        (checkpoints_dir / "realistic_v1.safetensors").write_text("fake model")
        metadata1 = {
            "name": "realistic_v1.safetensors",
            "description": "Photorealistic model for portraits",
            "tags": ["realistic", "portrait", "human"]
        }
        (checkpoints_dir / "realistic_v1.safetensors.json").write_text(json.dumps(metadata1))
        
        # Model 2 - anime style
        (checkpoints_dir / "anime_v2.safetensors").write_text("fake model")
        metadata2 = {
            "name": "anime_v2.safetensors",
            "description": "Anime style model for characters",
            "tags": ["anime", "character", "cartoon"]
        }
        (checkpoints_dir / "anime_v2.safetensors.json").write_text(json.dumps(metadata2))
        
        with patch.object(model_service, 'models_dir', Path(temp_models_dir)):
            # Search by tag
            realistic_models = await model_service.search_models(query="realistic")
            assert len(realistic_models) == 1
            assert "realistic_v1.safetensors" in realistic_models[0]["name"]
            
            # Search by description
            anime_models = await model_service.search_models(query="anime")
            assert len(anime_models) == 1
            assert "anime_v2.safetensors" in anime_models[0]["name"]
            
            # Search by model type
            checkpoint_models = await model_service.search_models(
                model_type=ModelType.CHECKPOINT
            )
            assert len(checkpoint_models) == 2


class TestModelServiceEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_download_with_insufficient_disk_space(self, model_service):
        """Test download failure due to insufficient disk space."""
        model_name = "huge_model.safetensors"
        model_type = ModelType.CHECKPOINT
        download_url = "https://example.com/huge_model.safetensors"
        
        # Mock response with huge file
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-length": "100000000000"}  # 100GB
        
        with patch('aiohttp.ClientSession.get') as mock_get, \
             patch('shutil.disk_usage') as mock_disk_usage:
            
            mock_get.return_value.__aenter__.return_value = mock_response
            mock_disk_usage.return_value = (1000000000, 500000000, 500000000)  # 500MB free
            
            with pytest.raises(ModelError, match="Insufficient disk space"):
                await model_service.download_model(model_name, model_type, download_url)
    
    @pytest.mark.asyncio
    async def test_concurrent_model_downloads(self, model_service, mock_redis):
        """Test handling of concurrent downloads for the same model."""
        model_name = "concurrent_model.safetensors"
        model_type = ModelType.CHECKPOINT
        download_url = "https://example.com/model.safetensors"
        
        # Mock Redis to show download already in progress
        mock_redis.hget.side_effect = lambda key, field: {
            "status": "downloading",
            "downloaded": "100000000",
            "total": "1000000000"
        }.get(field)
        
        with patch.object(model_service, 'redis_client', mock_redis):
            result = await model_service.download_model(model_name, model_type, download_url)
        
        # Should return False (already downloading)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_corrupted_model_detection(self, model_service, temp_models_dir):
        """Test detection and handling of corrupted models."""
        model_name = "corrupted_model.safetensors"
        model_type = ModelType.CHECKPOINT
        
        # Create corrupted model file (too small)
        checkpoints_dir = Path(temp_models_dir) / "checkpoint"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = checkpoints_dir / model_name
        corrupted_path.write_text("corrupted")  # Very small file
        
        with patch.object(model_service, 'models_dir', Path(temp_models_dir)):
            is_valid = await model_service.validate_model(model_name, model_type)
        
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_model_version_management(self, model_service, temp_models_dir):
        """Test model versioning functionality."""
        model_name = "versioned_model"
        model_type = ModelType.CHECKPOINT
        
        # Create multiple versions
        checkpoints_dir = Path(temp_models_dir) / "checkpoint"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        for version in ["v1.0", "v1.1", "v2.0"]:
            model_file = checkpoints_dir / f"{model_name}_{version}.safetensors"
            model_file.write_text(f"fake model data {version}")
            
            # Create metadata with version info
            metadata = {
                "name": f"{model_name}_{version}.safetensors",
                "version": version,
                "base_name": model_name
            }
            metadata_file = checkpoints_dir / f"{model_name}_{version}.safetensors.json"
            metadata_file.write_text(json.dumps(metadata))
        
        with patch.object(model_service, 'models_dir', Path(temp_models_dir)):
            versions = await model_service.list_model_versions(model_name, model_type)
        
        assert len(versions) == 3
        assert "v2.0" in [v["version"] for v in versions]
        
        # Test getting latest version
        latest = await model_service.get_latest_version(model_name, model_type)
        assert latest["version"] == "v2.0"
    
    @pytest.mark.asyncio
    async def test_model_dependency_resolution(self, model_service):
        """Test resolving model dependencies."""
        workflow = {
            "nodes": {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "base_model.safetensors"}
                },
                "2": {
                    "class_type": "LoraLoader", 
                    "inputs": {
                        "model": ["1", 0],
                        "lora_name": "style_lora.safetensors"
                    }
                },
                "3": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": "vae_model.safetensors"}
                }
            }
        }
        
        dependencies = await model_service.resolve_dependencies(workflow)
        
        assert "base_model.safetensors" in dependencies
        assert dependencies["base_model.safetensors"] == ModelType.CHECKPOINT
        assert "style_lora.safetensors" in dependencies
        assert dependencies["style_lora.safetensors"] == ModelType.LORA
        assert "vae_model.safetensors" in dependencies
        assert dependencies["vae_model.safetensors"] == ModelType.VAE
    
    @pytest.mark.asyncio
    async def test_model_usage_tracking(self, model_service, mock_redis):
        """Test model usage tracking and statistics."""
        model_name = "tracked_model.safetensors"
        model_type = ModelType.CHECKPOINT
        
        with patch.object(model_service, 'redis_client', mock_redis):
            await model_service.track_model_usage(model_name, model_type)
        
        # Verify Redis calls for usage tracking
        mock_redis.hset.assert_called()
        mock_redis.hincrby.assert_called()
        mock_redis.expire.assert_called()
    
    @pytest.mark.asyncio
    async def test_model_preloading_optimization(self, model_service):
        """Test intelligent model preloading based on usage patterns."""
        # Mock usage statistics
        usage_stats = {
            "popular_model.safetensors": {
                "usage_count": 150,
                "last_used": datetime.now() - timedelta(hours=2),
                "avg_load_time": 45.0
            },
            "rarely_used_model.safetensors": {
                "usage_count": 3,
                "last_used": datetime.now() - timedelta(days=5),
                "avg_load_time": 60.0
            }
        }
        
        with patch.object(model_service, '_get_usage_statistics', return_value=usage_stats):
            preload_candidates = await model_service.get_preload_candidates(max_models=1)
        
        assert len(preload_candidates) == 1
        assert "popular_model.safetensors" in preload_candidates[0]["name"]