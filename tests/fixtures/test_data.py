"""Enhanced test fixtures and data generators."""
import pytest
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import os

from src.models.schemas import (
    WorkflowDefinition, WorkflowNode, WorkflowNodeInput,
    ModelType, WorkflowStatus, Priority
)
from src.models.database import User, WorkflowExecution


class WorkflowBuilder:
    """Builder for creating complex test workflows."""
    
    def __init__(self):
        self.nodes = {}
        self.metadata = {}
        self._node_counter = 1
    
    def add_checkpoint_loader(self, model_name: str = "test_model.safetensors") -> str:
        """Add checkpoint loader node."""
        node_id = str(self._node_counter)
        self._node_counter += 1
        
        self.nodes[node_id] = WorkflowNode(
            id=node_id,
            class_type="CheckpointLoaderSimple",
            inputs=[
                WorkflowNodeInput(
                    name="ckpt_name",
                    type="string",
                    value=model_name,
                    required=True
                )
            ],
            outputs=["MODEL", "CLIP", "VAE"]
        )
        return node_id
    
    def add_text_encoder(self, text: str, clip_connection: str) -> str:
        """Add CLIP text encoder node."""
        node_id = str(self._node_counter)
        self._node_counter += 1
        
        self.nodes[node_id] = WorkflowNode(
            id=node_id,
            class_type="CLIPTextEncode",
            inputs=[
                WorkflowNodeInput(
                    name="text",
                    type="string",
                    value=text,
                    required=True
                ),
                WorkflowNodeInput(
                    name="clip",
                    type="connection",
                    value=[clip_connection, 1],
                    required=True
                )
            ],
            outputs=["CONDITIONING"]
        )
        return node_id
    
    def add_ksampler(
        self,
        model_connection: str,
        positive_connection: str,
        negative_connection: str,
        steps: int = 20,
        cfg: float = 7.0,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        denoise: float = 1.0,
        seed: int = None
    ) -> str:
        """Add KSampler node."""
        node_id = str(self._node_counter)
        self._node_counter += 1
        
        if seed is None:
            seed = uuid.uuid4().int % (2**32)
        
        self.nodes[node_id] = WorkflowNode(
            id=node_id,
            class_type="KSampler",
            inputs=[
                WorkflowNodeInput(name="model", type="connection", value=[model_connection, 0], required=True),
                WorkflowNodeInput(name="positive", type="connection", value=[positive_connection, 0], required=True),
                WorkflowNodeInput(name="negative", type="connection", value=[negative_connection, 0], required=True),
                WorkflowNodeInput(name="seed", type="int", value=seed, required=True),
                WorkflowNodeInput(name="steps", type="int", value=steps, required=True),
                WorkflowNodeInput(name="cfg", type="float", value=cfg, required=True),
                WorkflowNodeInput(name="sampler_name", type="string", value=sampler_name, required=True),
                WorkflowNodeInput(name="scheduler", type="string", value=scheduler, required=True),
                WorkflowNodeInput(name="denoise", type="float", value=denoise, required=True)
            ],
            outputs=["LATENT"]
        )
        return node_id
    
    def add_vae_decoder(self, samples_connection: str, vae_connection: str) -> str:
        """Add VAE decoder node."""
        node_id = str(self._node_counter)
        self._node_counter += 1
        
        self.nodes[node_id] = WorkflowNode(
            id=node_id,
            class_type="VAEDecode",
            inputs=[
                WorkflowNodeInput(
                    name="samples",
                    type="connection",
                    value=[samples_connection, 0],
                    required=True
                ),
                WorkflowNodeInput(
                    name="vae",
                    type="connection",
                    value=[vae_connection, 2],
                    required=True
                )
            ],
            outputs=["IMAGE"]
        )
        return node_id
    
    def add_save_image(self, images_connection: str, filename_prefix: str = "ComfyUI") -> str:
        """Add save image node."""
        node_id = str(self._node_counter)
        self._node_counter += 1
        
        self.nodes[node_id] = WorkflowNode(
            id=node_id,
            class_type="SaveImage",
            inputs=[
                WorkflowNodeInput(
                    name="images",
                    type="connection",
                    value=[images_connection, 0],
                    required=True
                ),
                WorkflowNodeInput(
                    name="filename_prefix",
                    type="string",
                    value=filename_prefix,
                    required=False
                )
            ],
            outputs=[]
        )
        return node_id
    
    def add_lora_loader(
        self,
        model_connection: str,
        clip_connection: str,
        lora_name: str,
        strength_model: float = 1.0,
        strength_clip: float = 1.0
    ) -> str:
        """Add LoRA loader node."""
        node_id = str(self._node_counter)
        self._node_counter += 1
        
        self.nodes[node_id] = WorkflowNode(
            id=node_id,
            class_type="LoraLoader",
            inputs=[
                WorkflowNodeInput(name="model", type="connection", value=[model_connection, 0], required=True),
                WorkflowNodeInput(name="clip", type="connection", value=[clip_connection, 1], required=True),
                WorkflowNodeInput(name="lora_name", type="string", value=lora_name, required=True),
                WorkflowNodeInput(name="strength_model", type="float", value=strength_model, required=True),
                WorkflowNodeInput(name="strength_clip", type="float", value=strength_clip, required=True)
            ],
            outputs=["MODEL", "CLIP"]
        )
        return node_id
    
    def add_upscaler(self, image_connection: str, upscale_model: str, scale_by: float = 2.0) -> str:
        """Add upscaler node."""
        node_id = str(self._node_counter)
        self._node_counter += 1
        
        self.nodes[node_id] = WorkflowNode(
            id=node_id,
            class_type="ImageUpscaleWithModel",
            inputs=[
                WorkflowNodeInput(name="upscale_model", type="string", value=upscale_model, required=True),
                WorkflowNodeInput(name="image", type="connection", value=[image_connection, 0], required=True),
                WorkflowNodeInput(name="scale_by", type="float", value=scale_by, required=False)
            ],
            outputs=["IMAGE"]
        )
        return node_id
    
    def set_metadata(self, **kwargs):
        """Set workflow metadata."""
        self.metadata.update(kwargs)
        return self
    
    def build(self) -> WorkflowDefinition:
        """Build the workflow definition."""
        return WorkflowDefinition(
            nodes={k: v.dict() for k, v in self.nodes.items()},
            metadata=self.metadata
        )


@pytest.fixture
def workflow_builder():
    """Workflow builder fixture."""
    return WorkflowBuilder


@pytest.fixture
def simple_txt2img_workflow(workflow_builder):
    """Simple text-to-image workflow."""
    builder = workflow_builder()
    
    # Build a basic text-to-image workflow
    checkpoint_node = builder.add_checkpoint_loader("sd_xl_base_1.0.safetensors")
    positive_node = builder.add_text_encoder("a beautiful landscape", checkpoint_node)
    negative_node = builder.add_text_encoder("low quality, blurry", checkpoint_node)
    sampler_node = builder.add_ksampler(
        checkpoint_node, positive_node, negative_node,
        steps=20, cfg=7.0, seed=42
    )
    decoder_node = builder.add_vae_decoder(sampler_node, checkpoint_node)
    builder.add_save_image(decoder_node)
    
    builder.set_metadata(
        title="Simple Text-to-Image",
        description="Basic SDXL text-to-image generation",
        expected_duration=60
    )
    
    return {
        "workflow": builder.build().dict(),
        "priority": "normal"
    }


@pytest.fixture
def complex_workflow_with_lora(workflow_builder):
    """Complex workflow with LoRA."""
    builder = workflow_builder()
    
    checkpoint_node = builder.add_checkpoint_loader("sd_xl_base_1.0.safetensors")
    lora_node = builder.add_lora_loader(
        checkpoint_node, checkpoint_node,
        "character_lora.safetensors",
        strength_model=0.8, strength_clip=0.8
    )
    
    positive_node = builder.add_text_encoder(
        "masterpiece, best quality, detailed character portrait",
        lora_node
    )
    negative_node = builder.add_text_encoder(
        "low quality, blurry, distorted",
        lora_node
    )
    
    sampler_node = builder.add_ksampler(
        lora_node, positive_node, negative_node,
        steps=30, cfg=8.0, sampler_name="dpmpp_2m", scheduler="karras"
    )
    
    decoder_node = builder.add_vae_decoder(sampler_node, checkpoint_node)
    upscaler_node = builder.add_upscaler(decoder_node, "RealESRGAN_x2plus.pth", scale_by=2.0)
    builder.add_save_image(upscaler_node, "character_portrait")
    
    builder.set_metadata(
        title="Character Portrait with LoRA",
        description="High-quality character portrait using LoRA",
        expected_duration=180,
        complexity="high"
    )
    
    return {
        "workflow": builder.build().dict(),
        "priority": "high"
    }


@pytest.fixture
def batch_workflow_data():
    """Generate batch workflow test data."""
    workflows = []
    prompts = [
        "a serene mountain landscape",
        "futuristic city at sunset",
        "ancient forest with mystical creatures",
        "underwater coral reef scene",
        "space station in orbit"
    ]
    
    for i, prompt in enumerate(prompts):
        builder = WorkflowBuilder()
        checkpoint_node = builder.add_checkpoint_loader("sd_xl_base_1.0.safetensors")
        positive_node = builder.add_text_encoder(prompt, checkpoint_node)
        negative_node = builder.add_text_encoder("low quality", checkpoint_node)
        sampler_node = builder.add_ksampler(
            checkpoint_node, positive_node, negative_node,
            seed=i * 1000  # Different seed for each
        )
        decoder_node = builder.add_vae_decoder(sampler_node, checkpoint_node)
        builder.add_save_image(decoder_node, f"batch_{i}")
        
        builder.set_metadata(batch_id=f"batch_{i}", prompt=prompt)
        
        workflows.append({
            "workflow": builder.build().dict(),
            "priority": "normal"
        })
    
    return workflows


@pytest.fixture
def model_test_data():
    """Test data for model management."""
    return {
        "checkpoints": [
            {
                "name": "sd_xl_base_1.0.safetensors",
                "type": ModelType.CHECKPOINT,
                "size_gb": 6.94,
                "description": "Stable Diffusion XL Base Model",
                "download_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
            },
            {
                "name": "sd_xl_refiner_1.0.safetensors", 
                "type": ModelType.CHECKPOINT,
                "size_gb": 6.08,
                "description": "Stable Diffusion XL Refiner",
                "download_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
            }
        ],
        "loras": [
            {
                "name": "character_lora.safetensors",
                "type": ModelType.LORA,
                "size_mb": 144.5,
                "description": "Character style LoRA",
                "download_url": "https://example.com/character_lora.safetensors"
            },
            {
                "name": "landscape_lora.safetensors",
                "type": ModelType.LORA,
                "size_mb": 132.2,
                "description": "Landscape enhancement LoRA",
                "download_url": "https://example.com/landscape_lora.safetensors"
            }
        ],
        "upscalers": [
            {
                "name": "RealESRGAN_x2plus.pth",
                "type": ModelType.UPSCALER,
                "size_mb": 67.1,
                "description": "Real-ESRGAN 2x upscaler",
                "download_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth"
            }
        ]
    }


@pytest.fixture
def user_test_data():
    """Test data for user management."""
    return [
        {
            "email": "user1@example.com",
            "password": "SecurePass123!",
            "username": "user1",
            "is_active": True
        },
        {
            "email": "user2@example.com", 
            "password": "AnotherPass456!",
            "username": "user2",
            "is_active": True
        },
        {
            "email": "inactive@example.com",
            "password": "InactivePass789!",
            "username": "inactive_user",
            "is_active": False
        }
    ]


@pytest.fixture
def execution_history_data():
    """Generate execution history test data."""
    executions = []
    statuses = [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
    priorities = [Priority.LOW, Priority.NORMAL, Priority.HIGH]
    
    base_time = datetime.utcnow()
    
    for i in range(50):  # 50 test executions
        execution_time = base_time - timedelta(hours=i)
        completion_time = execution_time + timedelta(minutes=30 + (i % 120))  # 30-150 minutes
        
        execution = {
            "id": f"test-execution-{i:03d}",
            "user_id": (i % 3) + 1,  # Rotate between users 1, 2, 3
            "workflow_definition": {
                "nodes": {
                    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "test_model.safetensors"}},
                    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}
                },
                "metadata": {"test_execution": i}
            },
            "status": statuses[i % len(statuses)],
            "priority": priorities[i % len(priorities)],
            "created_at": execution_time,
            "started_at": execution_time + timedelta(minutes=i % 10),  # Queue time
            "completed_at": completion_time if i % 4 != 1 else None,  # Some incomplete
            "queue_position": max(1, 10 - i) if i < 10 else 1,
            "estimated_duration": 1800,  # 30 minutes
            "outputs": {"images": [f"output_{i}.png"]} if i % 4 == 0 else None,
            "error_message": "Test error message" if i % 4 == 1 else None,
            "logs": [f"Log entry {j}" for j in range(3)],
            "metadata": {"test": True, "batch_id": f"batch_{i // 10}"}
        }
        
        executions.append(execution)
    
    return executions


@pytest.fixture
def webhook_test_data():
    """Test data for webhook testing."""
    return {
        "valid_urls": [
            "https://webhook.site/test-endpoint",
            "https://example.com/webhook",
            "http://localhost:8080/webhook",
            "https://api.service.com/callbacks/workflow"
        ],
        "invalid_urls": [
            "ftp://example.com/webhook",
            "not-a-url",
            "javascript:alert('xss')",
            "",
            None,
            "http://",
            "https://"
        ],
        "payload_templates": {
            "completion": {
                "type": "workflow.completed",
                "execution_id": "test-execution-123",
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "outputs": {"images": ["output.png"]},
                    "duration_seconds": 45.2
                }
            },
            "progress": {
                "type": "workflow.progress", 
                "execution_id": "test-execution-456",
                "timestamp": datetime.utcnow().isoformat(),
                "progress": {
                    "progress_percent": 65.0,
                    "current_node": "KSampler",
                    "eta_seconds": 30
                }
            },
            "error": {
                "type": "workflow.failed",
                "execution_id": "test-execution-789",
                "success": False,
                "timestamp": datetime.utcnow().isoformat(),
                "error": {
                    "code": "EXECUTION_FAILED",
                    "message": "Node execution failed",
                    "details": "Invalid input parameters"
                }
            }
        }
    }


@pytest.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        "load_test": {
            "concurrent_users": 10,
            "requests_per_user": 50,
            "ramp_up_time": 30,  # seconds
            "test_duration": 300  # 5 minutes
        },
        "stress_test": {
            "max_concurrent_users": 100,
            "step_users": 10,
            "step_duration": 30,
            "max_response_time": 5000  # ms
        },
        "endurance_test": {
            "concurrent_users": 20,
            "test_duration": 3600,  # 1 hour
            "acceptable_error_rate": 0.01  # 1%
        },
        "thresholds": {
            "response_time_p95": 2000,  # ms
            "response_time_p99": 5000,  # ms
            "throughput_min": 10,  # requests/second
            "error_rate_max": 0.05,  # 5%
            "memory_usage_max": 2048  # MB
        }
    }


@pytest.fixture
def temp_test_files():
    """Create temporary test files."""
    files = {}
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test image files
        image_files = ["test1.jpg", "test2.png", "test3.gif"]
        for filename in image_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "wb") as f:
                f.write(b"fake image data for " + filename.encode())
            files[filename] = file_path
        
        # Create test workflow files
        workflow_files = ["simple.json", "complex.json"]
        for filename in workflow_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                json.dump({"test": f"workflow from {filename}"}, f)
            files[filename] = file_path
        
        # Create test model files
        model_files = ["test_model.safetensors", "test_lora.safetensors"]
        for filename in model_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "wb") as f:
                f.write(b"fake model data for " + filename.encode() + b"x" * 1000)
            files[filename] = file_path
        
        yield files
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_external_services():
    """Mock external service responses."""
    return {
        "comfyui": {
            "health_check": {"status": "ok", "version": "1.0.0"},
            "queue_status": {
                "exec_info": {
                    "queue_remaining": 2,
                    "queue_pending": [["prompt1", {}], ["prompt2", {}]]
                }
            },
            "system_stats": {
                "system": {
                    "ram": {"total": 32000, "used": 8000},
                    "vram": {"total": 24000, "used": 6000}
                }
            }
        },
        "redis": {
            "ping": True,
            "info": {"redis_version": "6.2.0", "used_memory_human": "128M"}
        },
        "database": {
            "connection": True,
            "version": "PostgreSQL 13.0"
        },
        "s3": {
            "list_buckets": {"Buckets": [{"Name": "test-bucket"}]},
            "head_bucket": {}
        }
    }


class TestDataFactory:
    """Factory for generating test data."""
    
    @staticmethod
    def create_user(email: str = None, **kwargs) -> Dict[str, Any]:
        """Create user test data."""
        if email is None:
            email = f"user_{uuid.uuid4().hex[:8]}@example.com"
        
        defaults = {
            "email": email,
            "password": "TestPassword123!",
            "username": email.split("@")[0],
            "is_active": True
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def create_workflow_execution(
        execution_id: str = None,
        user_id: int = 1,
        status: WorkflowStatus = WorkflowStatus.PENDING,
        **kwargs
    ) -> Dict[str, Any]:
        """Create workflow execution test data."""
        if execution_id is None:
            execution_id = str(uuid.uuid4())
        
        defaults = {
            "id": execution_id,
            "user_id": user_id,
            "workflow_definition": {"nodes": {"1": {"class_type": "TestNode"}}},
            "status": status,
            "priority": Priority.NORMAL,
            "created_at": datetime.utcnow(),
            "queue_position": 1,
            "estimated_duration": 300
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def create_model_info(name: str = None, model_type: ModelType = ModelType.CHECKPOINT, **kwargs) -> Dict[str, Any]:
        """Create model info test data."""
        if name is None:
            name = f"test_model_{uuid.uuid4().hex[:8]}.safetensors"
        
        defaults = {
            "name": name,
            "type": model_type,
            "version": "1.0",
            "description": f"Test {model_type.value} model",
            "file_size": 1024 * 1024 * 1024,  # 1GB
            "download_url": f"https://example.com/{name}",
            "metadata": {"test": True}
        }
        defaults.update(kwargs)
        return defaults


@pytest.fixture
def test_data_factory():
    """Test data factory fixture."""
    return TestDataFactory