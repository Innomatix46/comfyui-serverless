"""Data generators for testing."""

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from faker import Faker

fake = Faker()


def generate_workflow_data(
    complexity: str = "simple",
    include_metadata: bool = True
) -> Dict[str, Any]:
    """Generate workflow data for testing.
    
    Args:
        complexity: "simple", "medium", or "complex"
        include_metadata: Whether to include metadata
    """
    workflows = {
        "simple": {
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
        "medium": {
            "nodes": {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": [
                        {
                            "name": "ckpt_name", 
                            "value": "realistic_model.safetensors",
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
                    "class_type": "KSampler",
                    "inputs": [
                        {
                            "name": "model",
                            "value": ["1", 0],
                            "type": "connection",
                            "required": True
                        },
                        {
                            "name": "positive",
                            "value": ["2", 0],
                            "type": "connection", 
                            "required": True
                        },
                        {
                            "name": "steps",
                            "value": 20,
                            "type": "int",
                            "required": True
                        },
                        {
                            "name": "cfg",
                            "value": 7.0,
                            "type": "float",
                            "required": True
                        }
                    ],
                    "outputs": ["LATENT"]
                },
                "4": {
                    "class_type": "VAEDecode",
                    "inputs": [
                        {
                            "name": "samples",
                            "value": ["3", 0],
                            "type": "connection",
                            "required": True
                        },
                        {
                            "name": "vae",
                            "value": ["1", 2],
                            "type": "connection",
                            "required": True
                        }
                    ],
                    "outputs": ["IMAGE"]
                },
                "5": {
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
            }
        },
        "complex": {
            "nodes": {
                str(i): {
                    "class_type": random.choice([
                        "CheckpointLoaderSimple", "CLIPTextEncode", "KSampler",
                        "VAEDecode", "LoraLoader", "ControlNetApply", 
                        "SaveImage", "UpscaleModel"
                    ]),
                    "inputs": [
                        {
                            "name": f"input_{j}",
                            "value": fake.word(),
                            "type": random.choice(["string", "int", "float", "connection"]),
                            "required": random.choice([True, False])
                        }
                        for j in range(random.randint(1, 5))
                    ],
                    "outputs": [fake.word().upper() for _ in range(random.randint(1, 3))]
                }
                for i in range(1, 16)  # 15 nodes for complex workflow
            }
        }
    }
    
    workflow = workflows[complexity].copy()
    
    if include_metadata:
        workflow["metadata"] = {
            "title": fake.sentence(nb_words=3),
            "description": fake.paragraph(),
            "author": fake.name(),
            "version": f"{fake.random_int(1, 5)}.{fake.random_int(0, 9)}",
            "tags": [fake.word() for _ in range(3)],
            "created_at": fake.date_time_between(start_date="-1y", end_date="now").isoformat()
        }
    
    return workflow


def generate_user_data(include_password: bool = True) -> Dict[str, Any]:
    """Generate user data for testing."""
    data = {
        "email": fake.email(),
        "username": fake.user_name(),
        "is_active": True,
        "created_at": fake.date_time_between(start_date="-1y", end_date="now")
    }
    
    if include_password:
        data["password"] = fake.password(length=12)
    
    return data


def generate_file_data(
    file_type: str = "image",
    size_mb: Optional[float] = None
) -> Dict[str, Any]:
    """Generate file data for testing.
    
    Args:
        file_type: "image", "video", "audio", "document"
        size_mb: File size in MB, random if None
    """
    type_mapping = {
        "image": {
            "extensions": ["jpg", "jpeg", "png", "gif", "webp"],
            "content_types": ["image/jpeg", "image/png", "image/gif", "image/webp"],
            "size_range": (0.1, 50)  # MB
        },
        "video": {
            "extensions": ["mp4", "avi", "mov", "mkv"],
            "content_types": ["video/mp4", "video/avi", "video/quicktime"],
            "size_range": (10, 1000)
        },
        "audio": {
            "extensions": ["mp3", "wav", "flac", "ogg"], 
            "content_types": ["audio/mp3", "audio/wav", "audio/flac"],
            "size_range": (1, 100)
        },
        "document": {
            "extensions": ["pdf", "doc", "docx", "txt"],
            "content_types": ["application/pdf", "application/msword", "text/plain"],
            "size_range": (0.01, 10)
        }
    }
    
    file_info = type_mapping[file_type]
    ext = random.choice(file_info["extensions"])
    content_type = random.choice(file_info["content_types"])
    
    if size_mb is None:
        size_mb = random.uniform(*file_info["size_range"])
    
    size_bytes = int(size_mb * 1024 * 1024)
    
    return {
        "file_id": str(uuid.uuid4()),
        "filename": f"{fake.file_name(extension=ext)}",
        "original_filename": f"{fake.word()}.{ext}",
        "content_type": content_type,
        "file_size": size_bytes,
        "content": fake.binary(length=min(size_bytes, 1024)),  # Sample content
        "metadata": {
            "format": ext.upper(),
            "created_at": fake.date_time_between(start_date="-1y", end_date="now").isoformat(),
            "source": fake.word()
        }
    }


def generate_comfyui_response(
    status: str = "completed",
    include_outputs: bool = True,
    include_errors: bool = False
) -> Dict[str, Any]:
    """Generate ComfyUI response data.
    
    Args:
        status: "completed", "failed", "running"  
        include_outputs: Whether to include output data
        include_errors: Whether to include error information
    """
    response = {
        "prompt_id": str(uuid.uuid4()),
        "number": random.randint(1, 1000),
        "status": {
            "status_str": status,
            "completed": status == "completed",
            "messages": []
        }
    }
    
    if include_outputs and status == "completed":
        response["outputs"] = {
            str(random.randint(1, 10)): {
                "images": [
                    {
                        "filename": f"{uuid.uuid4()}.png",
                        "subfolder": random.choice(["", "temp", "outputs"]),
                        "type": "output"
                    }
                    for _ in range(random.randint(1, 3))
                ]
            }
        }
    
    if include_errors or status == "failed":
        response["status"]["error"] = {
            "type": fake.word(),
            "message": fake.sentence(),
            "details": fake.paragraph(),
            "node_id": str(random.randint(1, 10))
        }
    
    return response


def generate_execution_metrics() -> Dict[str, Any]:
    """Generate execution metrics for testing."""
    return {
        "total_executions": random.randint(100, 10000),
        "completed_executions": random.randint(80, 9500), 
        "failed_executions": random.randint(5, 500),
        "average_duration_seconds": random.uniform(30, 600),
        "executions_per_minute": random.uniform(0.1, 10),
        "queue_wait_time_seconds": random.uniform(0, 300),
        "cpu_usage_percent": random.uniform(10, 95),
        "memory_usage_percent": random.uniform(20, 90),
        "gpu_usage_percent": random.uniform(0, 100),
        "gpu_memory_usage_percent": random.uniform(0, 100),
        "disk_usage_percent": random.uniform(10, 80),
        "active_executions": random.randint(0, 10),
        "queue_size": random.randint(0, 50)
    }


def generate_batch_test_data(count: int = 100) -> List[Dict[str, Any]]:
    """Generate batch test data for load testing."""
    return [
        {
            "workflow": generate_workflow_data(complexity=random.choice(["simple", "medium"])),
            "priority": random.choice(["low", "normal", "high"]),
            "metadata": {
                "batch_id": str(uuid.uuid4()),
                "test_run": True,
                "sequence": i
            }
        }
        for i in range(count)
    ]


def generate_edge_case_data() -> Dict[str, Any]:
    """Generate edge case test data."""
    return {
        "empty_workflow": {"nodes": {}},
        "single_node": {
            "nodes": {
                "1": {
                    "class_type": "TestNode",
                    "inputs": [],
                    "outputs": []
                }
            }
        },
        "large_workflow": generate_workflow_data(complexity="complex"),
        "unicode_text": {
            "nodes": {
                "1": {
                    "class_type": "CLIPTextEncode",
                    "inputs": [
                        {
                            "name": "text",
                            "value": "🌟✨ Unicode test: 中文, العربية, 日本語 🚀🎨",
                            "type": "string",
                            "required": True
                        }
                    ],
                    "outputs": ["CONDITIONING"]
                }
            }
        },
        "special_characters": {
            "filename": "test!@#$%^&*()_+{}|:<>?[]\;',./`~.jpg",
            "content_type": "image/jpeg"
        },
        "boundary_values": {
            "max_int": 2147483647,
            "min_int": -2147483648,
            "max_float": 1.7976931348623157e+308,
            "empty_string": "",
            "very_long_string": "x" * 10000,
            "null_value": None
        }
    }