"""Test data factories using factory_boy."""

import factory
import factory.fuzzy
from datetime import datetime, timedelta
from faker import Faker

try:
    from src.models.database import User, WorkflowExecution, FileUpload
    from src.models.schemas import WorkflowStatus, Priority
except ImportError:
    # Handle import errors for CI environments
    User = None
    WorkflowExecution = None
    FileUpload = None
    WorkflowStatus = None
    Priority = None

fake = Faker()


class UserFactory(factory.Factory):
    """Factory for User models."""
    
    class Meta:
        model = User
        
    email = factory.Sequence(lambda n: f"user{n}@example.com")
    username = factory.LazyAttribute(lambda obj: obj.email.split('@')[0])
    hashed_password = "$2b$12$test.password.hash"
    is_active = True
    created_at = factory.LazyFunction(datetime.utcnow)


class WorkflowFactory(factory.Factory):
    """Factory for WorkflowExecution models."""
    
    class Meta:
        model = WorkflowExecution
        
    id = factory.LazyFunction(lambda: fake.uuid4())
    user_id = 1
    workflow_definition = factory.LazyFunction(lambda: {
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
        },
        "metadata": {
            "title": fake.sentence(nb_words=3),
            "description": fake.paragraph()
        }
    })
    status = WorkflowStatus.PENDING if WorkflowStatus else "pending"
    priority = Priority.NORMAL if Priority else "normal"
    created_at = factory.LazyFunction(datetime.utcnow)
    queue_position = factory.fuzzy.FuzzyInteger(1, 10)
    metadata = factory.LazyFunction(lambda: {"test": True, "created_by": "factory"})


class FileUploadFactory(factory.Factory):
    """Factory for FileUpload models."""
    
    class Meta:
        model = FileUpload
        
    id = factory.LazyFunction(lambda: fake.uuid4())
    user_id = 1
    filename = factory.LazyFunction(lambda: fake.uuid4())
    original_filename = factory.LazyFunction(lambda: fake.file_name(category="image"))
    content_type = "image/jpeg"
    file_size = factory.fuzzy.FuzzyInteger(1024, 10*1024*1024)  # 1KB to 10MB
    storage_path = factory.LazyAttribute(lambda obj: f"files/{obj.filename}")
    storage_type = "local"
    is_uploaded = True
    created_at = factory.LazyFunction(datetime.utcnow)
    expires_at = factory.LazyFunction(lambda: datetime.utcnow() + timedelta(days=7))
    metadata = factory.LazyFunction(lambda: {"format": "JPEG", "quality": 85})


class ExecutionFactory(WorkflowFactory):
    """Factory for completed workflow executions."""
    
    status = WorkflowStatus.COMPLETED if WorkflowStatus else "completed"
    started_at = factory.LazyFunction(lambda: datetime.utcnow() - timedelta(minutes=5))
    completed_at = factory.LazyFunction(datetime.utcnow)
    outputs = factory.LazyFunction(lambda: {
        "images": [
            {
                "filename": f"{fake.uuid4()}.png",
                "subfolder": "",
                "type": "output"
            }
        ]
    })
    logs = factory.LazyFunction(lambda: [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "message": "Workflow started",
            "component": "workflow"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO", 
            "message": "Workflow completed successfully",
            "component": "workflow"
        }
    ])


class FailedExecutionFactory(WorkflowFactory):
    """Factory for failed workflow executions."""
    
    status = WorkflowStatus.FAILED if WorkflowStatus else "failed"
    started_at = factory.LazyFunction(lambda: datetime.utcnow() - timedelta(minutes=2))
    completed_at = factory.LazyFunction(datetime.utcnow)
    error_message = factory.LazyFunction(lambda: fake.sentence(nb_words=8))
    logs = factory.LazyFunction(lambda: [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "message": "Workflow started",
            "component": "workflow"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "message": "Workflow failed with error",
            "component": "workflow"
        }
    ])