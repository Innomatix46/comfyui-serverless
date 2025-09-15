"""Unit tests for validation utilities."""
import pytest
from unittest.mock import Mock, patch

from src.utils.validation import (
    validate_workflow,
    validate_file_upload,
    validate_model_name,
    validate_webhook_url,
    WorkflowValidator
)
from src.models.schemas import WorkflowDefinition, WorkflowNode, WorkflowNodeInput


class TestWorkflowValidator:
    """Test workflow validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = WorkflowValidator()
    
    def test_validate_empty_workflow(self):
        """Test validation of empty workflow."""
        workflow = WorkflowDefinition(nodes={})
        result = self.validator.validate(workflow)
        
        assert not result.is_valid
        assert "Workflow must contain at least one node" in result.errors
    
    def test_validate_valid_workflow(self):
        """Test validation of valid workflow."""
        nodes = {
            "1": WorkflowNode(
                class_type="CheckpointLoaderSimple",
                inputs=[
                    WorkflowNodeInput(name="ckpt_name", value="model.safetensors", type="string", required=True)
                ],
                outputs=["MODEL", "CLIP", "VAE"]
            ),
            "2": WorkflowNode(
                class_type="KSampler", 
                inputs=[
                    WorkflowNodeInput(name="model", value=["1", 0], type="connection", required=True),
                    WorkflowNodeInput(name="positive", value="test", type="string", required=True),
                    WorkflowNodeInput(name="negative", value="", type="string", required=True),
                    WorkflowNodeInput(name="latent_image", value="empty", type="string", required=True)
                ],
                outputs=["LATENT"]
            ),
            "3": WorkflowNode(
                class_type="SaveImage",
                inputs=[
                    WorkflowNodeInput(name="images", value=["2", 0], type="connection", required=True)
                ],
                outputs=[]
            )
        }
        
        workflow = WorkflowDefinition(nodes=nodes)
        result = self.validator.validate(workflow)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_unsupported_node_type(self):
        """Test validation of unsupported node type."""
        nodes = {
            "1": WorkflowNode(
                class_type="UnsupportedNode",
                inputs=[],
                outputs=[]
            )
        }
        
        workflow = WorkflowDefinition(nodes=nodes)
        result = self.validator.validate(workflow)
        
        assert not result.is_valid
        assert "Unsupported node type 'UnsupportedNode'" in result.errors[0]
    
    def test_validate_missing_required_input(self):
        """Test validation of missing required input."""
        nodes = {
            "1": WorkflowNode(
                class_type="CheckpointLoaderSimple",
                inputs=[],  # Missing required ckpt_name input
                outputs=[]
            )
        }
        
        workflow = WorkflowDefinition(nodes=nodes)
        result = self.validator.validate(workflow)
        
        assert not result.is_valid
        assert "Missing required input 'ckpt_name'" in result.errors[0]
    
    def test_validate_invalid_connection(self):
        """Test validation of invalid node connection."""
        nodes = {
            "1": WorkflowNode(
                class_type="KSampler",
                inputs=[
                    WorkflowNodeInput(name="model", value=["nonexistent", 0], type="connection", required=True),
                    WorkflowNodeInput(name="positive", value="test", type="string", required=True),
                    WorkflowNodeInput(name="negative", value="", type="string", required=True),
                    WorkflowNodeInput(name="latent_image", value="empty", type="string", required=True)
                ],
                outputs=[]
            )
        }
        
        workflow = WorkflowDefinition(nodes=nodes)
        result = self.validator.validate(workflow)
        
        assert not result.is_valid
        assert "references non-existent node 'nonexistent'" in result.errors[0]
    
    def test_validate_performance_warnings(self):
        """Test performance warning generation."""
        nodes = {
            "1": WorkflowNode(
                class_type="KSampler",
                inputs=[
                    WorkflowNodeInput(name="width", value=4096, type="int", required=False),
                    WorkflowNodeInput(name="height", value=4096, type="int", required=False),
                    WorkflowNodeInput(name="steps", value=100, type="int", required=False),
                ],
                outputs=[]
            )
        }
        
        workflow = WorkflowDefinition(nodes=nodes)
        result = self.validator.validate(workflow)
        
        assert "High resolution" in str(result.warnings)
        assert "High step count" in str(result.warnings)


class TestFileValidation:
    """Test file upload validation."""
    
    def test_valid_file_upload(self):
        """Test validation of valid file upload."""
        result = validate_file_upload(
            filename="test.jpg",
            content_type="image/jpeg", 
            file_size=1024 * 1024  # 1MB
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_file_too_large(self):
        """Test validation of oversized file."""
        result = validate_file_upload(
            filename="large.jpg",
            content_type="image/jpeg",
            file_size=200 * 1024 * 1024  # 200MB
        )
        
        assert not result.is_valid
        assert "exceeds maximum allowed size" in result.errors[0]
    
    def test_empty_filename(self):
        """Test validation of empty filename."""
        result = validate_file_upload(
            filename="",
            content_type="image/jpeg",
            file_size=1024
        )
        
        assert not result.is_valid
        assert "Filename cannot be empty" in result.errors
    
    def test_dangerous_file_extension(self):
        """Test validation of dangerous file extensions."""
        result = validate_file_upload(
            filename="malware.exe",
            content_type="application/octet-stream",
            file_size=1024
        )
        
        assert not result.is_valid
        assert "File type not allowed" in result.errors[0]
    
    def test_unsupported_content_type(self):
        """Test validation of unsupported content type."""
        result = validate_file_upload(
            filename="test.xyz",
            content_type="application/unknown",
            file_size=1024
        )
        
        assert result.is_valid  # Should be valid but with warning
        assert "may not be supported" in result.warnings[0]


class TestModelValidation:
    """Test model name validation."""
    
    def test_valid_model_name(self):
        """Test validation of valid model name."""
        result = validate_model_name("stable_diffusion_v1.5.safetensors")
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_empty_model_name(self):
        """Test validation of empty model name."""
        result = validate_model_name("")
        
        assert not result.is_valid
        assert "Model name cannot be empty" in result.errors
    
    def test_model_name_too_long(self):
        """Test validation of overly long model name."""
        long_name = "a" * 300
        result = validate_model_name(long_name)
        
        assert not result.is_valid
        assert "Model name too long" in result.errors[0]
    
    def test_model_name_invalid_characters(self):
        """Test validation of model name with invalid characters."""
        result = validate_model_name("model<test>name.safetensors")
        
        assert not result.is_valid
        assert "contains invalid characters" in result.errors[0]


class TestWebhookValidation:
    """Test webhook URL validation."""
    
    def test_valid_https_webhook(self):
        """Test validation of valid HTTPS webhook."""
        result = validate_webhook_url("https://api.example.com/webhook")
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_valid_localhost_webhook(self):
        """Test validation of valid localhost webhook."""
        result = validate_webhook_url("http://localhost:3000/webhook")
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_empty_webhook_url(self):
        """Test validation of empty webhook URL."""
        result = validate_webhook_url("")
        
        assert not result.is_valid
        assert "Webhook URL cannot be empty" in result.errors
    
    def test_invalid_webhook_format(self):
        """Test validation of invalid webhook URL format."""
        result = validate_webhook_url("not-a-url")
        
        assert not result.is_valid
        assert "Invalid webhook URL format" in result.errors
    
    def test_insecure_webhook_warning(self):
        """Test validation of insecure webhook URL."""
        result = validate_webhook_url("http://api.example.com/webhook")
        
        assert not result.is_valid  # Should fail security check
        assert "should use HTTPS" in result.errors[0]


@pytest.mark.asyncio
async def test_validate_workflow_async():
    """Test async workflow validation."""
    nodes = {
        "1": WorkflowNode(
            class_type="CheckpointLoaderSimple",
            inputs=[
                WorkflowNodeInput(name="ckpt_name", value="model.safetensors", type="string", required=True)
            ],
            outputs=["MODEL"]
        )
    }
    
    workflow = WorkflowDefinition(nodes=nodes)
    result = await validate_workflow(workflow)
    
    assert result.is_valid