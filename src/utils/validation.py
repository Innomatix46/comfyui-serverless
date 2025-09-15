"""Validation utilities for workflows and data."""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import jsonschema
from src.models.schemas import WorkflowDefinition


class ValidationResult(BaseModel):
    """Validation result model."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class WorkflowValidator:
    """Validator for ComfyUI workflows."""
    
    def __init__(self):
        self.supported_node_types = {
            # Core nodes
            'CheckpointLoaderSimple',
            'CheckpointLoader',
            'LoraLoader',
            'VAELoader',
            'VAEDecode',
            'VAEEncode',
            
            # Sampling nodes
            'KSampler',
            'KSamplerAdvanced',
            'SamplerCustom',
            
            # Conditioning nodes
            'CLIPTextEncode',
            'ConditioningCombine',
            'ConditioningSetArea',
            
            # Image nodes
            'LoadImage',
            'SaveImage',
            'PreviewImage',
            'ImageScale',
            'ImageCrop',
            
            # ControlNet nodes
            'ControlNetLoader',
            'ControlNetApply',
            
            # Upscaling nodes
            'ESRGAN_UPSCALER',
            'UpscaleModelLoader',
            
            # Utility nodes
            'Note',
            'Reroute',
            'PrimitiveNode'
        }
        
        self.required_inputs = {
            'CheckpointLoaderSimple': ['ckpt_name'],
            'LoraLoader': ['model', 'clip', 'lora_name'],
            'VAELoader': ['vae_name'],
            'VAEDecode': ['samples', 'vae'],
            'VAEEncode': ['pixels', 'vae'],
            'KSampler': ['model', 'positive', 'negative', 'latent_image'],
            'CLIPTextEncode': ['text', 'clip'],
            'LoadImage': ['image'],
            'SaveImage': ['images']
        }
    
    def validate(self, workflow: WorkflowDefinition) -> ValidationResult:
        """Validate a workflow definition."""
        errors = []
        warnings = []
        
        try:
            # Basic structure validation
            if not workflow.nodes:
                errors.append("Workflow must contain at least one node")
                return ValidationResult(is_valid=False, errors=errors)
            
            # Validate individual nodes
            for node_id, node in workflow.nodes.items():
                node_errors = self._validate_node(node_id, node)
                errors.extend(node_errors)
            
            # Validate workflow connections
            connection_errors, connection_warnings = self._validate_connections(workflow)
            errors.extend(connection_errors)
            warnings.extend(connection_warnings)
            
            # Validate workflow completeness
            completeness_errors = self._validate_completeness(workflow)
            errors.extend(completeness_errors)
            
            # Performance warnings
            performance_warnings = self._check_performance_issues(workflow)
            warnings.extend(performance_warnings)
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def _validate_node(self, node_id: str, node: Any) -> List[str]:
        """Validate an individual node."""
        errors = []
        
        # Check if node type is supported
        if node.class_type not in self.supported_node_types:
            errors.append(f"Node {node_id}: Unsupported node type '{node.class_type}'")
        
        # Check required inputs
        if node.class_type in self.required_inputs:
            required = self.required_inputs[node.class_type]
            input_names = {inp.name for inp in node.inputs}
            
            for required_input in required:
                if required_input not in input_names:
                    errors.append(
                        f"Node {node_id}: Missing required input '{required_input}'"
                    )
        
        # Validate input values
        for input_item in node.inputs:
            if input_item.required and input_item.value is None:
                errors.append(
                    f"Node {node_id}: Required input '{input_item.name}' cannot be null"
                )
            
            # Type-specific validations
            if input_item.name in ['width', 'height', 'steps', 'seed']:
                if not isinstance(input_item.value, int) or input_item.value < 0:
                    errors.append(
                        f"Node {node_id}: Input '{input_item.name}' must be a positive integer"
                    )
            
            elif input_item.name in ['cfg', 'denoise']:
                if not isinstance(input_item.value, (int, float)) or input_item.value < 0:
                    errors.append(
                        f"Node {node_id}: Input '{input_item.name}' must be a positive number"
                    )
        
        return errors
    
    def _validate_connections(self, workflow: WorkflowDefinition) -> tuple[List[str], List[str]]:
        """Validate node connections."""
        errors = []
        warnings = []
        
        # Track which nodes produce outputs and which consume inputs
        output_producers = {}
        input_consumers = {}
        
        for node_id, node in workflow.nodes.items():
            # Check for connection-type inputs (usually lists/references)
            for input_item in node.inputs:
                if isinstance(input_item.value, list) and len(input_item.value) == 2:
                    # This looks like a connection [source_node_id, output_index]
                    source_node = input_item.value[0]
                    output_index = input_item.value[1]
                    
                    if source_node not in workflow.nodes:
                        errors.append(
                            f"Node {node_id}: Input '{input_item.name}' references "
                            f"non-existent node '{source_node}'"
                        )
                    
                    # Track connections
                    if source_node not in output_producers:
                        output_producers[source_node] = []
                    output_producers[source_node].append((node_id, input_item.name))
                    
                    input_consumers[node_id] = input_consumers.get(node_id, []) + [input_item.name]
        
        # Check for orphaned nodes (nodes with no connections)
        connected_nodes = set(output_producers.keys()) | set(input_consumers.keys())
        all_nodes = set(workflow.nodes.keys())
        orphaned_nodes = all_nodes - connected_nodes
        
        for node_id in orphaned_nodes:
            node_type = workflow.nodes[node_id].class_type
            if node_type not in ['Note', 'PrimitiveNode']:  # These can be standalone
                warnings.append(f"Node {node_id}: Node appears to be disconnected from workflow")
        
        return errors, warnings
    
    def _validate_completeness(self, workflow: WorkflowDefinition) -> List[str]:
        """Validate workflow completeness (has proper inputs and outputs)."""
        errors = []
        
        # Check for essential workflow components
        has_model_loader = False
        has_sampler = False
        has_output = False
        has_text_encoder = False
        
        for node_id, node in workflow.nodes.items():
            class_type = node.class_type
            
            if class_type in ['CheckpointLoaderSimple', 'CheckpointLoader']:
                has_model_loader = True
            
            elif class_type in ['KSampler', 'KSamplerAdvanced', 'SamplerCustom']:
                has_sampler = True
            
            elif class_type in ['SaveImage', 'PreviewImage']:
                has_output = True
            
            elif class_type == 'CLIPTextEncode':
                has_text_encoder = True
        
        if not has_model_loader:
            errors.append("Workflow missing model loader (CheckpointLoader)")
        
        if not has_sampler:
            errors.append("Workflow missing sampler node (KSampler)")
        
        if not has_output:
            errors.append("Workflow missing output node (SaveImage/PreviewImage)")
        
        if not has_text_encoder:
            warnings = warnings if 'warnings' in locals() else []
            warnings.append("Workflow missing text encoder (CLIPTextEncode)")
        
        return errors
    
    def _check_performance_issues(self, workflow: WorkflowDefinition) -> List[str]:
        """Check for potential performance issues."""
        warnings = []
        
        node_count = len(workflow.nodes)
        if node_count > 50:
            warnings.append(f"Large workflow with {node_count} nodes may be slow to execute")
        
        # Check for high-resolution settings
        for node_id, node in workflow.nodes.items():
            for input_item in node.inputs:
                if input_item.name in ['width', 'height']:
                    if isinstance(input_item.value, int) and input_item.value > 2048:
                        warnings.append(
                            f"Node {node_id}: High resolution ({input_item.value}) may require "
                            f"significant GPU memory and processing time"
                        )
                
                elif input_item.name == 'steps':
                    if isinstance(input_item.value, int) and input_item.value > 50:
                        warnings.append(
                            f"Node {node_id}: High step count ({input_item.value}) will "
                            f"increase processing time significantly"
                        )
        
        return warnings


# Global validator instance
workflow_validator = WorkflowValidator()


async def validate_workflow(workflow: WorkflowDefinition) -> ValidationResult:
    """Validate a workflow definition."""
    return workflow_validator.validate(workflow)


def validate_file_upload(filename: str, content_type: str, file_size: int) -> ValidationResult:
    """Validate file upload parameters."""
    errors = []
    warnings = []
    
    # Check file size
    max_size = 100 * 1024 * 1024  # 100MB
    if file_size > max_size:
        errors.append(f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)")
    
    # Check filename
    if not filename or len(filename.strip()) == 0:
        errors.append("Filename cannot be empty")
    
    if len(filename) > 255:
        errors.append("Filename too long (maximum 255 characters)")
    
    # Check for potentially dangerous file extensions
    dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
    if any(filename.lower().endswith(ext) for ext in dangerous_extensions):
        errors.append(f"File type not allowed: {filename}")
    
    # Check content type
    allowed_types = [
        'image/jpeg', 'image/png', 'image/webp', 'image/tiff', 'image/bmp',
        'application/json', 'text/plain', 'application/octet-stream'
    ]
    
    if content_type not in allowed_types:
        warnings.append(f"Content type '{content_type}' may not be supported")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def validate_model_name(model_name: str) -> ValidationResult:
    """Validate model name."""
    errors = []
    
    if not model_name or len(model_name.strip()) == 0:
        errors.append("Model name cannot be empty")
    
    if len(model_name) > 255:
        errors.append("Model name too long (maximum 255 characters)")
    
    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    if any(char in model_name for char in invalid_chars):
        errors.append(f"Model name contains invalid characters: {', '.join(invalid_chars)}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_webhook_url(webhook_url: str) -> ValidationResult:
    """Validate webhook URL."""
    errors = []
    
    if not webhook_url:
        errors.append("Webhook URL cannot be empty")
        return ValidationResult(is_valid=False, errors=errors)
    
    # Basic URL validation
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(webhook_url):
        errors.append("Invalid webhook URL format")
    
    # Security checks
    if not webhook_url.startswith('https://') and not webhook_url.startswith('http://localhost'):
        errors.append("Webhook URL should use HTTPS for security")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)