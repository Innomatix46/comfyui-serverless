# ComfyUI Serverless API Architecture Specification

## Overview

This document provides a comprehensive API architecture design for ComfyUI serverless deployment, optimized for GPU workloads, asynchronous execution, and scalable processing.

## 1. API Endpoint Specifications

### 1.1 Core Workflow Execution API

#### POST /v1/workflows/execute

**Purpose:** Execute ComfyUI workflow asynchronously with intelligent resource management.

**HTTP Method:** `POST`
**Path:** `/v1/workflows/execute`

**Request Headers:**
- `Content-Type: application/json` (Required)
- `Authorization: Bearer {api_key}` (Required)
- `X-Webhook-URL: {callback_url}` (Optional)
- `X-Client-ID: {client_identifier}` (Optional)
- `X-Request-ID: {idempotency_key}` (Optional)

**Request Body Schema:**
```json
{
  "workflow": {
    "nodes": {
      "1": {
        "inputs": {"text": "beautiful landscape"},
        "class_type": "CLIPTextEncode"
      },
      "2": {
        "inputs": {"noise": ["3", 0], "steps": 20},
        "class_type": "KSampler"
      }
    },
    "metadata": {
      "title": "Text to Image Generation",
      "description": "Standard SDXL workflow",
      "version": "1.0"
    }
  },
  "input_overrides": {
    "prompt": "A serene mountain landscape at sunset",
    "steps": 25,
    "cfg": 7.5,
    "seed": 12345
  },
  "output_config": {
    "format": "png",
    "quality": 95,
    "compression": "lossless",
    "storage": {
      "provider": "s3",
      "bucket": "comfyui-results",
      "path": "outputs/{execution_id}/",
      "public_access": false,
      "retention_days": 30
    }
  },
  "execution_config": {
    "priority": 5,
    "timeout": 1800,
    "gpu_memory_fraction": 0.8,
    "batch_size": 1,
    "enable_optimization": true,
    "retry_attempts": 2
  },
  "notification_config": {
    "webhook_url": "https://api.client.com/comfyui/callback",
    "webhook_secret": "webhook_signature_secret",
    "include_progress": true,
    "include_intermediate": false
  }
}
```

**Response Status Codes:**

**202 Accepted - Workflow Queued Successfully:**
```json
{
  "status": "accepted",
  "execution_id": "exec_7f9e8d2c4a1b",
  "queue_position": 3,
  "estimated_duration": 45,
  "estimated_start_time": "2024-08-31T10:15:30Z",
  "estimated_completion_time": "2024-08-31T10:16:15Z",
  "webhook_url": "https://api.client.com/comfyui/callback",
  "status_url": "/v1/workflows/exec_7f9e8d2c4a1b/status",
  "cancel_url": "/v1/workflows/exec_7f9e8d2c4a1b/cancel"
}
```

**400 Bad Request - Invalid Workflow:**
```json
{
  "status": "error",
  "error_code": "WORKFLOW_VALIDATION_FAILED",
  "message": "Workflow validation failed: Invalid node connections",
  "details": {
    "validation_errors": [
      {
        "node_id": "2",
        "error": "Missing required input connection 'model'",
        "suggestion": "Connect a checkpoint loader node to input 'model'"
      }
    ],
    "missing_nodes": ["CheckpointLoaderSimple"],
    "invalid_connections": [
      {
        "from_node": "1",
        "from_socket": "CLIP", 
        "to_node": "2",
        "to_socket": "positive",
        "error": "Type mismatch: CLIP cannot connect to CONDITIONING"
      }
    ]
  },
  "documentation_url": "https://docs.comfyui-api.com/workflow-validation"
}
```

**401 Unauthorized - Invalid API Key:**
```json
{
  "status": "error",
  "error_code": "UNAUTHORIZED",
  "message": "Invalid or missing API key",
  "details": {
    "auth_required": true,
    "key_format": "Bearer {api_key}",
    "get_key_url": "https://dashboard.comfyui-api.com/api-keys"
  }
}
```

**429 Too Many Requests - Rate Limited:**
```json
{
  "status": "error",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Maximum 10 concurrent executions per API key.",
  "details": {
    "current_executions": 10,
    "max_concurrent": 10,
    "retry_after": 120,
    "upgrade_url": "https://dashboard.comfyui-api.com/billing"
  }
}
```

### 1.2 Workflow Status and Management APIs

#### GET /v1/workflows/{execution_id}/status

**Purpose:** Get real-time status and progress of workflow execution.

**HTTP Method:** `GET`
**Path:** `/v1/workflows/{execution_id}/status`

**Request Headers:**
- `Authorization: Bearer {api_key}` (Required)

**Response 200 OK - Execution Status:**
```json
{
  "execution_id": "exec_7f9e8d2c4a1b",
  "status": "processing",
  "progress": 0.65,
  "current_node": "KSampler",
  "current_step": 13,
  "total_steps": 20,
  "submitted_at": "2024-08-31T10:14:30Z",
  "started_at": "2024-08-31T10:15:30Z",
  "estimated_completion": "2024-08-31T10:16:45Z",
  "estimated_remaining": 75,
  "resource_allocation": {
    "gpu_id": "gpu-node-7",
    "gpu_memory_used": "14.2GB",
    "gpu_memory_total": "24GB",
    "container_id": "container-abc123"
  },
  "execution_log": [
    {
      "timestamp": "2024-08-31T10:15:30Z",
      "event": "execution_started",
      "node_id": null,
      "message": "Workflow execution initiated"
    },
    {
      "timestamp": "2024-08-31T10:15:35Z", 
      "event": "model_loaded",
      "node_id": "1",
      "message": "SDXL model loaded successfully",
      "details": {"memory_used": "6.2GB"}
    },
    {
      "timestamp": "2024-08-31T10:15:45Z",
      "event": "node_completed",
      "node_id": "1",
      "message": "Text encoding completed"
    }
  ],
  "metrics": {
    "total_nodes": 8,
    "completed_nodes": 5,
    "gpu_utilization": 0.89,
    "memory_efficiency": 0.78
  }
}
```

#### DELETE /v1/workflows/{execution_id}/cancel

**Purpose:** Cancel a queued or running workflow execution.

**HTTP Method:** `DELETE`
**Path:** `/v1/workflows/{execution_id}/cancel`

**Response 200 OK - Cancellation Successful:**
```json
{
  "status": "cancelled",
  "execution_id": "exec_7f9e8d2c4a1b",
  "cancelled_at": "2024-08-31T10:16:15Z",
  "reason": "user_requested",
  "refund_eligible": true,
  "partial_results": {
    "available": false,
    "completed_nodes": 3,
    "total_nodes": 8
  }
}
```

### 1.3 Result Management APIs

#### GET /v1/workflows/{execution_id}/results

**Purpose:** Retrieve execution results and output files.

**HTTP Method:** `GET`
**Path:** `/v1/workflows/{execution_id}/results`

**Query Parameters:**
- `include_metadata=true` (Optional) - Include execution metadata
- `download_format=urls|base64` (Optional) - Result delivery format

**Response 200 OK - Results Available:**
```json
{
  "execution_id": "exec_7f9e8d2c4a1b",
  "status": "completed",
  "completed_at": "2024-08-31T10:16:45Z",
  "execution_time": 75,
  "results": {
    "outputs": [
      {
        "node_id": "9",
        "output_type": "IMAGE",
        "file_info": {
          "filename": "output_001.png",
          "size_bytes": 2048576,
          "dimensions": "1024x1024",
          "format": "PNG"
        },
        "storage": {
          "provider": "s3",
          "bucket": "comfyui-results",
          "key": "outputs/exec_7f9e8d2c4a1b/output_001.png",
          "url": "https://signed-url.s3.amazonaws.com/...",
          "expires_at": "2024-08-31T18:16:45Z"
        }
      }
    ],
    "intermediate_outputs": [],
    "metadata": {
      "total_outputs": 1,
      "total_size_bytes": 2048576,
      "generation_seed": 12345,
      "model_used": "sdxl_base_1.0.safetensors"
    }
  },
  "execution_metrics": {
    "total_time": 75,
    "queue_time": 60,
    "processing_time": 75,
    "gpu_time": 68,
    "memory_peak": "14.8GB",
    "gpu_utilization_avg": 0.87,
    "cost_breakdown": {
      "gpu_compute": 0.0125,
      "storage": 0.0001,
      "bandwidth": 0.0002,
      "total": 0.0128
    }
  }
}
```

### 1.4 Model and Asset Management APIs

#### GET /v1/models/list

**Purpose:** List available models and their metadata.

**HTTP Method:** `GET`
**Path:** `/v1/models/list`

**Query Parameters:**
- `category=checkpoint|lora|controlnet|vae` (Optional)
- `search={query}` (Optional)
- `limit=50` (Optional)
- `offset=0` (Optional)

**Response 200 OK:**
```json
{
  "models": [
    {
      "model_id": "sdxl_base_1.0",
      "name": "SDXL Base 1.0",
      "category": "checkpoint",
      "file_info": {
        "filename": "sdxl_base_1.0.safetensors",
        "size_bytes": 6938078208,
        "hash": "sha256:31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893d7e5b"
      },
      "capabilities": {
        "resolution": "1024x1024",
        "styles": ["realistic", "artistic"],
        "languages": ["en"]
      },
      "performance": {
        "memory_requirement": "6.2GB",
        "avg_inference_time": 12.5,
        "cache_priority": 9
      },
      "availability": {
        "regions": ["us-east-1", "eu-west-1"],
        "warm_cache": true,
        "load_time": 8.2
      }
    }
  ],
  "pagination": {
    "total": 45,
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

#### POST /v1/models/preload

**Purpose:** Preload models into cache for faster execution.

**Request Body:**
```json
{
  "models": [
    {
      "model_id": "sdxl_base_1.0",
      "priority": 9,
      "regions": ["us-east-1"]
    }
  ],
  "cache_duration": 3600
}
```

### 1.5 Queue Management APIs

#### GET /v1/queue/status

**Purpose:** Get current queue status and system capacity.

**Response 200 OK:**
```json
{
  "queue_status": {
    "total_queued": 15,
    "total_processing": 8,
    "avg_wait_time": 45,
    "avg_processing_time": 67
  },
  "system_capacity": {
    "total_gpu_nodes": 12,
    "available_gpu_nodes": 4,
    "total_gpu_memory": "288GB",
    "available_gpu_memory": "96GB",
    "utilization": 0.67
  },
  "queue_breakdown": [
    {
      "priority": 10,
      "count": 2,
      "avg_wait_time": 15
    },
    {
      "priority": 5,
      "count": 8,
      "avg_wait_time": 60
    },
    {
      "priority": 1,
      "count": 5,
      "avg_wait_time": 180
    }
  ]
}
```

## 2. Authentication and Authorization

### 2.1 API Key Authentication

**Authentication Method:** Bearer Token (API Key)

**Header Format:**
```
Authorization: Bearer comfyui_sk_1234567890abcdef
```

**API Key Types:**
- `comfyui_sk_*` - Secret keys for server-to-server
- `comfyui_pk_*` - Public keys for client-side (limited permissions)

### 2.2 Permission Levels

```json
{
  "permission_levels": {
    "basic": {
      "max_concurrent_executions": 3,
      "max_queue_size": 10,
      "timeout_limit": 300,
      "models": ["basic_models"],
      "gpu_types": ["T4"]
    },
    "pro": {
      "max_concurrent_executions": 10,
      "max_queue_size": 50,
      "timeout_limit": 1800,
      "models": ["all_models"],
      "gpu_types": ["T4", "A10G", "A100"]
    },
    "enterprise": {
      "max_concurrent_executions": 100,
      "max_queue_size": 1000,
      "timeout_limit": 3600,
      "models": ["all_models", "custom_models"],
      "gpu_types": ["all"],
      "dedicated_resources": true
    }
  }
}
```

## 3. Error Handling and Status Codes

### 3.1 Standard HTTP Status Codes

- **200 OK**: Request completed successfully
- **202 Accepted**: Request accepted and queued for processing
- **400 Bad Request**: Invalid request format or parameters
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Insufficient permissions for operation
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflict (duplicate execution ID)
- **422 Unprocessable Entity**: Valid format but invalid workflow
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Unexpected server error
- **502 Bad Gateway**: Upstream service error
- **503 Service Unavailable**: System overloaded or maintenance

### 3.2 Error Response Schema

```json
{
  "status": "error",
  "error_code": "WORKFLOW_EXECUTION_FAILED",
  "message": "Workflow execution failed during KSampler node",
  "details": {
    "execution_id": "exec_7f9e8d2c4a1b",
    "failed_node": "2",
    "node_type": "KSampler",
    "error_type": "OUT_OF_MEMORY",
    "gpu_memory_used": "23.8GB",
    "gpu_memory_limit": "24GB",
    "suggested_fixes": [
      "Reduce batch_size to 1",
      "Use lower resolution (512x512)",
      "Enable CPU offloading"
    ]
  },
  "retry_info": {
    "retryable": true,
    "retry_delay": 60,
    "max_retries": 2,
    "remaining_retries": 1
  },
  "support": {
    "documentation_url": "https://docs.comfyui-api.com/errors/out-of-memory",
    "contact_url": "https://support.comfyui-api.com"
  }
}
```

## 4. Webhook System

### 4.1 Webhook Event Types

**Execution Events:**
- `execution.started` - Workflow execution began
- `execution.progress` - Progress update (configurable frequency)
- `execution.completed` - Workflow completed successfully
- `execution.failed` - Workflow execution failed
- `execution.cancelled` - Workflow was cancelled

**System Events:**
- `system.maintenance` - Scheduled maintenance notification
- `system.capacity` - System capacity alerts
- `model.updated` - Model availability changes

### 4.2 Webhook Payload Schema

```json
{
  "event_type": "execution.completed",
  "event_id": "evt_9a8b7c6d5e4f",
  "timestamp": "2024-08-31T10:16:45Z",
  "execution_id": "exec_7f9e8d2c4a1b",
  "data": {
    "status": "completed",
    "execution_time": 75,
    "results": {
      "output_count": 1,
      "output_urls": [
        "https://signed-url.s3.amazonaws.com/outputs/exec_7f9e8d2c4a1b/output_001.png"
      ]
    },
    "metrics": {
      "gpu_time": 68,
      "memory_peak": "14.8GB",
      "cost": 0.0128
    }
  },
  "signature": "sha256=d4f5e6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
  "delivery_attempt": 1,
  "max_retries": 3
}
```

### 4.3 Webhook Security

**Signature Verification:**
```python
import hmac
import hashlib

def verify_webhook_signature(payload: str, signature: str, secret: str) -> bool:
    expected_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected_signature}", signature)
```

## 5. Input Validation and Schema

### 5.1 Workflow Validation Rules

**Node Validation:**
```json
{
  "validation_rules": {
    "required_fields": ["class_type", "inputs"],
    "node_types": {
      "whitelist": ["LoadImage", "CLIPTextEncode", "KSampler", "VAEDecode", "SaveImage"],
      "blacklist": ["ExecuteCode", "SystemCommand"],
      "custom_nodes": {
        "enabled": true,
        "require_verification": true,
        "security_scan": true
      }
    },
    "input_validation": {
      "max_string_length": 10000,
      "max_numeric_value": 1000000,
      "allowed_url_schemes": ["https"],
      "max_image_size": "50MB",
      "supported_formats": ["png", "jpg", "webp"]
    },
    "resource_limits": {
      "max_nodes": 100,
      "max_connections": 500,
      "max_execution_time": 3600,
      "max_memory_usage": "24GB"
    }
  }
}
```

### 5.2 Input Sanitization

**Security Measures:**
- URL validation for external resources
- Image format and size validation
- Text input sanitization (XSS prevention)
- Model hash verification
- Custom node security scanning

## 6. Rate Limiting and Quotas

### 6.1 Rate Limiting Strategy

**Rate Limit Tiers:**
```json
{
  "rate_limits": {
    "basic": {
      "requests_per_minute": 10,
      "concurrent_executions": 3,
      "monthly_quota": 100,
      "burst_allowance": 5
    },
    "pro": {
      "requests_per_minute": 60,
      "concurrent_executions": 10,
      "monthly_quota": 1000,
      "burst_allowance": 20
    },
    "enterprise": {
      "requests_per_minute": 300,
      "concurrent_executions": 100,
      "monthly_quota": 10000,
      "burst_allowance": 100,
      "custom_limits": true
    }
  }
}
```

### 6.2 Rate Limit Headers

**Response Headers:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1724234567
X-RateLimit-Retry-After: 15
```

## 7. API Versioning Strategy

### 7.1 Version Management

**URL Versioning:**
- Current: `/v1/workflows/execute`
- Future: `/v2/workflows/execute`

**Version Support Policy:**
- Support 2 major versions simultaneously
- 6-month deprecation notice for old versions
- Backward compatibility within major versions

### 7.2 Version Headers

**Request/Response Headers:**
```
API-Version: v1
API-Deprecated: false
API-Sunset: 2025-08-31
```

## 8. Documentation and OpenAPI Specification

### 8.1 OpenAPI 3.0 Schema

**Basic Structure:**
```yaml
openapi: 3.0.3
info:
  title: ComfyUI Serverless API
  version: 1.0.0
  description: High-performance serverless API for ComfyUI workflow execution
  contact:
    name: API Support
    url: https://support.comfyui-api.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.comfyui-serverless.com/v1
    description: Production server
  - url: https://staging-api.comfyui-serverless.com/v1  
    description: Staging server

security:
  - ApiKeyAuth: []

components:
  securitySchemes:
    ApiKeyAuth:
      type: http
      scheme: bearer
      bearerFormat: API Key
```

This API architecture provides a comprehensive foundation for ComfyUI serverless deployment with robust error handling, security, and scalability considerations.