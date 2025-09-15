# ComfyUI Serverless API Reference

Complete API reference documentation for the ComfyUI Serverless API.

## Table of Contents

- [Authentication](#authentication)
- [Workflows](#workflows)
- [Models](#models)
- [Files](#files)
- [Health & Monitoring](#health--monitoring)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

## Base URL

- **Production**: `https://api.comfyui-serverless.com`
- **Staging**: `https://staging-api.comfyui-serverless.com`
- **Local Development**: `http://localhost:8000`

## Authentication

The API uses JWT Bearer token authentication. All authenticated endpoints require an `Authorization` header:

```
Authorization: Bearer <your_access_token>
```

### Register User

Create a new user account.

**Endpoint**: `POST /auth/register`

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "username": "johndoe"
}
```

**Response** (201 Created):
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": null
}
```

### Login

Authenticate and receive access tokens.

**Endpoint**: `POST /auth/login`

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Refresh Token

Refresh an expired access token using a refresh token.

**Endpoint**: `POST /auth/refresh`

**Request Body**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Get Current User

Get information about the currently authenticated user.

**Endpoint**: `GET /auth/me`

**Response** (200 OK):
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

### Logout

Logout and blacklist the current token.

**Endpoint**: `POST /auth/logout`

**Response** (200 OK):
```json
{
  "message": "Successfully logged out"
}
```

## Workflows

Execute and manage ComfyUI workflows.

### Execute Workflow

Submit a ComfyUI workflow for asynchronous execution.

**Endpoint**: `POST /workflows/execute`

**Request Body**:
```json
{
  "workflow": {
    "nodes": {
      "1": {
        "id": "1",
        "class_type": "CheckpointLoaderSimple",
        "inputs": [
          {
            "name": "ckpt_name",
            "type": "STRING",
            "value": "v1-5-pruned-emaonly.ckpt",
            "required": true
          }
        ]
      },
      "2": {
        "id": "2", 
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "beautiful landscape, masterpiece",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["1", 1],
            "required": true
          }
        ]
      },
      "3": {
        "id": "3",
        "class_type": "KSampler",
        "inputs": [
          {
            "name": "seed",
            "type": "INT",
            "value": 12345,
            "required": true
          },
          {
            "name": "steps",
            "type": "INT", 
            "value": 20,
            "required": true
          },
          {
            "name": "cfg",
            "type": "FLOAT",
            "value": 7.0,
            "required": true
          },
          {
            "name": "sampler_name",
            "type": "STRING",
            "value": "euler",
            "required": true
          },
          {
            "name": "scheduler",
            "type": "STRING", 
            "value": "normal",
            "required": true
          },
          {
            "name": "positive",
            "type": "CONDITIONING",
            "value": ["2", 0],
            "required": true
          },
          {
            "name": "model",
            "type": "MODEL",
            "value": ["1", 0],
            "required": true
          }
        ]
      }
    },
    "metadata": {
      "description": "Simple text-to-image generation",
      "version": "1.0"
    }
  },
  "priority": "normal",
  "webhook_url": "https://your-app.com/webhook",
  "metadata": {
    "user_tag": "batch_001"
  },
  "timeout_minutes": 30
}
```

**Response** (200 OK):
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending", 
  "created_at": "2024-01-01T00:00:00Z",
  "estimated_duration": 120,
  "queue_position": 3
}
```

### Get Workflow Result

Get the result of a workflow execution.

**Endpoint**: `GET /workflows/{execution_id}`

**Response** (200 OK):
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "started_at": "2024-01-01T00:01:00Z",
  "completed_at": "2024-01-01T00:03:00Z", 
  "duration_seconds": 120.5,
  "outputs": {
    "images": [
      {
        "filename": "output_001.png",
        "url": "https://storage.example.com/outputs/output_001.png",
        "width": 512,
        "height": 512
      }
    ]
  },
  "error": null,
  "logs": [
    {
      "timestamp": "2024-01-01T00:01:00Z",
      "level": "INFO",
      "message": "Starting workflow execution"
    },
    {
      "timestamp": "2024-01-01T00:03:00Z",
      "level": "INFO", 
      "message": "Workflow completed successfully"
    }
  ],
  "metadata": {
    "user_tag": "batch_001",
    "model_used": "v1-5-pruned-emaonly.ckpt"
  }
}
```

### List Workflow Executions

List user's workflow executions with optional filtering.

**Endpoint**: `GET /workflows/`

**Query Parameters**:
- `status` (optional): Filter by status (`pending`, `running`, `completed`, `failed`, `cancelled`)
- `limit` (optional): Number of results (1-100, default: 20)
- `offset` (optional): Pagination offset (default: 0)

**Response** (200 OK):
```json
[
  {
    "execution_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "started_at": "2024-01-01T00:01:00Z",
    "completed_at": "2024-01-01T00:03:00Z",
    "duration_seconds": 120.5,
    "outputs": {...},
    "error": null,
    "logs": [...],
    "metadata": {...}
  }
]
```

### Cancel Workflow

Cancel a pending or running workflow.

**Endpoint**: `POST /workflows/{execution_id}/cancel`

**Response** (200 OK):
```json
{
  "message": "Workflow cancelled successfully"
}
```

### Get Workflow Status

Get real-time status and progress of a workflow execution.

**Endpoint**: `GET /workflows/{execution_id}/status`

**Response** (200 OK):
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "created_at": "2024-01-01T00:00:00Z",
  "started_at": "2024-01-01T00:01:00Z",
  "queue_position": null,
  "progress": {
    "current_step": 15,
    "total_steps": 20,
    "percentage": 75.0,
    "current_node": "3",
    "eta_seconds": 30
  },
  "estimated_completion": "2024-01-01T00:03:30Z"
}
```

### Get Workflow Logs

Get execution logs for a workflow.

**Endpoint**: `GET /workflows/{execution_id}/logs`

**Query Parameters**:
- `level` (optional): Filter by log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `tail` (optional): Number of recent logs (1-1000, default: 100)

**Response** (200 OK):
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "logs": [
    {
      "timestamp": "2024-01-01T00:01:00Z",
      "level": "INFO",
      "message": "Loading model: v1-5-pruned-emaonly.ckpt",
      "node_id": "1"
    },
    {
      "timestamp": "2024-01-01T00:01:30Z", 
      "level": "INFO",
      "message": "Encoding text prompt",
      "node_id": "2"
    }
  ],
  "total_count": 25,
  "filtered": false
}
```

### Retry Failed Workflow

Retry a failed workflow execution.

**Endpoint**: `POST /workflows/{execution_id}/retry`

**Response** (200 OK):
```json
{
  "execution_id": "660f9500-f39c-52e5-b827-557766551111",
  "status": "pending",
  "queue_position": 1,
  "retried_from": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Models

Manage ComfyUI models including loading, downloading, and cleanup.

### List Models

Get a list of all available models with their current status.

**Endpoint**: `GET /models/`

**Query Parameters**:
- `model_type` (optional): Filter by model type (`checkpoint`, `lora`, `embedding`, `vae`, `controlnet`, `upscaler`)
- `available_only` (optional): Show only available models (boolean, default: false)

**Response** (200 OK):
```json
{
  "models": [
    {
      "name": "v1-5-pruned-emaonly.ckpt",
      "type": "checkpoint",
      "is_loaded": true,
      "is_downloading": false,
      "download_progress": null,
      "last_used": "2024-01-01T00:01:00Z",
      "memory_usage_mb": 3800.5
    },
    {
      "name": "detail_tweaker_lora.safetensors",
      "type": "lora", 
      "is_loaded": false,
      "is_downloading": true,
      "download_progress": 0.75,
      "last_used": null,
      "memory_usage_mb": null
    }
  ],
  "total_memory_usage_mb": 3800.5,
  "available_memory_mb": 20480.0
}
```

### Get Model Status

Get detailed status for a specific model.

**Endpoint**: `GET /models/{model_name}`

**Response** (200 OK):
```json
{
  "name": "v1-5-pruned-emaonly.ckpt",
  "type": "checkpoint",
  "is_loaded": true,
  "is_downloading": false,
  "download_progress": null,
  "last_used": "2024-01-01T00:01:00Z",
  "memory_usage_mb": 3800.5,
  "file_size_mb": 4265.8,
  "file_path": "/opt/ComfyUI/models/checkpoints/v1-5-pruned-emaonly.ckpt",
  "hash": "cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516",
  "metadata": {
    "resolution": "512x512",
    "base_model": "sd-1.5",
    "architecture": "stable-diffusion-v1"
  }
}
```

### Download Model

Download a model from a URL.

**Endpoint**: `POST /models/{model_name}/download`

**Query Parameters**:
- `model_type` (required): Model type (`checkpoint`, `lora`, etc.)
- `download_url` (required): URL to download the model from
- `description` (optional): Model description

**Response** (200 OK):
```json
{
  "message": "Model download started",
  "model_name": "new-model.safetensors",
  "model_type": "checkpoint"
}
```

### Load Model

Load a model into memory for use.

**Endpoint**: `POST /models/{model_name}/load`

**Query Parameters**:
- `model_type` (required): Model type

**Response** (200 OK):
```json
{
  "message": "Model loaded successfully", 
  "model_name": "v1-5-pruned-emaonly.ckpt",
  "model_type": "checkpoint"
}
```

### Unload Model

Unload a model from memory to free up space.

**Endpoint**: `POST /models/{model_name}/unload`

**Response** (200 OK):
```json
{
  "message": "Model unloaded successfully",
  "model_name": "v1-5-pruned-emaonly.ckpt"
}
```

### Cleanup Unused Models

Clean up models that haven't been used recently to free memory.

**Endpoint**: `POST /models/cleanup`

**Query Parameters**:
- `max_age_hours` (optional): Maximum age in hours (1-24, default: 1)

**Response** (200 OK):
```json
{
  "message": "Cleanup completed",
  "models_unloaded": [
    "old-model-1.ckpt",
    "unused-lora.safetensors"
  ]
}
```

### Get Download Progress

Get download progress for a model being downloaded.

**Endpoint**: `GET /models/{model_name}/download-progress`

**Response** (200 OK):
```json
{
  "model_name": "new-model.safetensors",
  "is_downloading": true,
  "download_progress": 0.75,
  "is_available": false
}
```

## Files

Upload and manage files for use in workflows.

### Upload File

Upload a file for use in workflows.

**Endpoint**: `POST /files/upload`

**Content-Type**: `multipart/form-data`

**Form Data**:
- `file`: The file to upload (binary)
- `description` (optional): File description

**Response** (200 OK):
```json
{
  "file_id": "file_abc123",
  "filename": "input_image.png",
  "size": 1048576,
  "content_type": "image/png",
  "upload_url": null
}
```

### Download File

Download a file by its ID.

**Endpoint**: `GET /files/{file_id}`

**Response** (200 OK):
- Content-Type: Based on the original file
- Body: Binary file content

### Get File Info

Get information and metadata about a file.

**Endpoint**: `GET /files/{file_id}/info`

**Response** (200 OK):
```json
{
  "file_id": "file_abc123",
  "filename": "input_image.png",
  "size": 1048576,
  "content_type": "image/png",
  "created_at": "2024-01-01T00:00:00Z",
  "expires_at": "2024-01-08T00:00:00Z",
  "download_url": "https://storage.example.com/files/file_abc123"
}
```

## Health & Monitoring

Monitor API and system health.

### Basic Health Check

Basic health check endpoint.

**Endpoint**: `GET /health/`

**Authentication**: Not required

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z", 
  "version": "1.0.0",
  "uptime_seconds": 86400.5
}
```

### Detailed Health Check

Detailed health check with service statuses.

**Endpoint**: `GET /health/detailed`

**Authentication**: Not required

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0", 
  "uptime_seconds": 86400.5,
  "services": [
    {
      "name": "database",
      "status": "healthy",
      "response_time_ms": 5.2,
      "error": null
    },
    {
      "name": "redis",
      "status": "healthy", 
      "response_time_ms": 2.1,
      "error": null
    },
    {
      "name": "comfyui",
      "status": "healthy",
      "response_time_ms": 15.8,
      "error": null
    }
  ],
  "system": {
    "cpu_usage_percent": 25.6,
    "memory_usage_percent": 45.2,
    "gpu_usage_percent": 80.1,
    "disk_usage_percent": 60.5
  }
}
```

### Readiness Probe

Kubernetes readiness probe endpoint.

**Endpoint**: `GET /health/readiness`

**Response** (200 OK):
```json
{
  "status": "ready"
}
```

**Response** (503 Service Unavailable):
```json
{
  "status": "not ready",
  "reason": "Database not available"
}
```

### Liveness Probe

Kubernetes liveness probe endpoint.

**Endpoint**: `GET /health/liveness`

**Response** (200 OK):
```json
{
  "status": "alive",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### System Metrics

Get detailed system performance metrics.

**Endpoint**: `GET /metrics/`

**Response** (200 OK):
```json
{
  "cpu_usage_percent": 25.6,
  "memory_usage_percent": 45.2,
  "gpu_usage_percent": 80.1,
  "gpu_memory_usage_percent": 65.3,
  "disk_usage_percent": 60.5,
  "active_executions": 3,
  "queue_size": 5,
  "total_executions": 1247,
  "average_execution_time_seconds": 145.2
}
```

## Error Handling

All API errors follow a consistent format:

```json
{
  "error": "error_type",
  "message": "Human-readable error message", 
  "details": [
    {
      "code": "VALIDATION_ERROR",
      "message": "Field 'workflow.nodes' is required",
      "field": "workflow.nodes"
    }
  ],
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_abc123"
}
```

### Common Error Codes

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | `validation_error` | Invalid request data or parameters |
| 401 | `unauthorized` | Authentication required or invalid token |
| 403 | `forbidden` | Access denied for the requested resource |
| 404 | `not_found` | Requested resource not found |
| 413 | `payload_too_large` | Request body or file too large |
| 429 | `rate_limit_exceeded` | Too many requests, rate limit exceeded |
| 500 | `internal_error` | Internal server error |
| 503 | `service_unavailable` | Service temporarily unavailable |

### Validation Errors

Validation errors include detailed information about which fields are invalid:

```json
{
  "error": "validation_error",
  "message": "Invalid request data",
  "details": [
    {
      "code": "REQUIRED_FIELD",
      "message": "Field 'workflow' is required",
      "field": "workflow"
    },
    {
      "code": "INVALID_VALUE",
      "message": "Priority must be one of: low, normal, high",
      "field": "priority"
    }
  ],
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Per Minute**: 60 requests
- **Per Hour**: 1000 requests
- **Per Day**: 10,000 requests

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1704067200
```

When rate limits are exceeded, the API returns a `429 Too Many Requests` response:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Rate limit: 1000 requests per hour.",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Rate Limit Recommendations

- Implement exponential backoff when receiving 429 responses
- Cache results when possible to reduce API calls
- Use batch operations where available
- Monitor rate limit headers to avoid hitting limits
- Contact support for higher rate limits if needed

## Webhook Notifications

For long-running workflows, you can provide a webhook URL to receive notifications when the workflow completes.

### Webhook Request

When a workflow completes, the API will send a POST request to your webhook URL:

```json
{
  "event": "workflow.completed",
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "timestamp": "2024-01-01T00:03:00Z",
  "workflow_result": {
    "execution_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "outputs": {...},
    "duration_seconds": 120.5
  }
}
```

### Webhook Security

- Webhooks include a signature header for verification
- Use HTTPS URLs for webhook endpoints
- Implement idempotency handling for retry scenarios
- Return a 2xx status code to acknowledge receipt

### Webhook Events

| Event | Description |
|-------|-------------|
| `workflow.started` | Workflow execution started |
| `workflow.completed` | Workflow completed successfully |
| `workflow.failed` | Workflow execution failed |
| `workflow.cancelled` | Workflow was cancelled |