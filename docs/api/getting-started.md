# Getting Started Guide

This guide will help you get started with the ComfyUI Serverless API quickly and efficiently.

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication Setup](#authentication-setup)
- [Your First Workflow](#your-first-workflow)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Create an Account

First, register for an API account:

```bash
curl -X POST https://api.comfyui-serverless.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "password": "your-secure-password",
    "username": "your-username"
  }'
```

### 2. Get Your Access Token

Login to get your authentication tokens:

```bash
curl -X POST https://api.comfyui-serverless.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "password": "your-secure-password"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 3. Set Your Token

Export your access token for easy use:

```bash
export COMFYUI_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### 4. Test the API

Verify your setup with a health check:

```bash
curl -H "Authorization: Bearer $COMFYUI_TOKEN" \
  https://api.comfyui-serverless.com/health/
```

## Authentication Setup

### Using Environment Variables

Create a `.env` file for your credentials:

```bash
# .env file
COMFYUI_API_URL=https://api.comfyui-serverless.com
COMFYUI_EMAIL=your-email@example.com
COMFYUI_PASSWORD=your-secure-password
COMFYUI_TOKEN=your-access-token
```

### Token Management

Access tokens expire after 30 minutes. Use the refresh token to get a new access token:

```bash
curl -X POST $COMFYUI_API_URL/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "your-refresh-token"
  }'
```

### Automatic Token Refresh

Here's a bash script for automatic token management:

```bash
#!/bin/bash
# auth.sh - Automatic token management

API_URL="https://api.comfyui-serverless.com"
EMAIL="your-email@example.com"
PASSWORD="your-secure-password"

# Function to get new tokens
get_tokens() {
    response=$(curl -s -X POST "$API_URL/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")
    
    access_token=$(echo $response | jq -r '.access_token')
    refresh_token=$(echo $response | jq -r '.refresh_token')
    
    echo "ACCESS_TOKEN=$access_token" > .tokens
    echo "REFRESH_TOKEN=$refresh_token" >> .tokens
    
    export ACCESS_TOKEN=$access_token
    export REFRESH_TOKEN=$refresh_token
}

# Function to refresh tokens
refresh_tokens() {
    if [ -f ".tokens" ]; then
        source .tokens
        response=$(curl -s -X POST "$API_URL/auth/refresh" \
            -H "Content-Type: application/json" \
            -d "{\"refresh_token\":\"$REFRESH_TOKEN\"}")
        
        if echo $response | jq -e '.access_token' > /dev/null; then
            access_token=$(echo $response | jq -r '.access_token')
            refresh_token=$(echo $response | jq -r '.refresh_token')
            
            echo "ACCESS_TOKEN=$access_token" > .tokens
            echo "REFRESH_TOKEN=$refresh_token" >> .tokens
            
            export ACCESS_TOKEN=$access_token
            export REFRESH_TOKEN=$refresh_token
        else
            echo "Failed to refresh token, getting new ones..."
            get_tokens
        fi
    else
        get_tokens
    fi
}

# Check if we have valid tokens, refresh if needed
refresh_tokens
echo "Tokens ready: ACCESS_TOKEN is set"
```

## Your First Workflow

### Simple Text-to-Image Generation

Let's create a basic text-to-image workflow:

```bash
curl -X POST $COMFYUI_API_URL/workflows/execute \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
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
              "value": "a beautiful landscape, mountains, lake, sunset, masterpiece, high quality",
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
          "class_type": "CLIPTextEncode", 
          "inputs": [
            {
              "name": "text",
              "type": "STRING",
              "value": "blurry, low quality, distorted",
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
        "4": {
          "id": "4",
          "class_type": "EmptyLatentImage",
          "inputs": [
            {
              "name": "width",
              "type": "INT",
              "value": 512,
              "required": true
            },
            {
              "name": "height", 
              "type": "INT",
              "value": 512,
              "required": true
            },
            {
              "name": "batch_size",
              "type": "INT",
              "value": 1,
              "required": true
            }
          ]
        },
        "5": {
          "id": "5",
          "class_type": "KSampler",
          "inputs": [
            {
              "name": "seed",
              "type": "INT", 
              "value": 42,
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
              "name": "denoise",
              "type": "FLOAT",
              "value": 1.0,
              "required": true
            },
            {
              "name": "model",
              "type": "MODEL",
              "value": ["1", 0],
              "required": true
            },
            {
              "name": "positive",
              "type": "CONDITIONING",
              "value": ["2", 0],
              "required": true
            },
            {
              "name": "negative",
              "type": "CONDITIONING", 
              "value": ["3", 0],
              "required": true
            },
            {
              "name": "latent_image",
              "type": "LATENT",
              "value": ["4", 0],
              "required": true
            }
          ]
        },
        "6": {
          "id": "6",
          "class_type": "VAEDecode",
          "inputs": [
            {
              "name": "samples",
              "type": "LATENT",
              "value": ["5", 0],
              "required": true
            },
            {
              "name": "vae",
              "type": "VAE",
              "value": ["1", 2], 
              "required": true
            }
          ]
        },
        "7": {
          "id": "7",
          "class_type": "SaveImage", 
          "inputs": [
            {
              "name": "images",
              "type": "IMAGE",
              "value": ["6", 0],
              "required": true
            },
            {
              "name": "filename_prefix",
              "type": "STRING",
              "value": "ComfyUI",
              "required": false
            }
          ]
        }
      },
      "metadata": {
        "description": "Basic text-to-image generation",
        "version": "1.0"
      }
    },
    "priority": "normal",
    "timeout_minutes": 30
  }'
```

Response:
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2024-01-01T00:00:00Z",
  "estimated_duration": 120,
  "queue_position": 1
}
```

### Check Workflow Status

Monitor your workflow's progress:

```bash
execution_id="550e8400-e29b-41d4-a716-446655440000"

curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  $COMFYUI_API_URL/workflows/$execution_id/status
```

### Get the Results

Once completed, retrieve your generated images:

```bash
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  $COMFYUI_API_URL/workflows/$execution_id
```

## Common Patterns

### 1. Workflow with Image Input

Upload an image and use it in your workflow:

```bash
# First, upload an image
curl -X POST $COMFYUI_API_URL/files/upload \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@input_image.jpg" \
  -F "description=Input image for processing"

# Response includes file_id
# {
#   "file_id": "file_abc123",
#   "filename": "input_image.jpg",
#   ...
# }

# Then use it in a workflow
curl -X POST $COMFYUI_API_URL/workflows/execute \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {
      "nodes": {
        "1": {
          "id": "1",
          "class_type": "LoadImage",
          "inputs": [
            {
              "name": "image",
              "type": "STRING",
              "value": "file_abc123",
              "required": true
            }
          ]
        }
      }
    }
  }'
```

### 2. Batch Processing

Process multiple images with different parameters:

```bash
#!/bin/bash
# batch_process.sh

prompts=(
    "a beautiful sunset over mountains"
    "a futuristic city skyline" 
    "a serene forest lake"
    "abstract geometric patterns"
)

seeds=(42 123 456 789)

for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    seed="${seeds[$i]}"
    
    echo "Processing: $prompt (seed: $seed)"
    
    response=$(curl -s -X POST $COMFYUI_API_URL/workflows/execute \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"workflow\": {
                \"nodes\": {
                    \"2\": {
                        \"inputs\": [{
                            \"name\": \"text\",
                            \"value\": \"$prompt\"
                        }]
                    },
                    \"5\": {
                        \"inputs\": [{
                            \"name\": \"seed\",
                            \"value\": $seed
                        }]
                    }
                }
            },
            \"metadata\": {
                \"batch_id\": \"batch_$(date +%s)\",
                \"prompt_index\": $i
            }
        }")
    
    execution_id=$(echo $response | jq -r '.execution_id')
    echo "Submitted: $execution_id"
    
    # Optional: wait between submissions
    sleep 5
done
```

### 3. Webhook Integration

Set up a webhook to receive notifications:

```bash
# Your webhook endpoint should handle POST requests
# Example webhook handler (pseudo-code):
# 
# POST /webhook
# {
#   "event": "workflow.completed",
#   "execution_id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "completed",
#   "workflow_result": {...}
# }

curl -X POST $COMFYUI_API_URL/workflows/execute \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {...},
    "webhook_url": "https://your-app.com/webhook",
    "priority": "high"
  }'
```

### 4. Model Management

Ensure required models are loaded:

```bash
# Check if model is loaded
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  $COMFYUI_API_URL/models/v1-5-pruned-emaonly.ckpt

# Load model if not loaded
curl -X POST $COMFYUI_API_URL/models/v1-5-pruned-emaonly.ckpt/load \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "model_type=checkpoint"

# Download new model
curl -X POST "$COMFYUI_API_URL/models/new-model.safetensors/download" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "model_type=checkpoint&download_url=https://example.com/model.safetensors"
```

## Best Practices

### 1. Error Handling

Always implement proper error handling:

```bash
#!/bin/bash
# robust_workflow.sh

submit_workflow() {
    local workflow_json="$1"
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        response=$(curl -s -w "%{http_code}" -X POST $COMFYUI_API_URL/workflows/execute \
            -H "Authorization: Bearer $ACCESS_TOKEN" \
            -H "Content-Type: application/json" \
            -d "$workflow_json")
        
        http_code="${response: -3}"
        body="${response%???}"
        
        case $http_code in
            200)
                echo "Success: $(echo $body | jq -r '.execution_id')"
                return 0
                ;;
            401)
                echo "Token expired, refreshing..."
                refresh_tokens
                ;;
            429)
                echo "Rate limited, waiting..."
                sleep 60
                ;;
            *)
                echo "Error $http_code: $body"
                ;;
        esac
        
        ((retry_count++))
        sleep $((retry_count * 5))  # Exponential backoff
    done
    
    echo "Failed after $max_retries attempts"
    return 1
}
```

### 2. Monitoring Workflows

Monitor long-running workflows:

```bash
#!/bin/bash
# monitor_workflow.sh

monitor_workflow() {
    local execution_id="$1"
    local timeout_seconds="${2:-1800}"  # 30 minutes default
    local start_time=$(date +%s)
    
    echo "Monitoring workflow: $execution_id"
    
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout_seconds ]; then
            echo "Timeout reached, cancelling workflow..."
            curl -X POST $COMFYUI_API_URL/workflows/$execution_id/cancel \
                -H "Authorization: Bearer $ACCESS_TOKEN"
            return 1
        fi
        
        response=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
            $COMFYUI_API_URL/workflows/$execution_id/status)
        
        status=$(echo $response | jq -r '.status')
        progress=$(echo $response | jq -r '.progress.percentage // 0')
        
        echo "Status: $status, Progress: $progress%"
        
        case $status in
            "completed")
                echo "Workflow completed successfully!"
                return 0
                ;;
            "failed")
                echo "Workflow failed:"
                curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
                    $COMFYUI_API_URL/workflows/$execution_id | jq '.error'
                return 1
                ;;
            "cancelled")
                echo "Workflow was cancelled"
                return 1
                ;;
        esac
        
        sleep 10
    done
}
```

### 3. Resource Management

Optimize resource usage:

```bash
# Clean up old models periodically
curl -X POST $COMFYUI_API_URL/models/cleanup \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "max_age_hours=2"

# Check system resources before heavy workloads
response=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
    $COMFYUI_API_URL/metrics/)

gpu_usage=$(echo $response | jq -r '.gpu_usage_percent')
memory_usage=$(echo $response | jq -r '.gpu_memory_usage_percent')

if (( $(echo "$gpu_usage > 90" | bc -l) )); then
    echo "GPU usage high ($gpu_usage%), waiting..."
    sleep 30
fi
```

### 4. Workflow Templates

Create reusable workflow templates:

```bash
# workflow_templates.sh

# Text-to-image template
generate_txt2img_workflow() {
    local prompt="$1"
    local negative_prompt="${2:-blurry, low quality}"
    local seed="${3:-42}"
    local steps="${4:-20}"
    local cfg="${5:-7.0}"
    local width="${6:-512}"
    local height="${7:-512}"
    
    cat << EOF
{
  "workflow": {
    "nodes": {
      "1": {
        "id": "1",
        "class_type": "CheckpointLoaderSimple",
        "inputs": [{"name": "ckpt_name", "type": "STRING", "value": "v1-5-pruned-emaonly.ckpt", "required": true}]
      },
      "2": {
        "id": "2", 
        "class_type": "CLIPTextEncode",
        "inputs": [
          {"name": "text", "type": "STRING", "value": "$prompt", "required": true},
          {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": true}
        ]
      },
      "3": {
        "id": "3",
        "class_type": "CLIPTextEncode", 
        "inputs": [
          {"name": "text", "type": "STRING", "value": "$negative_prompt", "required": true},
          {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": true}
        ]
      },
      "4": {
        "id": "4",
        "class_type": "EmptyLatentImage",
        "inputs": [
          {"name": "width", "type": "INT", "value": $width, "required": true},
          {"name": "height", "type": "INT", "value": $height, "required": true},
          {"name": "batch_size", "type": "INT", "value": 1, "required": true}
        ]
      },
      "5": {
        "id": "5",
        "class_type": "KSampler", 
        "inputs": [
          {"name": "seed", "type": "INT", "value": $seed, "required": true},
          {"name": "steps", "type": "INT", "value": $steps, "required": true},
          {"name": "cfg", "type": "FLOAT", "value": $cfg, "required": true},
          {"name": "sampler_name", "type": "STRING", "value": "euler", "required": true},
          {"name": "scheduler", "type": "STRING", "value": "normal", "required": true},
          {"name": "denoise", "type": "FLOAT", "value": 1.0, "required": true},
          {"name": "model", "type": "MODEL", "value": ["1", 0], "required": true},
          {"name": "positive", "type": "CONDITIONING", "value": ["2", 0], "required": true},
          {"name": "negative", "type": "CONDITIONING", "value": ["3", 0], "required": true},
          {"name": "latent_image", "type": "LATENT", "value": ["4", 0], "required": true}
        ]
      },
      "6": {
        "id": "6",
        "class_type": "VAEDecode",
        "inputs": [
          {"name": "samples", "type": "LATENT", "value": ["5", 0], "required": true},
          {"name": "vae", "type": "VAE", "value": ["1", 2], "required": true}
        ]
      },
      "7": {
        "id": "7",
        "class_type": "SaveImage",
        "inputs": [
          {"name": "images", "type": "IMAGE", "value": ["6", 0], "required": true},
          {"name": "filename_prefix", "type": "STRING", "value": "generated", "required": false}
        ]
      }
    }
  },
  "priority": "normal",
  "timeout_minutes": 30
}
EOF
}

# Usage
workflow_json=$(generate_txt2img_workflow "a beautiful sunset" "blurry" 123 25 8.0 768 768)
echo "$workflow_json" | curl -X POST $COMFYUI_API_URL/workflows/execute \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d @-
```

## Troubleshooting

### Common Issues

1. **Authentication Errors (401)**
   ```bash
   # Check token expiration
   echo $ACCESS_TOKEN | cut -d'.' -f2 | base64 -d | jq '.exp'
   
   # Refresh token
   refresh_tokens
   ```

2. **Rate Limiting (429)**
   ```bash
   # Check rate limit headers
   curl -I -H "Authorization: Bearer $ACCESS_TOKEN" \
     $COMFYUI_API_URL/health/
   
   # Implement backoff
   sleep 60
   ```

3. **Workflow Validation Errors (400)**
   ```bash
   # Validate workflow JSON
   echo "$workflow_json" | jq '.'
   
   # Check required fields
   echo "$workflow_json" | jq '.workflow.nodes | keys[]'
   ```

4. **Model Not Found**
   ```bash
   # List available models
   curl -H "Authorization: Bearer $ACCESS_TOKEN" \
     $COMFYUI_API_URL/models/
   
   # Load required model
   curl -X POST $COMFYUI_API_URL/models/model-name/load \
     -H "Authorization: Bearer $ACCESS_TOKEN"
   ```

### Debug Mode

Enable debug logging:

```bash
# Set debug environment
export DEBUG=1
export COMFYUI_LOG_LEVEL=DEBUG

# Get detailed logs
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  "$COMFYUI_API_URL/workflows/$execution_id/logs?level=DEBUG&tail=1000"
```

### Health Checks

Monitor API health:

```bash
#!/bin/bash
# health_monitor.sh

check_api_health() {
    response=$(curl -s $COMFYUI_API_URL/health/detailed)
    status=$(echo $response | jq -r '.status')
    
    if [ "$status" != "healthy" ]; then
        echo "API unhealthy: $status"
        echo $response | jq '.services[] | select(.status != "healthy")'
        return 1
    fi
    
    echo "API healthy"
    return 0
}

# Run health check every minute
while true; do
    if ! check_api_health; then
        echo "Waiting for API to recover..."
        sleep 60
    else
        sleep 60
    fi
done
```

This getting started guide provides a comprehensive foundation for working with the ComfyUI Serverless API. Start with the simple examples and gradually move to more complex workflows as you become familiar with the system.