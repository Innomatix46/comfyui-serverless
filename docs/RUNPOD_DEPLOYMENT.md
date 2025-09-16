# 🚀 RunPod Deployment Guide - ComfyUI Serverless API

## 📋 Schnell-Deployment

### 1. RunPod Console Setup

1. **Login**: [https://www.runpod.io/console/serverless](https://www.runpod.io/console/serverless)
2. **New Endpoint** klicken

### 2. Endpoint Konfiguration

```yaml
Endpoint Name: comfyui-serverless-api
Docker Image: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
GPU Type: NVIDIA GeForce RTX 3090 (empfohlen)
Min Workers: 0
Max Workers: 3
Idle Timeout: 60 seconds
Max Concurrency: 10
```

### 3. Environment Variables

```bash
PYTHONPATH=/workspace/src
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
SECRET_KEY=DEIN-SICHERER-SECRET-KEY
DATABASE_URL=sqlite:///./comfyui_serverless.db
COMFYUI_API_KEY=DEIN-NEUER-API-KEY
DEBUG=false
COMFYUI_PATH=/workspace/comfyui
COMFYUI_MODELS_PATH=/workspace/comfyui/models
COMFYUI_OUTPUT_PATH=/workspace/comfyui/output
COMFYUI_TEMP_PATH=/workspace/comfyui/temp
```

### 4. Container Start Commands

```bash
# System dependencies
apt-get update && apt-get install -y libpq-dev redis-server git curl

# Clone repository
cd /workspace
git clone https://github.com/Innomatix46/comfyui-serverless.git .

# Install Python dependencies
pip install --upgrade pip
pip install -r src/requirements.txt
pip install runpod

# Create directories
mkdir -p comfyui/{models,output,temp} logs

# Start Redis
redis-server --daemonize yes

# Start application
python runpod_handler.py
```

## 🧪 Test Deployment

### Health Check
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "endpoint": "/api/v1/health",
      "method": "GET",
      "api_key": "YOUR_COMFYUI_API_KEY"
    }
  }'
```

### Workflow Test
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "endpoint": "/api/v1/workflows",
      "method": "POST",
      "api_key": "YOUR_COMFYUI_API_KEY",
      "payload": {
        "workflow_definition": {
          "nodes": {
            "1": {"class_type": "TestNode", "inputs": {"test": "value"}}
          }
        },
        "priority": "high"
      }
    }
  }'
```

## ⚙️ Erweiterte Konfiguration

### GPU Optimierung
```bash
GPU_MEMORY_FRACTION=0.8
MAX_GPU_MEMORY_GB=24
ENABLE_MODEL_OFFLOAD=true
```

### Performance Tuning
- **RTX 4090**: Für große Modelle (24GB VRAM)
- **RTX 3090**: Standard (24GB VRAM)
- **A100**: Enterprise-Level (40GB/80GB VRAM)

### Skalierung
- **Min Workers**: 0 (kosteneffizient)
- **Max Workers**: 5 (Burst-Kapazität)
- **Idle Timeout**: 60-300s (Balance zwischen Kosten/Antwortzeit)

## 🔐 Sicherheit

1. **Secrets generieren:**
   ```bash
   # SECRET_KEY (64 Zeichen)
   openssl rand -hex 32
   
   # COMFYUI_API_KEY (32 Zeichen)  
   openssl rand -hex 16
   ```

2. **Niemals Secrets in Code hardcoden**
3. **Nur Environment Variables verwenden**

## 📊 Monitoring

- **RunPod Console**: Logs und Metriken
- **Custom Endpoints**: Health-Checks
- **Cost Tracking**: Worker-Usage überwachen

---
**✅ Deployment bereit!** Folge den Schritten und teste die Endpoints.