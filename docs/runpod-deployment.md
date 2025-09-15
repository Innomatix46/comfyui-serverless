# 🚀 RunPod Deployment Guide - ComfyUI Serverless API

## Schritt-für-Schritt Anleitung für RunPod Deployment

### 1. RunPod Vorbereitung

#### Account Setup
```bash
# 1. Erstelle RunPod Account: https://runpod.io
# 2. Gehe zu "Serverless" Tab
# 3. Klicke "Create Endpoint"
```

#### Docker Image vorbereiten
```bash
# Option A: Verwende fertiges Image (empfohlen)
# Das Image wird automatisch von deinem Code erstellt

# Option B: Eigenes Image bauen
docker build -f Dockerfile.runpod -t dein-username/comfyui-serverless .
docker push dein-username/comfyui-serverless
```

### 2. RunPod Konfiguration

#### Environment Variables für RunPod
```bash
# In RunPod Serverless Settings:
DATABASE_URL=sqlite:///./comfyui_serverless.db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=runpod-production-secret-key
API_TITLE=ComfyUI Serverless API - RunPod
API_VERSION=1.0.0
DEBUG=false
```

### 3. Deployment Schritte

#### Schritt 1: Code vorbereiten
```bash
# Stelle sicher alle Files sind da:
ls -la
# Du solltest sehen:
# - src/
# - Dockerfile.runpod
# - runpod_handler.py  
# - docs/
```

#### Schritt 2: Docker Image bauen
```bash
# Lokal testen
docker build -f Dockerfile.runpod -t comfyui-serverless-runpod .
docker run -p 8000:8000 comfyui-serverless-runpod

# Oder direkt auf RunPod deployen (siehe nächster Schritt)
```

#### Schritt 3: RunPod Endpoint erstellen

**A) Via RunPod Web Interface:**
1. Gehe zu https://runpod.io → Serverless
2. Klicke "Create Endpoint"
3. Wähle "Custom Image"
4. Docker Image: `dein-username/comfyui-serverless` oder verwende GitHub
5. Setze Environment Variables
6. Wähle GPU: A100 oder H100 (empfohlen für ComfyUI)
7. Klicke "Create Endpoint"

**B) Via GitHub Integration (empfohlen):**
1. Push deinen Code zu GitHub
2. In RunPod wähle "GitHub" als Source
3. Repository: `dein-username/comfyui-serverless`
4. Dockerfile: `Dockerfile.runpod`
5. Start Command: `python runpod_handler.py`

### 4. API Nutzung auf RunPod

#### Beispiel 1: Health Check
```bash
curl -X POST https://api.runpod.ai/v2/DEINE-ENDPOINT-ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer DEIN-RUNPOD-API-KEY" \
  -d '{
    "input": {
      "endpoint": "/api/v1/health",
      "method": "GET",
      "api_key": "***REMOVED***"
    }
  }'
```

#### Beispiel 2: ComfyUI Workflow ausführen
```bash
curl -X POST https://api.runpod.ai/v2/DEINE-ENDPOINT-ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer DEIN-RUNPOD-API-KEY" \
  -d '{
    "input": {
      "endpoint": "/api/v1/workflows",
      "method": "POST", 
      "api_key": "***REMOVED***",
      "payload": {
        "workflow_definition": {
          "nodes": {
            "1": {
              "class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
            },
            "2": {
              "class_type": "CLIPTextEncode", 
              "inputs": {
                "text": "a beautiful landscape, photorealistic, 8k",
                "clip": ["1", 1]
              }
            },
            "3": {
              "class_type": "KSampler",
              "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0], 
                "steps": 20,
                "cfg": 7.0,
                "seed": 42
              }
            }
          }
        },
        "priority": "high"
      }
    }
  }'
```

#### Beispiel 3: Asynchron mit Webhook
```bash
curl -X POST https://api.runpod.ai/v2/DEINE-ENDPOINT-ID/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer DEIN-RUNPOD-API-KEY" \
  -d '{
    "input": {
      "endpoint": "/api/v1/workflows",
      "method": "POST",
      "api_key": "***REMOVED***", 
      "payload": {
        "workflow_definition": {...},
        "webhook_url": "https://deine-domain.com/webhook"
      }
    }
  }'
```

### 5. Python Client für RunPod

```python
import requests
import json

class ComfyUIRunPodClient:
    def __init__(self, endpoint_id: str, api_key: str, comfyui_api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key 
        self.comfyui_api_key = comfyui_api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
    
    def execute_workflow(self, workflow_definition: dict, async_mode: bool = True):
        """Execute ComfyUI workflow on RunPod"""
        endpoint = "/run" if async_mode else "/runsync"
        
        payload = {
            "input": {
                "endpoint": "/api/v1/workflows",
                "method": "POST",
                "api_key": self.comfyui_api_key,
                "payload": {
                    "workflow_definition": workflow_definition,
                    "priority": "high"
                }
            }
        }
        
        response = requests.post(
            f"{self.base_url}{endpoint}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json=payload
        )
        
        return response.json()
    
    def get_result(self, job_id: str):
        """Get result from async job"""
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()

# Nutzung:
client = ComfyUIRunPodClient(
    endpoint_id="DEINE-ENDPOINT-ID",
    api_key="DEIN-RUNPOD-API-KEY", 
    comfyui_api_key="***REMOVED***"
)

# Workflow definieren
workflow = {
    "nodes": {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
        },
        # ... mehr nodes
    }
}

# Ausführen
result = client.execute_workflow(workflow, async_mode=False)
print(result)
```

### 6. Monitoring & Debugging

#### Logs anschauen
```bash
# In RunPod Console:
# Gehe zu "Logs" tab in deinem Endpoint
# Oder verwende RunPod CLI:
runpod logs DEINE-ENDPOINT-ID
```

#### Testing lokal
```bash
# Teste RunPod Handler lokal:
python runpod_handler.py

# In anderem Terminal:
curl -X POST http://localhost:8000/test \
  -H "Content-Type: application/json" \
  -d '{"endpoint": "/api/v1/health", "method": "GET"}'
```

### 7. Kosten Optimierung

#### GPU Auswahl:
- **A100 (80GB)**: Beste Performance für große Modelle (~$1.50/hr)
- **A4000 (16GB)**: Gut für Standard SDXL (~$0.50/hr) 
- **RTX 3090 (24GB)**: Budget Option (~$0.30/hr)

#### Auto-Scaling:
```yaml
# In RunPod Settings:
Min Workers: 0    # Kostet nichts wenn nicht verwendet
Max Workers: 10   # Skaliert automatisch bei Load
Idle Timeout: 60s # Stoppt Container nach 60s ohne Requests
```

### 8. Troubleshooting

#### Häufige Probleme:

**Problem: Container startet nicht**
```bash
# Lösung: Dockerfile prüfen
docker build -f Dockerfile.runpod -t test .
docker run test  # Lokal testen
```

**Problem: "Module not found"**
```bash
# Lösung: PYTHONPATH in Dockerfile setzen
ENV PYTHONPATH=/workspace/src
```

**Problem: "Redis connection refused"**
```bash
# Lösung: Redis im Handler starten
os.system("redis-server --daemonize yes")
```

**Problem: Timeout bei großen Workflows**
```bash
# Lösung: Timeout in RunPod Settings erhöhen
# Oder async Mode verwenden mit Webhooks
```

### 9. Weitere Features

#### A) Auto-Model Download
```python
# In runpod_handler.py hinzufügen:
def download_models():
    """Download ComfyUI models on first run"""
    models_dir = "/workspace/models"
    if not os.path.exists(f"{models_dir}/checkpoints"):
        os.makedirs(f"{models_dir}/checkpoints")
        # Download SDXL base model
        os.system("wget -O /workspace/models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors")
```

#### B) Webhook Integration
```python
# Webhook Handler für Callbacks
@app.post("/webhook")
async def webhook_handler(request: Request):
    data = await request.json()
    # Process completed workflow result
    return {"status": "received"}
```

### 10. Production Checklist

- [ ] Docker Image gebaut und getestet
- [ ] RunPod Endpoint erstellt  
- [ ] Environment Variables gesetzt
- [ ] API Keys konfiguriert
- [ ] Health Check funktioniert
- [ ] Workflow Execution getestet
- [ ] Monitoring/Logging aktiviert
- [ ] Auto-Scaling konfiguriert
- [ ] Webhook Callbacks implementiert
- [ ] Kosten Limits gesetzt

---

## 🚀 Los geht's!

1. **Erstelle Dockerfile.runpod** ✅ (bereits erstellt)
2. **Erstelle runpod_handler.py** ✅ (bereits erstellt)  
3. **Push Code zu GitHub**
4. **Erstelle RunPod Endpoint**
5. **Teste deine erste ComfyUI Workflow**

**Deine API ist jetzt bereit für RunPod! 🎉**
