# ✅ RunPod Deployment Checklist

## 🔒 Vorbereitung (Sicherheit)
- [x] Repository auf privat gesetzt
- [x] Git-History bereinigt (sensible Daten entfernt)
- [x] Secrets generiert (.env.production)
- [x] Hardcoded Credentials entfernt
- [ ] Secrets sicher gespeichert (Passwort-Manager)

## 🚀 RunPod Setup

### 1. Endpoint Erstellung
- [ ] RunPod Console geöffnet: https://www.runpod.io/console/serverless
- [ ] "New Endpoint" geklickt
- [ ] Endpoint Name: `comfyui-serverless-api`

### 2. Docker Konfiguration
```yaml
Docker Image: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
GPU Type: NVIDIA GeForce RTX 3090
vCPUs: 8
RAM: 32 GB
Container Disk: 20 GB
Volume Disk: 50 GB
```

### 3. Skalierung
```yaml
Min Workers: 0
Max Workers: 3
Idle Timeout: 60 seconds
Max Concurrency: 10
Scaler Type: Queue Delay
Scaler Value: 1
```

### 4. Environment Variables
Kopiere aus `.env.production`:
```bash
SECRET_KEY=<generated_secret>
COMFYUI_API_KEY=<generated_api_key>
ADMIN_PASSWORD=<generated_password>
POSTGRES_PASSWORD=<generated_password>
PYTHONPATH=/workspace/src
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
DATABASE_URL=sqlite:///./comfyui_serverless.db
COMFYUI_PATH=/workspace/comfyui
COMFYUI_MODELS_PATH=/workspace/comfyui/models
COMFYUI_OUTPUT_PATH=/workspace/comfyui/output
COMFYUI_TEMP_PATH=/workspace/comfyui/temp
DEBUG=false
HOST=0.0.0.0
PORT=8000
```

### 5. Container Start Commands
```bash
# System dependencies
apt-get update && apt-get install -y libpq-dev redis-server git curl

# Clone private repository (requires authentication setup)
cd /workspace
git clone https://github.com/Innomatix46/comfyui-serverless.git .

# Install dependencies  
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

## 🧪 Testing

### 1. Health Check
```bash
export RUNPOD_ENDPOINT_ID="your_endpoint_id"
export RUNPOD_API_KEY="your_runpod_api_key"
export COMFYUI_API_KEY="your_generated_api_key"

curl -X POST https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "endpoint": "/api/v1/health",
      "method": "GET",
      "api_key": "'$COMFYUI_API_KEY'"
    }
  }'
```

### 2. Test Client
```bash
python scripts/test_client.py
```

### 3. Erwartete Responses
- Health Check: `{"status": "healthy", "version": "1.0.0"}`
- Models: `{"models": [...], "count": 0}` (initial)
- Workflows: `{"workflow_id": "xxx", "status": "queued"}`

## 📊 Monitoring

### RunPod Console
- [ ] Endpoint Status: Running
- [ ] Workers: 0-3 (je nach Load)
- [ ] Logs: Keine Errors
- [ ] Metrics: Response Times < 5s

### Custom Monitoring
- [ ] Health Check Endpoint antwortet
- [ ] Workflow Execution funktioniert
- [ ] File Upload funktioniert
- [ ] Database Verbindung okay

## 🛠️ Troubleshooting

### Häufige Probleme:

1. **Cold Start Timeout**
   - Lösung: Idle Timeout auf 300s erhöhen
   - Oder: Min Workers auf 1 setzen

2. **Memory Issues**
   - Lösung: RTX 4090 verwenden (24GB)
   - Oder: Model Offloading aktivieren

3. **Git Clone Fails**
   - Problem: Private Repository
   - Lösung: GitHub Deploy Key oder Token verwenden

4. **Import Errors**
   - Problem: PYTHONPATH nicht gesetzt
   - Lösung: Environment Variable prüfen

5. **Database Errors**
   - Problem: Permissions
   - Lösung: SQLite verwenden (FILE-basiert)

## 🔧 Optimierungen

### Performance
- [ ] Model Caching aktiviert
- [ ] GPU Memory optimiert
- [ ] Redis für Caching
- [ ] Worker Scaling getestet

### Kosten
- [ ] Idle Timeout optimiert
- [ ] Min Workers = 0 (außer Production)
- [ ] Spot Instances (wenn verfügbar)
- [ ] Usage Monitoring aktiv

### Sicherheit
- [ ] Secrets rotiert
- [ ] Logs monitoring
- [ ] Rate Limiting konfiguriert
- [ ] API Key Scopes definiert

## ✅ Go-Live Checklist

- [ ] Alle Tests bestanden
- [ ] Monitoring eingerichtet  
- [ ] Backup-Strategie definiert
- [ ] Incident Response Plan
- [ ] Documentation aktualisiert
- [ ] Team informiert
- [ ] Rollback-Plan bereit

---

**🎯 Nächste Schritte nach Deployment:**
1. Load Testing durchführen
2. Model Installation (Stable Diffusion, etc.)
3. Custom Workflows entwickeln
4. CI/CD Pipeline einrichten
5. Production Monitoring