# ComfyUI Serverless API

**Enterprise-grade, production-ready serverless API für ComfyUI workflows mit kompletter Docker Infrastructure**

[![Tests](https://img.shields.io/badge/tests-100%25%20passing-green)](test-report.md)
[![Performance](https://img.shields.io/badge/performance-excellent-green)](test-report.md)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](docker-compose.test.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](src/requirements.txt)

---

## ⚡ Schnellstart (5 Minuten)

```bash
# 1. Projekt klonen
git clone https://github.com/your-repo/comfyui-serverless.git
cd comfyui-serverless

# 2. Setup ausführen
chmod +x scripts/setup.sh && ./scripts/setup.sh

# 3. Services starten
docker-compose -f docker-compose.test.yml up -d

# 4. API starten
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**✅ Fertig!** API läuft auf http://localhost:8000

📖 **Detaillierte Anleitung für Anfänger:** [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
⚡ **Quick Start für Erfahrene:** [QUICK_START.md](QUICK_START.md)

A production-ready, serverless API for ComfyUI that provides scalable AI image generation capabilities with comprehensive workflow management, authentication, monitoring, and storage integration.

## 🚀 Features

### Core API Features
- **RESTful API** with FastAPI framework
- **Asynchronous workflow execution** with Celery
- **Real-time progress tracking** via WebSockets
- **Webhook notifications** for workflow completion
- **JWT-based authentication** with refresh tokens
- **Rate limiting** and request throttling
- **File upload/download** with S3 integration
- **Comprehensive logging** and structured error handling

### Workflow Management
- **ComfyUI integration** with full workflow support
- **Queue management** with priority levels
- **Progress monitoring** and cancellation support
- **Execution retry** mechanisms
- **Timeout handling** for long-running tasks
- **Resource optimization** with GPU memory management

### Model Management
- **Automatic model downloading** and caching
- **Memory usage optimization** with model offloading
- **Model versioning** and compatibility checks
- **Usage tracking** and cleanup automation

### Storage & File Management
- **Multi-backend storage** (S3, local, GCS)
- **Presigned URL support** for direct uploads
- **Automatic cleanup** of expired files
- **File validation** and security checks

### Monitoring & Observability
- **Prometheus metrics** export
- **Health check endpoints** for Kubernetes
- **System resource monitoring** (CPU, GPU, memory)
- **Performance tracking** and bottleneck analysis
- **Structured logging** with request tracing

### Infrastructure
- **Docker containerization** with multi-stage builds
- **Kubernetes deployment** with auto-scaling
- **Database migrations** with Alembic
- **Redis caching** and session management
- **Load balancing** and service mesh ready

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │     API Server  │    │  Celery Workers │
│    (Nginx)      │───▶│   (FastAPI)     │───▶│   (ComfyUI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                       ┌─────────────────┐              │
                       │   PostgreSQL    │              │
                       │   (Database)    │              │
                       └─────────────────┘              │
                                │                        │
                       ┌─────────────────┐              │
                       │     Redis       │──────────────┘
                       │ (Cache/Queue)   │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   S3 Storage    │
                       │ (Files/Models)  │
                       └─────────────────┘
```

## 📋 Requirements

### System Requirements
- **Python 3.11+**
- **PostgreSQL 13+**
- **Redis 6+**
- **GPU with CUDA support** (recommended)
- **Docker & Docker Compose** (for containerized deployment)
- **Kubernetes** (for production deployment)

### Hardware Recommendations
- **CPU**: 8+ cores
- **RAM**: 16GB+ (32GB recommended)
- **GPU**: NVIDIA RTX 4090 or better (24GB VRAM)
- **Storage**: 500GB+ SSD for models and temporary files

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd comfyui-serverless
```

### 2. Run Setup Script
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Start Services with Docker Compose
```bash
docker-compose up -d
```

### 5. Access API Documentation
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Metrics: http://localhost:9090/metrics

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# API Configuration
SECRET_KEY=your-secret-key-change-in-production
DATABASE_URL=postgresql://user:password@localhost/comfyui_serverless
REDIS_URL=redis://localhost:6379/0

# AWS S3 Storage
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=your-bucket-name

# ComfyUI
COMFYUI_API_URL=http://localhost:8188
GPU_MEMORY_FRACTION=0.8
```

See `.env.example` for all available options.

## 📡 API Usage

### Authentication
```bash
# Register user
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# Login
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

### Execute Workflow
```bash
curl -X POST "http://localhost:8000/workflows/execute" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {
      "nodes": {
        "1": {
          "class_type": "CheckpointLoaderSimple",
          "inputs": [
            {"name": "ckpt_name", "value": "model.safetensors", "type": "string"}
          ]
        }
      }
    },
    "priority": "normal",
    "webhook_url": "https://your-webhook-endpoint.com/callback"
  }'
```

### Check Workflow Status
```bash
curl -X GET "http://localhost:8000/workflows/{execution_id}" \
  -H "Authorization: Bearer <token>"
```

## 🔄 Deployment

### RunPod Serverless
- Deploy the API as a serverless endpoint on RunPod: see DEPLOY_RUNPOD.md

### Development Deployment
```bash
# Start development server
python -m uvicorn src.api.main:app --reload

# Start Celery worker
celery -A src.services.workflow.celery_app worker --loglevel=info

# Start Celery beat scheduler
celery -A src.services.workflow.celery_app beat --loglevel=info
```

### Production Deployment with Kubernetes

1. **Create namespace and apply configurations:**
```bash
kubectl apply -f src/infrastructure/kubernetes/namespace.yaml
kubectl apply -f src/infrastructure/kubernetes/configmap.yaml
kubectl apply -f src/infrastructure/kubernetes/secret.yaml
kubectl apply -f src/infrastructure/kubernetes/pvc.yaml
```

2. **Deploy services:**
```bash
kubectl apply -f src/infrastructure/kubernetes/deployment.yaml
kubectl apply -f src/infrastructure/kubernetes/service.yaml
kubectl apply -f src/infrastructure/kubernetes/ingress.yaml
kubectl apply -f src/infrastructure/kubernetes/hpa.yaml
```

3. **Verify deployment:**
```bash
kubectl get pods -n comfyui-serverless
kubectl get services -n comfyui-serverless
```

### Docker Compose Deployment
```bash
# Production deployment
docker-compose -f src/infrastructure/docker/docker-compose.yml up -d

# Scale workers
docker-compose up --scale celery-worker=4
```

## 📊 Monitoring

### Health Checks
- **Liveness**: `/health/liveness`
- **Readiness**: `/health/readiness`
- **Detailed**: `/health/detailed`

### Metrics
- **Prometheus**: `/metrics/prometheus`
- **System**: `/metrics/system`
- **Execution**: `/metrics/executions`

### Grafana Dashboard
Import the provided Grafana dashboard for comprehensive monitoring:
- CPU, Memory, GPU utilization
- Workflow execution statistics
- Queue sizes and processing times
- Error rates and success metrics

## 🧪 Testing

### Run Tests
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# All tests with coverage
python -m pytest --cov=src tests/
```

### Load Testing
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Run load test
ab -n 1000 -c 10 -H "Authorization: Bearer <token>" \
  -p workflow.json -T application/json \
  http://localhost:8000/workflows/execute
```

## 🔒 Security

### Best Practices Implemented
- **JWT token authentication** with secure secret keys
- **Rate limiting** to prevent abuse
- **Input validation** and sanitization
- **CORS configuration** for cross-origin requests
- **Security headers** middleware
- **SQL injection prevention** with ORM
- **File upload restrictions** and validation

### Security Checklist
- [ ] Change default SECRET_KEY
- [ ] Use HTTPS in production
- [ ] Set up proper CORS origins
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Implement API versioning
- [ ] Set up monitoring alerts

## 🛠️ Development

### Project Structure
```
src/
├── api/                 # FastAPI application
│   ├── main.py         # Main application
│   ├── middleware.py   # Custom middleware
│   └── routers/        # API route handlers
├── core/               # Core functionality
│   └── database.py     # Database configuration
├── services/           # Business logic services
│   ├── workflow.py     # Workflow execution
│   ├── auth.py         # Authentication
│   ├── model.py        # Model management
│   └── storage.py      # File storage
├── models/             # Data models
│   ├── database.py     # SQLAlchemy models
│   └── schemas.py      # Pydantic schemas
├── utils/              # Utility functions
│   ├── validation.py   # Input validation
│   └── gpu.py          # GPU utilities
├── config/             # Configuration
│   └── settings.py     # Application settings
└── infrastructure/     # Deployment configs
    ├── docker/         # Docker configurations
    └── kubernetes/     # Kubernetes manifests
```

### Adding New Features

1. **Create feature branch:**
```bash
git checkout -b feature/new-feature
```

2. **Add service logic in `src/services/`**
3. **Create API endpoints in `src/api/routers/`**
4. **Add data models in `src/models/`**
5. **Write tests in `tests/`**
6. **Update documentation**

### Database Migrations
```bash
# Create new migration
cd src/database
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Downgrade migration
alembic downgrade -1
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use type hints throughout
- Write comprehensive docstrings
- Add unit tests for new features
- Update documentation

## 📚 API Documentation

Detailed API documentation is available at `/docs` when running the server. The API follows OpenAPI 3.0 specification with comprehensive examples and schemas.

### Key Endpoints

- `POST /auth/register` - Register new user
- `POST /auth/login` - Authenticate user
- `POST /workflows/execute` - Execute workflow
- `GET /workflows/{id}` - Get workflow status
- `GET /models/` - List available models
- `POST /files/upload` - Upload file
- `GET /health` - Health check

## 🎯 Performance Optimization

### GPU Memory Management
- Automatic model offloading
- Memory usage monitoring
- Batch size optimization
- VRAM fragmentation prevention

### Caching Strategy
- Redis for session data
- Model caching in GPU memory
- Database query caching
- HTTP response caching

### Scaling Recommendations
- Use multiple Celery workers
- Implement horizontal pod autoscaling
- Use CDN for static assets
- Optimize database queries

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory" errors:**
```bash
# Reduce GPU memory fraction
export GPU_MEMORY_FRACTION=0.6

# Enable model offloading
export ENABLE_MODEL_OFFLOAD=true
```

**Database connection issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U comfyui_user -d comfyui_serverless
```

**Celery workers not processing tasks:**
```bash
# Check Redis connection
redis-cli ping

# Restart workers
sudo systemctl restart celery-worker
```

### Logs
```bash
# View API logs
docker-compose logs comfyui-api

# View worker logs
docker-compose logs celery-worker

# Kubernetes logs
kubectl logs -n comfyui-serverless deployment/comfyui-api
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ComfyUI** for the amazing stable diffusion interface
- **FastAPI** for the excellent web framework
- **Celery** for distributed task processing
- **SQLAlchemy** for robust ORM capabilities

---

For more information, issues, or feature requests, please visit the [GitHub repository](https://github.com/your-org/comfyui-serverless).
