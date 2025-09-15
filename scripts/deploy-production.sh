#!/bin/bash

# ComfyUI Serverless API - Production Deployment Script
# Für Anfänger - Schritt für Schritt Production Setup

set -e

echo "🚀 ComfyUI Serverless API - Production Deployment"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "src/requirements.txt" ]; then
    print_error "Bitte führe dieses Script im Hauptverzeichnis des Projekts aus"
    exit 1
fi

print_status "Prüfe System-Voraussetzungen..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker ist nicht installiert. Bitte installiere Docker Desktop first."
    echo "https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose ist nicht verfügbar."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 ist nicht installiert."
    exit 1
fi

print_success "Alle Voraussetzungen erfüllt!"

# Create production environment file
print_status "Erstelle Production Environment..."

if [ ! -f ".env.production" ]; then
    print_status "Generiere sichere Konfiguration..."
    
    # Generate secure secret key
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    
    cat > .env.production << EOF
# Production Configuration
# NEVER commit this file to git!

# API Settings
API_TITLE=ComfyUI Serverless API
API_VERSION=1.0.0
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Security - CRITICALLY IMPORTANT
SECRET_KEY=${SECRET_KEY}
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database - Use real PostgreSQL in production
DATABASE_URL=postgresql://comfyui_user:CHANGE_THIS_PASSWORD@localhost/comfyui_production

# Redis - Use real Redis in production
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Storage - Use real S3 or MinIO in production
STORAGE_TYPE=s3
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
AWS_S3_BUCKET=comfyui-production
AWS_S3_REGION=us-east-1
S3_ENDPOINT=https://your-s3-endpoint.com

# ComfyUI - Point to real ComfyUI instance
COMFYUI_PATH=/opt/ComfyUI
COMFYUI_MODELS_PATH=/opt/ComfyUI/models
COMFYUI_OUTPUT_PATH=/opt/ComfyUI/output
COMFYUI_API_URL=https://your-comfyui-server.com

# GPU Settings
GPU_MEMORY_FRACTION=0.8
MAX_GPU_MEMORY_GB=24

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000

# Webhook Settings
WEBHOOK_TIMEOUT_SECONDS=30
WEBHOOK_MAX_RETRIES=3
EOF

    print_success "Production Konfiguration erstellt: .env.production"
    print_warning "WICHTIG: Bearbeite .env.production und ändere alle Passwörter und URLs!"
    
else
    print_warning ".env.production existiert bereits"
fi

# Create production docker-compose
print_status "Erstelle Production Docker Compose..."

cat > docker-compose.production.yml << 'EOF'
version: '3.8'

services:
  # Production PostgreSQL
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: comfyui_production
      POSTGRES_USER: comfyui_user
      POSTGRES_PASSWORD: CHANGE_THIS_PASSWORD
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U comfyui_user -d comfyui_production"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Production Redis
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass YOUR_REDIS_PASSWORD
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Production MinIO (S3-compatible storage)
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: YOUR_MINIO_ACCESS_KEY
      MINIO_ROOT_PASSWORD: YOUR_MINIO_SECRET_KEY
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  # API Server
  api:
    build:
      context: .
      dockerfile: src/infrastructure/docker/Dockerfile
    env_file:
      - .env.production
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Celery Worker
  worker:
    build:
      context: .
      dockerfile: src/infrastructure/docker/Dockerfile
    command: celery -A src.services.workflow.celery_app worker --loglevel=info
    env_file:
      - .env.production
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  minio_data:
EOF

print_success "Production Docker Compose erstellt"

# Create nginx configuration
print_status "Erstelle Nginx Konfiguration..."

mkdir -p nginx

cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security Headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

        # API Proxy
        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://api/health;
            access_log off;
        }

        # File upload size
        client_max_body_size 100M;
    }
}
EOF

print_success "Nginx Konfiguration erstellt"

# Create deployment script
print_status "Erstelle Deployment Scripts..."

cat > scripts/start-production.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting ComfyUI Serverless API in Production Mode..."

# Check if .env.production exists
if [ ! -f ".env.production" ]; then
    echo "❌ .env.production not found. Run deploy-production.sh first!"
    exit 1
fi

# Start all services
docker-compose -f docker-compose.production.yml up -d

echo "✅ Production deployment started!"
echo ""
echo "🌐 Services running on:"
echo "  - API: https://your-domain.com"
echo "  - API Docs: https://your-domain.com/docs"
echo "  - MinIO Console: http://localhost:9001"
echo ""
echo "📊 Monitor with:"
echo "  - docker-compose -f docker-compose.production.yml logs -f"
echo "  - docker-compose -f docker-compose.production.yml ps"
EOF

chmod +x scripts/start-production.sh

cat > scripts/stop-production.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping Production Services..."

docker-compose -f docker-compose.production.yml down

echo "✅ Production services stopped"
EOF

chmod +x scripts/stop-production.sh

cat > scripts/backup-production.sh << 'EOF'
#!/bin/bash

echo "💾 Creating Production Backup..."

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup database
docker exec $(docker-compose -f docker-compose.production.yml ps -q postgres) \
    pg_dump -U comfyui_user comfyui_production > $BACKUP_DIR/database.sql

# Backup MinIO data
docker run --rm \
    -v $(docker-compose -f docker-compose.production.yml ps -q minio):/data \
    -v $(pwd)/$BACKUP_DIR:/backup \
    busybox tar czf /backup/minio_data.tar.gz -C /data .

# Backup configuration
cp .env.production $BACKUP_DIR/
cp docker-compose.production.yml $BACKUP_DIR/

echo "✅ Backup created in $BACKUP_DIR"
EOF

chmod +x scripts/backup-production.sh

print_success "Deployment Scripts erstellt"

# Create monitoring setup
print_status "Erstelle Monitoring Setup..."

cat > scripts/monitor-production.sh << 'EOF'
#!/bin/bash

echo "📊 ComfyUI Serverless API - Production Monitoring"
echo "================================================"

echo ""
echo "🐳 Docker Services Status:"
docker-compose -f docker-compose.production.yml ps

echo ""
echo "📈 Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo ""
echo "🌐 Service Health Checks:"
curl -s http://localhost:8000/health | jq '.' 2>/dev/null || echo "API not responding"
curl -s http://localhost:9000/minio/health/live >/dev/null && echo "✅ MinIO healthy" || echo "❌ MinIO unhealthy"

echo ""
echo "📋 Recent Logs (last 20 lines):"
docker-compose -f docker-compose.production.yml logs --tail=20
EOF

chmod +x scripts/monitor-production.sh

print_success "Monitoring Setup erstellt"

# Final instructions
echo ""
echo "🎉 Production Deployment Setup abgeschlossen!"
echo "============================================="
echo ""
echo "📝 Nächste Schritte:"
echo "1. Bearbeite .env.production und ändere ALLE Passwörter"
echo "2. Konfiguriere deine Domain in nginx/nginx.conf"
echo "3. SSL-Zertifikate in nginx/ssl/ platzieren"
echo "4. Production starten: ./scripts/start-production.sh"
echo ""
echo "🔧 Wichtige Dateien:"
echo "  - .env.production (Hauptkonfiguration)"
echo "  - docker-compose.production.yml (Services)"
echo "  - nginx/nginx.conf (Reverse Proxy)"
echo ""
echo "📊 Monitoring & Management:"
echo "  - Start: ./scripts/start-production.sh"
echo "  - Stop: ./scripts/stop-production.sh"
echo "  - Monitor: ./scripts/monitor-production.sh"
echo "  - Backup: ./scripts/backup-production.sh"
echo ""
echo "🚨 WICHTIG: Ändere alle Standard-Passwörter vor dem Start!"
echo "🔒 WICHTIG: Konfiguriere SSL-Zertifikate für HTTPS!"
echo ""
print_success "Setup abgeschlossen. Bereit für Production Deployment!"