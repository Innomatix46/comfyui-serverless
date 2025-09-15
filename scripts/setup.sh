#!/bin/bash

# ComfyUI Serverless API Setup Script
set -e

echo "🚀 Setting up ComfyUI Serverless API..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

print_success "Python version check passed: $python_version"

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing Python dependencies..."
pip install -r src/requirements.txt
print_success "Dependencies installed"

# Install development dependencies if available
if [ -f "requirements-dev.txt" ]; then
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
    print_success "Development dependencies installed"
fi

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating environment configuration file..."
    cat > .env << EOF
# API Configuration
API_TITLE=ComfyUI Serverless API
API_VERSION=1.0.0
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database
DATABASE_URL=postgresql://comfyui_user:password@localhost/comfyui_serverless

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Storage
STORAGE_TYPE=local
STORAGE_BASE_PATH=/tmp/comfyui
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_S3_BUCKET=
AWS_S3_REGION=us-east-1

# ComfyUI
COMFYUI_PATH=/opt/ComfyUI
COMFYUI_MODELS_PATH=/opt/ComfyUI/models
COMFYUI_OUTPUT_PATH=/opt/ComfyUI/output
COMFYUI_API_URL=http://localhost:8188

# GPU
GPU_MEMORY_FRACTION=0.8
MAX_GPU_MEMORY_GB=24

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
EOF
    print_success "Environment file created (.env)"
    print_warning "Please update the database URL and other configuration in .env file"
else
    print_warning "Environment file already exists (.env)"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs data temp models output
print_success "Directories created"

# Set up Alembic if not already done
if [ ! -d "src/database/migrations/versions" ]; then
    print_status "Setting up database migrations..."
    cd src/database
    alembic init migrations
    cd ../..
    print_success "Database migrations initialized"
else
    print_warning "Database migrations already initialized"
fi

# Check if PostgreSQL is available
print_status "Checking PostgreSQL connection..."
if command -v psql > /dev/null; then
    if pg_isready -h localhost > /dev/null 2>&1; then
        print_success "PostgreSQL is running"
        
        # Create database if it doesn't exist
        print_status "Creating database if it doesn't exist..."
        createdb comfyui_serverless 2>/dev/null || print_warning "Database may already exist"
        
        # Run migrations
        print_status "Running database migrations..."
        cd src/database
        alembic upgrade head
        cd ../..
        print_success "Database migrations completed"
    else
        print_warning "PostgreSQL is not running. Please start PostgreSQL and run migrations manually."
    fi
else
    print_warning "PostgreSQL client not found. Please install PostgreSQL."
fi

# Check if Redis is available
print_status "Checking Redis connection..."
if command -v redis-cli > /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        print_success "Redis is running"
    else
        print_warning "Redis is not running. Please start Redis server."
    fi
else
    print_warning "Redis client not found. Please install Redis."
fi

# Create systemd service files (optional)
if [ -d "/etc/systemd/system" ] && [ "$EUID" -eq 0 ]; then
    print_status "Creating systemd service files..."
    
    cat > /etc/systemd/system/comfyui-api.service << EOF
[Unit]
Description=ComfyUI Serverless API
After=network.target postgresql.service redis.service

[Service]
Type=forking
User=www-data
Group=www-data
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    print_success "Systemd service files created"
    print_status "To start the service: sudo systemctl start comfyui-api"
    print_status "To enable on boot: sudo systemctl enable comfyui-api"
fi

# Final setup instructions
echo ""
echo "=========================================="
print_success "Setup completed successfully!"
echo "=========================================="
echo ""
print_status "Next steps:"
echo "1. Update configuration in .env file"
echo "2. Start PostgreSQL and Redis services"
echo "3. Run database migrations: cd src/database && alembic upgrade head"
echo "4. Start the API server: python -m uvicorn src.api.main:app --reload"
echo "5. Start Celery worker: celery -A src.services.workflow.celery_app worker --loglevel=info"
echo ""
print_status "API Documentation will be available at: http://localhost:8000/docs"
print_status "Health Check: http://localhost:8000/health"
echo ""
print_warning "Don't forget to:"
echo "- Change the SECRET_KEY in .env"
echo "- Set up proper database credentials"
echo "- Configure AWS credentials for S3 storage (if using S3)"
echo "- Install and configure ComfyUI"
echo ""
print_success "Happy coding! 🎉"