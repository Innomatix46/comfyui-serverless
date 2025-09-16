#!/bin/bash
# RunPod Startup Script

echo "🚀 Starting ComfyUI Serverless API..."

# Update environment
export PYTHONPATH=/workspace/src
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Install system dependencies
apt-get update && apt-get install -y libpq-dev redis-server git curl

# Clone repository (public read access)
cd /workspace
if [ ! -d "comfyui-serverless" ]; then
    # Clone from public mirror or use wget for files
    echo "📦 Setting up application files..."
    mkdir -p comfyui-serverless
    cd comfyui-serverless
    
    # Download essential files directly
    curl -L https://raw.githubusercontent.com/Innomatix46/comfyui-serverless/main/runpod_handler.py -o runpod_handler.py
    curl -L https://raw.githubusercontent.com/Innomatix46/comfyui-serverless/main/src/requirements.txt -o requirements.txt
    
    # Alternative: If repo is public temporarily
    # git clone https://github.com/Innomatix46/comfyui-serverless.git .
else
    cd comfyui-serverless
fi

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || pip install fastapi uvicorn sqlalchemy redis pydantic runpod

# Create directories
mkdir -p logs temp outputs models comfyui/{models,output,temp}

# Start Redis
redis-server --daemonize yes

# Create minimal handler if main fails
if [ ! -f "runpod_handler.py" ]; then
    cat > runpod_handler.py << 'EOF'
import runpod

def handler(event):
    return {
        "status_code": 200,
        "data": {"status": "healthy", "message": "Basic handler running"}
    }

runpod.serverless.start({"handler": handler})
EOF
fi

# Start handler
echo "✅ Starting RunPod handler..."
python runpod_handler.py