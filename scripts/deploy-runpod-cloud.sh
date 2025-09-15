#!/usr/bin/env bash

set -euo pipefail

echo "🚀 Deploy ComfyUI to RunPod using Cloud Build"

# Check if runpod CLI is installed
if ! command -v runpod >/dev/null 2>&1; then
  echo "❌ RunPod CLI not found. Installing..."
  pip install runpod
fi

# Configuration
PROJECT_NAME="comfyui-serverless"
TEMPLATE_NAME="${PROJECT_NAME}-template"

# Create RunPod template
echo "📋 Creating RunPod template..."
cat > runpod-template.json <<EOF
{
  "name": "${TEMPLATE_NAME}",
  "imageName": "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
  "containerDiskInGb": 50,
  "volumeInGb": 100,
  "volumeMountPath": "/workspace",
  "ports": "8000/http,8080/http",
  "env": [
    {
      "key": "PYTHONPATH",
      "value": "/workspace/src"
    },
    {
      "key": "PYTHONDONTWRITEBYTECODE", 
      "value": "1"
    },
    {
      "key": "PYTHONUNBUFFERED",
      "value": "1"
    }
  ],
  "startScript": "#!/bin/bash\ncd /workspace\n# Install system dependencies\napt-get update && apt-get install -y gcc g++ git curl libpq-dev redis-server\n# Install Python dependencies\npip install --upgrade pip\npip install -r requirements.txt\npip install runpod\n# Start the application\npython runpod_handler.py"
}
EOF

echo "🔧 Template created. Next steps:"
echo "1) Upload your code to RunPod storage or GitHub"
echo "2) Use the RunPod web interface to:"
echo "   - Create a new template"
echo "   - Set the container image to: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
echo "   - Add the start script from runpod-template.json"
echo "   - Configure environment variables"
echo "   - Set ports: 8000/http, 8080/http"
echo "3) Deploy as serverless endpoint"

echo ""
echo "📖 Manual setup guide:"
echo "1. Go to https://runpod.io/console/serverless"
echo "2. Create New -> Template"
echo "3. Use the configuration from runpod-template.json"
echo "4. Create endpoint from template"

rm -f runpod-template.json
echo "✅ Setup instructions generated!"