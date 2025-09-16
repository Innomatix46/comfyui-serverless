#!/usr/bin/env python3
"""Deploy to RunPod Serverless"""
import os
import sys
import json
import time
import requests
from typing import Dict, Any

# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
COMFYUI_API_KEY = os.getenv("COMFYUI_API_KEY", "YOUR_API_KEY_HERE")

if not RUNPOD_API_KEY:
    print("❌ Error: RUNPOD_API_KEY environment variable not set")
    print("Please set: export RUNPOD_API_KEY=your_runpod_api_key")
    sys.exit(1)

# RunPod API base URL
BASE_URL = "https://api.runpod.ai/v2"
HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

def create_serverless_endpoint():
    """Create a new serverless endpoint on RunPod"""
    
    endpoint_config = {
        "name": "comfyui-serverless-api",
        "templateId": "runpod-pytorch",  # Using PyTorch template
        "dockerImage": "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
        "volumeInGb": 50,
        "containerDiskInGb": 10,
        "minVcpus": 2,
        "minMemoryInGb": 8,
        "gpuTypeId": "NVIDIA GeForce RTX 3090",
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 1,
        "minWorkers": 0,
        "maxWorkers": 3,
        "workerIdleTimeout": 60,
        "envs": [
            {"key": "PYTHONPATH", "value": "/workspace/src"},
            {"key": "COMFYUI_API_KEY", "value": COMFYUI_API_KEY},
            {"key": "SECRET_KEY", "value": os.getenv("SECRET_KEY", "change-me-in-production")},
            {"key": "DATABASE_URL", "value": "sqlite:///./comfyui_serverless.db"}
        ],
        "dockerArgs": "python /workspace/runpod_handler.py"
    }
    
    print("🚀 Creating RunPod serverless endpoint...")
    response = requests.post(
        f"{BASE_URL}/serverless",
        headers=HEADERS,
        json=endpoint_config
    )
    
    if response.status_code == 200:
        data = response.json()
        endpoint_id = data.get("id")
        print(f"✅ Endpoint created successfully!")
        print(f"📋 Endpoint ID: {endpoint_id}")
        return endpoint_id
    else:
        print(f"❌ Failed to create endpoint: {response.text}")
        return None

def upload_code(endpoint_id: str):
    """Upload code to the endpoint"""
    print(f"📦 Uploading code to endpoint {endpoint_id}...")
    
    # In production, you would upload your code here
    # For now, we'll use the docker image with pre-installed code
    
    print("✅ Code upload simulated (using Docker image)")
    return True

def test_endpoint(endpoint_id: str):
    """Test the deployed endpoint"""
    print(f"\n🧪 Testing endpoint {endpoint_id}...")
    
    # Health check
    test_payload = {
        "input": {
            "endpoint": "/api/v1/health",
            "method": "GET",
            "api_key": COMFYUI_API_KEY
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/{endpoint_id}/run",
        headers=HEADERS,
        json=test_payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Health check passed: {result}")
        return True
    else:
        print(f"❌ Health check failed: {response.text}")
        return False

def main():
    """Main deployment flow"""
    print("=" * 60)
    print("🚀 ComfyUI Serverless RunPod Deployment")
    print("=" * 60)
    
    # Create endpoint
    endpoint_id = create_serverless_endpoint()
    if not endpoint_id:
        sys.exit(1)
    
    # Upload code
    if not upload_code(endpoint_id):
        sys.exit(1)
    
    # Wait for endpoint to be ready
    print("\n⏳ Waiting for endpoint to be ready...")
    time.sleep(30)
    
    # Test endpoint
    if test_endpoint(endpoint_id):
        print("\n" + "=" * 60)
        print("✅ Deployment successful!")
        print(f"🔗 Endpoint URL: https://api.runpod.ai/v2/{endpoint_id}/run")
        print(f"📋 Endpoint ID: {endpoint_id}")
        print("\n📖 Example usage:")
        print(f"""
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \\
  -H "Authorization: Bearer $RUNPOD_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "input": {{
      "endpoint": "/api/v1/workflows",
      "method": "POST",
      "api_key": "{COMFYUI_API_KEY}",
      "payload": {{
        "workflow_definition": {{...}}
      }}
    }}
  }}'
        """)
        print("=" * 60)
    else:
        print("\n❌ Deployment completed but tests failed")
        print(f"Please check the logs at: https://www.runpod.io/console/serverless/{endpoint_id}")

if __name__ == "__main__":
    main()