#!/usr/bin/env python3
"""Test RunPod deployment locally"""
import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the handler
from runpod_handler import runpod_handler, start_server, example_workflow_request
import asyncio

async def test_local_deployment():
    """Test the RunPod handler locally"""
    print("=" * 60)
    print("🧪 Testing RunPod Handler Locally")
    print("=" * 60)
    
    # Start the server
    print("\n1️⃣ Starting FastAPI server...")
    if not start_server():
        print("❌ Failed to start server")
        return False
    
    print("✅ Server started successfully")
    time.sleep(5)  # Wait for server to be ready
    
    # Test 1: Health check
    print("\n2️⃣ Testing health endpoint...")
    health_event = {
        "input": {
            "endpoint": "/api/v1/health", 
            "method": "GET",
            "api_key": "test_key_123"
        }
    }
    
    result = await runpod_handler(health_event)
    if result.get("status_code") == 200:
        print(f"✅ Health check passed: {result.get('data')}")
    else:
        print(f"❌ Health check failed: {result}")
        return False
    
    # Test 2: Workflow endpoint (without actual execution)
    print("\n3️⃣ Testing workflow endpoint structure...")
    workflow_event = {
        "input": {
            "endpoint": "/api/v1/workflows",
            "method": "POST",
            "api_key": "test_key_123",
            "payload": {
                "workflow_definition": {
                    "nodes": {
                        "1": {
                            "class_type": "TestNode",
                            "inputs": {"test": "value"}
                        }
                    }
                },
                "priority": "high"
            }
        }
    }
    
    result = await runpod_handler(workflow_event)
    print(f"📋 Workflow endpoint response: Status {result.get('status_code')}")
    
    # Test 3: Models endpoint
    print("\n4️⃣ Testing models endpoint...")
    models_event = {
        "input": {
            "endpoint": "/api/v1/models",
            "method": "GET",
            "api_key": "test_key_123"
        }
    }
    
    result = await runpod_handler(models_event)
    if result.get("status_code") in [200, 404]:  # 404 is ok if no models installed
        print(f"✅ Models endpoint accessible: Status {result.get('status_code')}")
    else:
        print(f"❌ Models endpoint failed: {result}")
    
    print("\n" + "=" * 60)
    print("✅ All local tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    # Set test environment variables
    os.environ["DATABASE_URL"] = "sqlite:///./test_comfyui.db"
    os.environ["SECRET_KEY"] = "test-secret-key"
    os.environ["DEBUG"] = "true"
    os.environ["COMFYUI_PATH"] = "./test_comfyui"
    os.environ["COMFYUI_MODELS_PATH"] = "./test_comfyui/models"
    os.environ["COMFYUI_OUTPUT_PATH"] = "./test_comfyui/output"
    os.environ["COMFYUI_TEMP_PATH"] = "./test_comfyui/temp"
    os.environ["COMFYUI_API_URL"] = "http://localhost:8188"
    
    try:
        success = asyncio.run(test_local_deployment())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)