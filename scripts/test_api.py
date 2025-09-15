#!/usr/bin/env python3
"""Test the ComfyUI Serverless API."""

import requests
import json
import sys
from pathlib import Path

# Read API token
try:
    with open("api_token.txt", "r") as f:
        api_token = f.read().strip()
except FileNotFoundError:
    print("❌ API token not found. Run: python3 scripts/create_admin.py")
    sys.exit(1)

BASE_URL = "http://localhost:8050"
HEADERS = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

def test_endpoints():
    """Test various API endpoints."""
    
    print("🚀 Testing ComfyUI Serverless API\n")
    
    # 1. Health check (no auth required)
    print("1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # 2. System status
    print("2️⃣ Testing system status...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/system/status", headers=HEADERS, timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Database: {'✅ Connected' if data.get('database') == 'connected' else '❌ Error'}")
            print(f"   Redis: {'✅ Connected' if data.get('redis') == 'connected' else '❌ Error'}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # 3. List models
    print("3️⃣ Testing models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/models", headers=HEADERS, timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Models loaded: {data.get('loaded_models_count', 0)}")
            print(f"   Total models: {len(data.get('models', []))}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # 4. Test workflow submission (simple example)
    print("4️⃣ Testing workflow submission...")
    try:
        sample_workflow = {
            "nodes": {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": "test.png"
                    }
                }
            },
            "output_nodes": ["1"]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/workflows",
            headers=HEADERS,
            json={
                "workflow": sample_workflow,
                "priority": "normal"
            },
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code in [200, 201, 202]:
            data = response.json()
            execution_id = data.get('execution_id')
            print(f"   Workflow submitted: {execution_id}")
            
            # Check status
            if execution_id:
                status_response = requests.get(
                    f"{BASE_URL}/api/v1/workflows/{execution_id}",
                    headers=HEADERS,
                    timeout=5
                )
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Status: {status_data.get('status', 'unknown')}")
        else:
            print(f"   Response: {response.text}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # 5. API Documentation
    print("5️⃣ API Documentation available at:")
    print(f"   📚 Swagger UI: {BASE_URL}/docs")
    print(f"   📖 ReDoc: {BASE_URL}/redoc")
    print()
    
    print("✅ API testing complete!")
    print(f"🔗 Your API is running at: {BASE_URL}")


if __name__ == "__main__":
    test_endpoints()