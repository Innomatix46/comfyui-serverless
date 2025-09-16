#!/usr/bin/env python3
"""Test client for RunPod deployed ComfyUI API"""
import requests
import json
import time
import os
from typing import Dict, Any, Optional

class ComfyUIRunPodClient:
    def __init__(self, endpoint_id: str, runpod_api_key: str, comfyui_api_key: str):
        self.endpoint_id = endpoint_id
        self.runpod_api_key = runpod_api_key
        self.comfyui_api_key = comfyui_api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        print("🏥 Health Check...")
        return self._request("/api/v1/health", "GET")
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        print("📋 Listing Models...")
        return self._request("/api/v1/models", "GET")
    
    def execute_workflow(self, workflow_definition: Dict[str, Any], priority: str = "medium", webhook_url: Optional[str] = None) -> Dict[str, Any]:
        """Execute a ComfyUI workflow"""
        print(f"🚀 Executing Workflow (Priority: {priority})...")
        
        payload = {
            "workflow_definition": workflow_definition,
            "priority": priority
        }
        
        if webhook_url:
            payload["webhook_url"] = webhook_url
            
        return self._request("/api/v1/workflows", "POST", payload)
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status"""
        print(f"📊 Checking Workflow Status: {workflow_id}")
        return self._request(f"/api/v1/workflows/{workflow_id}/status", "GET")
    
    def get_workflow_result(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution result"""
        print(f"📄 Getting Workflow Result: {workflow_id}")
        return self._request(f"/api/v1/workflows/{workflow_id}/result", "GET")
    
    def upload_file(self, file_path: str, file_type: str = "image") -> Dict[str, Any]:
        """Upload a file to the API"""
        print(f"📤 Uploading File: {file_path}")
        
        # Read file and encode as base64
        import base64
        with open(file_path, "rb") as f:
            file_content = base64.b64encode(f.read()).decode()
        
        payload = {
            "file_content": file_content,
            "file_name": os.path.basename(file_path),
            "file_type": file_type
        }
        
        return self._request("/api/v1/files/upload", "POST", payload)
    
    def _request(self, endpoint: str, method: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the RunPod endpoint"""
        data = {
            "input": {
                "endpoint": endpoint,
                "method": method,
                "api_key": self.comfyui_api_key
            }
        }
        
        if payload:
            data["input"]["payload"] = payload
        
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=300)
            result = response.json()
            
            # Pretty print the result
            print(f"📋 Response Status: {result.get('status_code', 'unknown')}")
            if result.get('data'):
                print(f"📄 Response: {json.dumps(result['data'], indent=2)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"error": str(e)}

def create_test_workflow() -> Dict[str, Any]:
    """Create a simple test workflow"""
    return {
        "nodes": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode", 
                "inputs": {
                    "text": "a beautiful sunset over mountains, highly detailed, 8k",
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "blurry, low quality, distorted",
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["5", 0]
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["4", 0],
                    "vae": ["1", 2]
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "ComfyUI_runpod_test"
                }
            }
        }
    }

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 ComfyUI RunPod Test Client")
    print("=" * 60)
    
    # Configuration - SET THESE VALUES!
    ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "YOUR_ENDPOINT_ID_HERE")
    RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY_HERE") 
    COMFYUI_API_KEY = os.getenv("COMFYUI_API_KEY", "YOUR_COMFYUI_API_KEY_HERE")
    
    if "YOUR_" in f"{ENDPOINT_ID}{RUNPOD_API_KEY}{COMFYUI_API_KEY}":
        print("❌ Please set environment variables:")
        print("export RUNPOD_ENDPOINT_ID=your_endpoint_id")
        print("export RUNPOD_API_KEY=your_runpod_api_key") 
        print("export COMFYUI_API_KEY=your_comfyui_api_key")
        return
    
    # Create client
    client = ComfyUIRunPodClient(ENDPOINT_ID, RUNPOD_API_KEY, COMFYUI_API_KEY)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check...")
    health_result = client.health_check()
    
    if health_result.get("status_code") != 200:
        print("❌ Health check failed, stopping tests")
        return
    
    # Test 2: List Models
    print("\n2️⃣ Testing Model List...")
    models_result = client.list_models()
    
    # Test 3: Simple Workflow
    print("\n3️⃣ Testing Workflow Execution...")
    workflow = create_test_workflow()
    workflow_result = client.execute_workflow(workflow, priority="high")
    
    # Extract workflow ID if successful
    if workflow_result.get("status_code") == 200:
        workflow_data = workflow_result.get("data", {})
        workflow_id = workflow_data.get("workflow_id") or workflow_data.get("id")
        
        if workflow_id:
            print(f"✅ Workflow started with ID: {workflow_id}")
            
            # Test 4: Check Status
            print("\n4️⃣ Checking Workflow Status...")
            time.sleep(2)  # Wait a bit
            status_result = client.get_workflow_status(workflow_id)
            
            # Test 5: Get Result (may not be ready yet)
            print("\n5️⃣ Attempting to Get Result...")
            result = client.get_workflow_result(workflow_id)
        else:
            print("⚠️ No workflow ID returned")
    else:
        print("❌ Workflow execution failed")
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()