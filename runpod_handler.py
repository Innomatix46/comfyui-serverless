"""RunPod Serverless Handler für ComfyUI API"""
import runpod
import asyncio
import json
import os
import sys
import uvicorn
from threading import Thread
import time
import httpx
from typing import Dict, Any

# Add src to Python path
sys.path.append('/workspace/src')

# Import your FastAPI app
from src.api.main import app as fastapi_app

class RunPodHandler:
    def __init__(self):
        self.server = None
        self.server_thread = None
        self.base_url = "http://localhost:8000"
        
    def start_fastapi_server(self):
        """Start FastAPI server in background thread"""
        config = uvicorn.Config(
            fastapi_app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)
        server.run()
    
    def wait_for_server(self, timeout=30):
        """Wait for FastAPI server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(f"{self.base_url}/docs", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False

# Global handler instance
handler = RunPodHandler()

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting ComfyUI Serverless API...")
    
    # Start Redis server
    os.system("redis-server --daemonize yes")
    time.sleep(2)
    
    # Start FastAPI server in thread
    server_thread = Thread(target=handler.start_fastapi_server, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready
    if handler.wait_for_server():
        print("✅ FastAPI server is ready!")
        return True
    else:
        print("❌ Failed to start FastAPI server")
        return False

async def runpod_handler(event):
    """
    RunPod Serverless Handler
    
    Unterstützte Endpunkte:
    - /workflows - ComfyUI Workflow ausführen
    - /models - Modelle verwalten
    - /files - Dateien hochladen
    - /health - Gesundheitscheck
    """
    try:
        # Extract input data
        input_data = event.get("input", {})
        endpoint = input_data.get("endpoint", "/health")
        method = input_data.get("method", "GET").upper()
        headers = input_data.get("headers", {})
        payload = input_data.get("payload", {})
        
        # Add authentication if provided
        api_key = input_data.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Make request to FastAPI server
        url = f"{handler.base_url}{endpoint}"
        
        async with httpx.AsyncClient(timeout=300) as client:
            if method == "GET":
                response = await client.get(url, headers=headers, params=payload)
            elif method == "POST":
                response = await client.post(url, headers=headers, json=payload)
            elif method == "PUT":
                response = await client.put(url, headers=headers, json=payload)
            elif method == "DELETE":
                response = await client.delete(url, headers=headers)
            else:
                return {
                    "error": f"Unsupported method: {method}",
                    "status_code": 405
                }
        
        # Return response
        try:
            response_data = response.json()
        except:
            response_data = {"message": response.text}
        
        return {
            "status_code": response.status_code,
            "data": response_data,
            "headers": dict(response.headers)
        }
        
    except Exception as e:
        print(f"❌ Error in runpod_handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "status_code": 500
        }

# Beispiel für Workflow-Ausführung
def example_workflow_request():
    """Beispiel wie man einen ComfyUI Workflow ausführt"""
    return {
        "input": {
            "endpoint": "/api/v1/workflows",
            "method": "POST",
            "api_key": "YOUR_API_KEY_HERE",
            "payload": {
                "workflow_definition": {
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
                                "text": "a beautiful landscape",
                                "clip": ["1", 1]
                            }
                        },
                        "3": {
                            "class_type": "KSampler",
                            "inputs": {
                                "model": ["1", 0],
                                "positive": ["2", 0],
                                "steps": 20,
                                "cfg": 7.0
                            }
                        }
                    }
                },
                "priority": "high",
                "webhook_url": "https://your-webhook.com/callback"
            }
        }
    }

if __name__ == "__main__":
    # Start server when running locally
    if os.getenv("RUNPOD_ENDPOINT_ID"):
        # Running on RunPod
        print("🔥 Running on RunPod Serverless")
        # Start server
        start_server()
        # Start RunPod serverless
        runpod.serverless.start({"handler": runpod_handler})
    else:
        # Running locally for testing
        print("🧪 Running locally for testing")
        start_server()
        
        # Keep alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("👋 Shutting down...")