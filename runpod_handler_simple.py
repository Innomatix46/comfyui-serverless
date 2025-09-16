"""Simple RunPod Handler without FastAPI dependency"""
import runpod
import json
import os

def handler(event):
    """
    Simple handler that processes requests directly
    """
    print(f"Received event: {json.dumps(event, indent=2)}")
    
    # Extract input
    input_data = event.get("input", {})
    endpoint = input_data.get("endpoint", "/health")
    method = input_data.get("method", "GET")
    api_key = input_data.get("api_key", "")
    payload = input_data.get("payload", {})
    
    # Check API key
    expected_key = os.getenv("COMFYUI_API_KEY", "eBt6dOjqEVpAG1mD7dMiI4qhk1ISlNOyW-GPj8Fh61M")
    if api_key != expected_key:
        return {
            "status_code": 401,
            "data": {"error": "Invalid API key"}
        }
    
    # Route handling
    if endpoint == "/api/v1/health" or endpoint == "/health":
        return {
            "status_code": 200,
            "data": {
                "status": "healthy",
                "message": "ComfyUI Serverless API is running (Simple Handler)",
                "version": "1.0.0",
                "handler": "simple"
            }
        }
    
    elif endpoint == "/api/v1/models" and method == "GET":
        return {
            "status_code": 200,
            "data": {
                "models": [],
                "count": 0,
                "message": "No models installed yet"
            }
        }
    
    elif endpoint == "/api/v1/workflows" and method == "POST":
        # Simple workflow response
        workflow_def = payload.get("workflow_definition", {})
        priority = payload.get("priority", "medium")
        
        return {
            "status_code": 200,
            "data": {
                "workflow_id": "test-workflow-123",
                "status": "queued",
                "priority": priority,
                "message": "Workflow queued (test mode)",
                "nodes": len(workflow_def.get("nodes", {}))
            }
        }
    
    else:
        return {
            "status_code": 404,
            "data": {
                "error": f"Endpoint {endpoint} not found",
                "available_endpoints": [
                    "/api/v1/health",
                    "/api/v1/models",
                    "/api/v1/workflows"
                ]
            }
        }

# Start RunPod serverless worker
print("Starting RunPod Simple Handler...")
runpod.serverless.start({"handler": handler})