# SDK Examples and Code Samples

This document provides comprehensive code examples for integrating with the ComfyUI Serverless API across multiple programming languages and platforms.

## Table of Contents

- [Python SDK](#python-sdk)
- [JavaScript/Node.js SDK](#javascriptnodejs-sdk)
- [curl Examples](#curl-examples)
- [Integration Examples](#integration-examples)
- [Error Handling Patterns](#error-handling-patterns)
- [Best Practices](#best-practices)

## Python SDK

### Installation and Setup

```python
# requirements.txt
requests>=2.28.0
python-dotenv>=0.19.0
Pillow>=9.0.0
aiohttp>=3.8.0

# Install dependencies
# pip install -r requirements.txt
```

### Basic Python Client

```python
# comfyui_client.py
import os
import time
import json
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowResult:
    execution_id: str
    status: WorkflowStatus
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

class ComfyUIClient:
    def __init__(self, api_url: str = None, email: str = None, password: str = None):
        self.api_url = api_url or os.getenv("COMFYUI_API_URL", "https://api.comfyui-serverless.com")
        self.email = email or os.getenv("COMFYUI_EMAIL")
        self.password = password or os.getenv("COMFYUI_PASSWORD")
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make authenticated request to API."""
        self._ensure_authenticated()
        
        headers = kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {self.access_token}'
        kwargs['headers'] = headers
        
        url = f"{self.api_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        
        # Handle token expiration
        if response.status_code == 401:
            self._refresh_tokens()
            headers['Authorization'] = f'Bearer {self.access_token}'
            response = requests.request(method, url, **kwargs)
            
        return response
    
    def _ensure_authenticated(self):
        """Ensure we have a valid access token."""
        if not self.access_token or self._token_expired():
            if self.refresh_token and not self._refresh_token_expired():
                self._refresh_tokens()
            else:
                self._login()
    
    def _token_expired(self) -> bool:
        """Check if access token is expired."""
        if not self.token_expires:
            return True
        return datetime.now() >= self.token_expires
    
    def _refresh_token_expired(self) -> bool:
        """Check if refresh token is expired (7 days)."""
        # Implement refresh token expiration logic
        return False
    
    def _login(self):
        """Authenticate and get tokens."""
        if not self.email or not self.password:
            raise ValueError("Email and password required for authentication")
            
        response = requests.post(f"{self.api_url}/auth/login", json={
            "email": self.email,
            "password": self.password
        })
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data['access_token']
        self.refresh_token = data['refresh_token']
        self.token_expires = datetime.now() + timedelta(seconds=data['expires_in'] - 60)  # 1 minute buffer
    
    def _refresh_tokens(self):
        """Refresh access token."""
        if not self.refresh_token:
            self._login()
            return
            
        response = requests.post(f"{self.api_url}/auth/refresh", json={
            "refresh_token": self.refresh_token
        })
        
        if response.status_code == 401:
            self._login()
            return
            
        response.raise_for_status()
        data = response.json()
        self.access_token = data['access_token']
        self.refresh_token = data['refresh_token']
        self.token_expires = datetime.now() + timedelta(seconds=data['expires_in'] - 60)
    
    def execute_workflow(self, workflow: Dict[str, Any], 
                        priority: str = "normal",
                        webhook_url: str = None,
                        timeout_minutes: int = 30,
                        metadata: Dict[str, Any] = None) -> str:
        """Execute a ComfyUI workflow."""
        payload = {
            "workflow": workflow,
            "priority": priority,
            "timeout_minutes": timeout_minutes
        }
        
        if webhook_url:
            payload["webhook_url"] = webhook_url
        if metadata:
            payload["metadata"] = metadata
            
        response = self._make_request("POST", "/workflows/execute", json=payload)
        response.raise_for_status()
        
        return response.json()["execution_id"]
    
    def get_workflow_result(self, execution_id: str) -> WorkflowResult:
        """Get workflow execution result."""
        response = self._make_request("GET", f"/workflows/{execution_id}")
        response.raise_for_status()
        
        data = response.json()
        return WorkflowResult(
            execution_id=data["execution_id"],
            status=WorkflowStatus(data["status"]),
            outputs=data.get("outputs"),
            error=data.get("error"),
            duration_seconds=data.get("duration_seconds")
        )
    
    def wait_for_completion(self, execution_id: str, 
                          timeout_seconds: int = 1800,
                          poll_interval: int = 10) -> WorkflowResult:
        """Wait for workflow to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            result = self.get_workflow_result(execution_id)
            
            if result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                return result
                
            time.sleep(poll_interval)
            
        raise TimeoutError(f"Workflow {execution_id} did not complete within {timeout_seconds} seconds")
    
    def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow status with progress."""
        response = self._make_request("GET", f"/workflows/{execution_id}/status")
        response.raise_for_status()
        return response.json()
    
    def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a workflow execution."""
        response = self._make_request("POST", f"/workflows/{execution_id}/cancel")
        return response.status_code == 200
    
    def list_models(self, model_type: str = None, available_only: bool = False) -> List[Dict[str, Any]]:
        """List available models."""
        params = {}
        if model_type:
            params['model_type'] = model_type
        if available_only:
            params['available_only'] = available_only
            
        response = self._make_request("GET", "/models/", params=params)
        response.raise_for_status()
        return response.json()["models"]
    
    def load_model(self, model_name: str, model_type: str) -> bool:
        """Load a model into memory."""
        response = self._make_request("POST", f"/models/{model_name}/load", 
                                    params={"model_type": model_type})
        return response.status_code == 200
    
    def upload_file(self, file_path: str, description: str = None) -> str:
        """Upload a file and return file_id."""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {}
            if description:
                data['description'] = description
                
            # Note: _make_request handles auth headers, but for multipart we need special handling
            self._ensure_authenticated()
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            response = requests.post(f"{self.api_url}/files/upload", 
                                   files=files, data=data, headers=headers)
            response.raise_for_status()
            
        return response.json()["file_id"]


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = ComfyUIClient()
    
    # Simple text-to-image workflow
    workflow = {
        "nodes": {
            "1": {
                "id": "1",
                "class_type": "CheckpointLoaderSimple",
                "inputs": [
                    {"name": "ckpt_name", "type": "STRING", "value": "v1-5-pruned-emaonly.ckpt", "required": True}
                ]
            },
            "2": {
                "id": "2",
                "class_type": "CLIPTextEncode", 
                "inputs": [
                    {"name": "text", "type": "STRING", "value": "beautiful landscape, masterpiece", "required": True},
                    {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": True}
                ]
            }
            # ... more nodes
        }
    }
    
    # Execute workflow
    execution_id = client.execute_workflow(workflow)
    print(f"Workflow submitted: {execution_id}")
    
    # Wait for completion
    result = client.wait_for_completion(execution_id)
    print(f"Workflow completed with status: {result.status}")
    
    if result.outputs:
        print("Generated images:", result.outputs.get("images", []))
```

### Advanced Python Examples

```python
# advanced_examples.py
import asyncio
import aiohttp
from typing import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

class AsyncComfyUIClient:
    """Async version of ComfyUI client."""
    
    def __init__(self, api_url: str, access_token: str):
        self.api_url = api_url
        self.access_token = access_token
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.access_token}'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def execute_workflow(self, workflow: Dict[str, Any]) -> str:
        """Execute workflow asynchronously."""
        async with self.session.post(f"{self.api_url}/workflows/execute", 
                                   json={"workflow": workflow}) as response:
            response.raise_for_status()
            data = await response.json()
            return data["execution_id"]
    
    async def stream_progress(self, execution_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream workflow progress updates."""
        while True:
            async with self.session.get(f"{self.api_url}/workflows/{execution_id}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    yield data
                    
                    if data["status"] in ["completed", "failed", "cancelled"]:
                        break
                        
            await asyncio.sleep(5)


# Batch processing with async
async def batch_process_workflows(workflows: List[Dict[str, Any]], 
                                api_url: str, access_token: str) -> List[str]:
    """Process multiple workflows concurrently."""
    async with AsyncComfyUIClient(api_url, access_token) as client:
        tasks = [client.execute_workflow(workflow) for workflow in workflows]
        execution_ids = await asyncio.gather(*tasks)
        return execution_ids


# Image processing utilities
from PIL import Image, ImageEnhance
import base64
import io

class ImageProcessor:
    """Helper class for image processing operations."""
    
    @staticmethod
    def resize_image(image_path: str, max_size: tuple = (512, 512)) -> str:
        """Resize image and return base64 encoded."""
        with Image.open(image_path) as img:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return encoded
    
    @staticmethod
    def enhance_image(image_path: str, brightness: float = 1.0, 
                     contrast: float = 1.0, saturation: float = 1.0) -> str:
        """Apply image enhancements."""
        with Image.open(image_path) as img:
            if brightness != 1.0:
                img = ImageEnhance.Brightness(img).enhance(brightness)
            if contrast != 1.0:
                img = ImageEnhance.Contrast(img).enhance(contrast)
            if saturation != 1.0:
                img = ImageEnhance.Color(img).enhance(saturation)
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return encoded


# Workflow templates
class WorkflowTemplates:
    """Pre-built workflow templates."""
    
    @staticmethod
    def text_to_image(prompt: str, negative_prompt: str = "blurry, low quality",
                     seed: int = 42, steps: int = 20, cfg: float = 7.0,
                     width: int = 512, height: int = 512) -> Dict[str, Any]:
        """Generate text-to-image workflow."""
        return {
            "nodes": {
                "1": {
                    "id": "1",
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": [{"name": "ckpt_name", "type": "STRING", "value": "v1-5-pruned-emaonly.ckpt", "required": True}]
                },
                "2": {
                    "id": "2", 
                    "class_type": "CLIPTextEncode",
                    "inputs": [
                        {"name": "text", "type": "STRING", "value": prompt, "required": True},
                        {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": True}
                    ]
                },
                "3": {
                    "id": "3",
                    "class_type": "CLIPTextEncode",
                    "inputs": [
                        {"name": "text", "type": "STRING", "value": negative_prompt, "required": True},
                        {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": True}
                    ]
                },
                "4": {
                    "id": "4",
                    "class_type": "EmptyLatentImage",
                    "inputs": [
                        {"name": "width", "type": "INT", "value": width, "required": True},
                        {"name": "height", "type": "INT", "value": height, "required": True},
                        {"name": "batch_size", "type": "INT", "value": 1, "required": True}
                    ]
                },
                "5": {
                    "id": "5",
                    "class_type": "KSampler",
                    "inputs": [
                        {"name": "seed", "type": "INT", "value": seed, "required": True},
                        {"name": "steps", "type": "INT", "value": steps, "required": True},
                        {"name": "cfg", "type": "FLOAT", "value": cfg, "required": True},
                        {"name": "sampler_name", "type": "STRING", "value": "euler", "required": True},
                        {"name": "scheduler", "type": "STRING", "value": "normal", "required": True},
                        {"name": "denoise", "type": "FLOAT", "value": 1.0, "required": True},
                        {"name": "model", "type": "MODEL", "value": ["1", 0], "required": True},
                        {"name": "positive", "type": "CONDITIONING", "value": ["2", 0], "required": True},
                        {"name": "negative", "type": "CONDITIONING", "value": ["3", 0], "required": True},
                        {"name": "latent_image", "type": "LATENT", "value": ["4", 0], "required": True}
                    ]
                },
                "6": {
                    "id": "6",
                    "class_type": "VAEDecode",
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "value": ["5", 0], "required": True},
                        {"name": "vae", "type": "VAE", "value": ["1", 2], "required": True}
                    ]
                },
                "7": {
                    "id": "7",
                    "class_type": "SaveImage",
                    "inputs": [
                        {"name": "images", "type": "IMAGE", "value": ["6", 0], "required": True},
                        {"name": "filename_prefix", "type": "STRING", "value": "generated", "required": False}
                    ]
                }
            }
        }
    
    @staticmethod
    def image_to_image(input_image_id: str, prompt: str, strength: float = 0.7) -> Dict[str, Any]:
        """Generate image-to-image workflow."""
        return {
            "nodes": {
                "1": {"id": "1", "class_type": "LoadImage", "inputs": [{"name": "image", "type": "STRING", "value": input_image_id, "required": True}]},
                "2": {"id": "2", "class_type": "CheckpointLoaderSimple", "inputs": [{"name": "ckpt_name", "type": "STRING", "value": "v1-5-pruned-emaonly.ckpt", "required": True}]},
                "3": {"id": "3", "class_type": "CLIPTextEncode", "inputs": [{"name": "text", "type": "STRING", "value": prompt, "required": True}, {"name": "clip", "type": "CLIP", "value": ["2", 1], "required": True}]},
                "4": {"id": "4", "class_type": "VAEEncode", "inputs": [{"name": "pixels", "type": "IMAGE", "value": ["1", 0], "required": True}, {"name": "vae", "type": "VAE", "value": ["2", 2], "required": True}]},
                "5": {"id": "5", "class_type": "KSampler", "inputs": [{"name": "seed", "type": "INT", "value": 42, "required": True}, {"name": "steps", "type": "INT", "value": 20, "required": True}, {"name": "cfg", "type": "FLOAT", "value": 7.0, "required": True}, {"name": "sampler_name", "type": "STRING", "value": "euler", "required": True}, {"name": "scheduler", "type": "STRING", "value": "normal", "required": True}, {"name": "denoise", "type": "FLOAT", "value": strength, "required": True}, {"name": "model", "type": "MODEL", "value": ["2", 0], "required": True}, {"name": "positive", "type": "CONDITIONING", "value": ["3", 0], "required": True}, {"name": "latent_image", "type": "LATENT", "value": ["4", 0], "required": True}]},
                "6": {"id": "6", "class_type": "VAEDecode", "inputs": [{"name": "samples", "type": "LATENT", "value": ["5", 0], "required": True}, {"name": "vae", "type": "VAE", "value": ["2", 2], "required": True}]},
                "7": {"id": "7", "class_type": "SaveImage", "inputs": [{"name": "images", "type": "IMAGE", "value": ["6", 0], "required": True}]}
            }
        }


# Usage example with error handling
async def main():
    """Example usage with comprehensive error handling."""
    client = ComfyUIClient()
    
    try:
        # Generate multiple variations
        prompts = [
            "beautiful sunset over mountains",
            "futuristic city skyline at night", 
            "serene lake with forest reflection"
        ]
        
        execution_ids = []
        
        for i, prompt in enumerate(prompts):
            workflow = WorkflowTemplates.text_to_image(
                prompt=prompt,
                seed=42 + i,  # Different seed for each image
                steps=25,
                cfg=8.0
            )
            
            execution_id = client.execute_workflow(
                workflow, 
                metadata={"batch": "demo", "prompt_index": i}
            )
            execution_ids.append(execution_id)
            print(f"Submitted workflow {i+1}/{len(prompts)}: {execution_id}")
        
        # Monitor all workflows
        results = []
        for execution_id in execution_ids:
            print(f"Waiting for {execution_id}...")
            result = client.wait_for_completion(execution_id)
            results.append(result)
            
            if result.status == WorkflowStatus.COMPLETED:
                print(f"✅ Completed: {execution_id}")
                if result.outputs and "images" in result.outputs:
                    for img in result.outputs["images"]:
                        print(f"   Generated: {img.get('filename', 'unknown')}")
            else:
                print(f"❌ Failed: {execution_id} - {result.error}")
        
        print(f"\nBatch complete: {len([r for r in results if r.status == WorkflowStatus.COMPLETED])}/{len(results)} successful")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

## JavaScript/Node.js SDK

### Package Installation

```json
{
  "name": "comfyui-client",
  "version": "1.0.0",
  "dependencies": {
    "axios": "^1.3.0",
    "form-data": "^4.0.0",
    "ws": "^8.12.0",
    "dotenv": "^16.0.0"
  }
}
```

### Basic JavaScript Client

```javascript
// comfyui-client.js
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
require('dotenv').config();

class ComfyUIClient {
    constructor(apiUrl = null, email = null, password = null) {
        this.apiUrl = apiUrl || process.env.COMFYUI_API_URL || 'https://api.comfyui-serverless.com';
        this.email = email || process.env.COMFYUI_EMAIL;
        this.password = password || process.env.COMFYUI_PASSWORD;
        this.accessToken = null;
        this.refreshToken = null;
        this.tokenExpires = null;
        
        // Create axios instance with interceptors
        this.axios = axios.create({
            baseURL: this.apiUrl,
            timeout: 30000
        });
        
        this.setupInterceptors();
    }
    
    setupInterceptors() {
        // Request interceptor to add auth headers
        this.axios.interceptors.request.use(async (config) => {
            await this.ensureAuthenticated();
            if (this.accessToken) {
                config.headers.Authorization = `Bearer ${this.accessToken}`;
            }
            return config;
        });
        
        // Response interceptor to handle token refresh
        this.axios.interceptors.response.use(
            (response) => response,
            async (error) => {
                if (error.response?.status === 401 && !error.config._retry) {
                    error.config._retry = true;
                    await this.refreshTokens();
                    error.config.headers.Authorization = `Bearer ${this.accessToken}`;
                    return this.axios.request(error.config);
                }
                return Promise.reject(error);
            }
        );
    }
    
    async ensureAuthenticated() {
        if (!this.accessToken || this.tokenExpired()) {
            if (this.refreshToken && !this.refreshTokenExpired()) {
                await this.refreshTokens();
            } else {
                await this.login();
            }
        }
    }
    
    tokenExpired() {
        return !this.tokenExpires || new Date() >= this.tokenExpires;
    }
    
    refreshTokenExpired() {
        // Implement refresh token expiration logic
        return false;
    }
    
    async login() {
        if (!this.email || !this.password) {
            throw new Error('Email and password required for authentication');
        }
        
        const response = await axios.post(`${this.apiUrl}/auth/login`, {
            email: this.email,
            password: this.password
        });
        
        const { access_token, refresh_token, expires_in } = response.data;
        this.accessToken = access_token;
        this.refreshToken = refresh_token;
        this.tokenExpires = new Date(Date.now() + (expires_in - 60) * 1000); // 1 minute buffer
    }
    
    async refreshTokens() {
        if (!this.refreshToken) {
            await this.login();
            return;
        }
        
        try {
            const response = await axios.post(`${this.apiUrl}/auth/refresh`, {
                refresh_token: this.refreshToken
            });
            
            const { access_token, refresh_token, expires_in } = response.data;
            this.accessToken = access_token;
            this.refreshToken = refresh_token;
            this.tokenExpires = new Date(Date.now() + (expires_in - 60) * 1000);
        } catch (error) {
            if (error.response?.status === 401) {
                await this.login();
            } else {
                throw error;
            }
        }
    }
    
    async executeWorkflow(workflow, options = {}) {
        const payload = {
            workflow,
            priority: options.priority || 'normal',
            timeout_minutes: options.timeoutMinutes || 30,
            ...options
        };
        
        const response = await this.axios.post('/workflows/execute', payload);
        return response.data.execution_id;
    }
    
    async getWorkflowResult(executionId) {
        const response = await this.axios.get(`/workflows/${executionId}`);
        return response.data;
    }
    
    async waitForCompletion(executionId, timeoutSeconds = 1800, pollInterval = 10) {
        const startTime = Date.now();
        const timeoutMs = timeoutSeconds * 1000;
        
        return new Promise((resolve, reject) => {
            const poll = async () => {
                try {
                    if (Date.now() - startTime > timeoutMs) {
                        reject(new Error(`Workflow ${executionId} did not complete within ${timeoutSeconds} seconds`));
                        return;
                    }
                    
                    const result = await this.getWorkflowResult(executionId);
                    
                    if (['completed', 'failed', 'cancelled'].includes(result.status)) {
                        resolve(result);
                        return;
                    }
                    
                    setTimeout(poll, pollInterval * 1000);
                } catch (error) {
                    reject(error);
                }
            };
            
            poll();
        });
    }
    
    async getWorkflowStatus(executionId) {
        const response = await this.axios.get(`/workflows/${executionId}/status`);
        return response.data;
    }
    
    async cancelWorkflow(executionId) {
        const response = await this.axios.post(`/workflows/${executionId}/cancel`);
        return response.status === 200;
    }
    
    async listModels(modelType = null, availableOnly = false) {
        const params = {};
        if (modelType) params.model_type = modelType;
        if (availableOnly) params.available_only = availableOnly;
        
        const response = await this.axios.get('/models/', { params });
        return response.data.models;
    }
    
    async loadModel(modelName, modelType) {
        const response = await this.axios.post(`/models/${modelName}/load`, null, {
            params: { model_type: modelType }
        });
        return response.status === 200;
    }
    
    async uploadFile(filePath, description = null) {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));
        if (description) {
            formData.append('description', description);
        }
        
        const response = await this.axios.post('/files/upload', formData, {
            headers: formData.getHeaders()
        });
        
        return response.data.file_id;
    }
    
    async downloadFile(fileId, outputPath) {
        const response = await this.axios.get(`/files/${fileId}`, {
            responseType: 'stream'
        });
        
        const writer = fs.createWriteStream(outputPath);
        response.data.pipe(writer);
        
        return new Promise((resolve, reject) => {
            writer.on('finish', resolve);
            writer.on('error', reject);
        });
    }
}

// Workflow templates
class WorkflowTemplates {
    static textToImage(prompt, negativePrompt = 'blurry, low quality', options = {}) {
        const {
            seed = 42,
            steps = 20,
            cfg = 7.0,
            width = 512,
            height = 512
        } = options;
        
        return {
            nodes: {
                "1": {
                    id: "1",
                    class_type: "CheckpointLoaderSimple",
                    inputs: [
                        { name: "ckpt_name", type: "STRING", value: "v1-5-pruned-emaonly.ckpt", required: true }
                    ]
                },
                "2": {
                    id: "2",
                    class_type: "CLIPTextEncode",
                    inputs: [
                        { name: "text", type: "STRING", value: prompt, required: true },
                        { name: "clip", type: "CLIP", value: ["1", 1], required: true }
                    ]
                },
                "3": {
                    id: "3",
                    class_type: "CLIPTextEncode",
                    inputs: [
                        { name: "text", type: "STRING", value: negativePrompt, required: true },
                        { name: "clip", type: "CLIP", value: ["1", 1], required: true }
                    ]
                },
                "4": {
                    id: "4",
                    class_type: "EmptyLatentImage",
                    inputs: [
                        { name: "width", type: "INT", value: width, required: true },
                        { name: "height", type: "INT", value: height, required: true },
                        { name: "batch_size", type: "INT", value: 1, required: true }
                    ]
                },
                "5": {
                    id: "5",
                    class_type: "KSampler",
                    inputs: [
                        { name: "seed", type: "INT", value: seed, required: true },
                        { name: "steps", type: "INT", value: steps, required: true },
                        { name: "cfg", type: "FLOAT", value: cfg, required: true },
                        { name: "sampler_name", type: "STRING", value: "euler", required: true },
                        { name: "scheduler", type: "STRING", value: "normal", required: true },
                        { name: "denoise", type: "FLOAT", value: 1.0, required: true },
                        { name: "model", type: "MODEL", value: ["1", 0], required: true },
                        { name: "positive", type: "CONDITIONING", value: ["2", 0], required: true },
                        { name: "negative", type: "CONDITIONING", value: ["3", 0], required: true },
                        { name: "latent_image", type: "LATENT", value: ["4", 0], required: true }
                    ]
                },
                "6": {
                    id: "6",
                    class_type: "VAEDecode",
                    inputs: [
                        { name: "samples", type: "LATENT", value: ["5", 0], required: true },
                        { name: "vae", type: "VAE", value: ["1", 2], required: true }
                    ]
                },
                "7": {
                    id: "7",
                    class_type: "SaveImage",
                    inputs: [
                        { name: "images", type: "IMAGE", value: ["6", 0], required: true },
                        { name: "filename_prefix", type: "STRING", value: "generated", required: false }
                    ]
                }
            }
        };
    }
}

// Example usage
async function main() {
    const client = new ComfyUIClient();
    
    try {
        // Generate image
        const workflow = WorkflowTemplates.textToImage(
            "beautiful sunset over mountains, masterpiece",
            "blurry, low quality",
            { seed: 123, steps: 25, cfg: 8.0 }
        );
        
        console.log('Submitting workflow...');
        const executionId = await client.executeWorkflow(workflow);
        console.log(`Workflow submitted: ${executionId}`);
        
        // Monitor progress
        const checkProgress = setInterval(async () => {
            try {
                const status = await client.getWorkflowStatus(executionId);
                console.log(`Status: ${status.status}, Progress: ${status.progress?.percentage || 0}%`);
            } catch (error) {
                console.error('Error checking progress:', error.message);
            }
        }, 5000);
        
        // Wait for completion
        const result = await client.waitForCompletion(executionId);
        clearInterval(checkProgress);
        
        console.log(`Workflow completed with status: ${result.status}`);
        
        if (result.outputs && result.outputs.images) {
            console.log('Generated images:');
            result.outputs.images.forEach((img, index) => {
                console.log(`  ${index + 1}: ${img.filename || 'unknown'}`);
            });
        }
        
    } catch (error) {
        console.error('Error:', error.message);
        if (error.response?.data) {
            console.error('API Error:', error.response.data);
        }
    }
}

// Advanced example with batch processing
async function batchProcess() {
    const client = new ComfyUIClient();
    
    const prompts = [
        "beautiful landscape with mountains",
        "futuristic city skyline",
        "serene ocean sunset",
        "mystical forest scene"
    ];
    
    try {
        // Submit all workflows
        const executionIds = await Promise.all(
            prompts.map(async (prompt, index) => {
                const workflow = WorkflowTemplates.textToImage(prompt, "blurry", {
                    seed: 100 + index,
                    steps: 20
                });
                
                return await client.executeWorkflow(workflow, {
                    metadata: { batch: 'demo', index }
                });
            })
        );
        
        console.log(`Submitted ${executionIds.length} workflows`);
        
        // Wait for all to complete
        const results = await Promise.all(
            executionIds.map(id => client.waitForCompletion(id))
        );
        
        const successful = results.filter(r => r.status === 'completed').length;
        console.log(`Batch complete: ${successful}/${results.length} successful`);
        
    } catch (error) {
        console.error('Batch processing error:', error.message);
    }
}

module.exports = { ComfyUIClient, WorkflowTemplates };

if (require.main === module) {
    main();
}
```

### React Integration Example

```javascript
// react-comfyui-hook.js
import { useState, useCallback, useRef, useEffect } from 'react';
import { ComfyUIClient } from './comfyui-client';

export const useComfyUI = (apiConfig) => {
    const [client] = useState(() => new ComfyUIClient(apiConfig.apiUrl, apiConfig.email, apiConfig.password));
    const [executions, setExecutions] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const progressIntervals = useRef({});
    
    const submitWorkflow = useCallback(async (workflow, options = {}) => {
        setIsLoading(true);
        try {
            const executionId = await client.executeWorkflow(workflow, options);
            
            // Initialize execution state
            setExecutions(prev => ({
                ...prev,
                [executionId]: {
                    id: executionId,
                    status: 'pending',
                    progress: 0,
                    result: null,
                    error: null
                }
            }));
            
            // Start progress monitoring
            progressIntervals.current[executionId] = setInterval(async () => {
                try {
                    const status = await client.getWorkflowStatus(executionId);
                    setExecutions(prev => ({
                        ...prev,
                        [executionId]: {
                            ...prev[executionId],
                            status: status.status,
                            progress: status.progress?.percentage || 0
                        }
                    }));
                    
                    // Stop monitoring if completed
                    if (['completed', 'failed', 'cancelled'].includes(status.status)) {
                        clearInterval(progressIntervals.current[executionId]);
                        delete progressIntervals.current[executionId];
                        
                        // Get final result
                        if (status.status === 'completed') {
                            const result = await client.getWorkflowResult(executionId);
                            setExecutions(prev => ({
                                ...prev,
                                [executionId]: {
                                    ...prev[executionId],
                                    result
                                }
                            }));
                        }
                    }
                } catch (error) {
                    console.error('Progress monitoring error:', error);
                }
            }, 2000);
            
            return executionId;
        } catch (error) {
            console.error('Workflow submission error:', error);
            throw error;
        } finally {
            setIsLoading(false);
        }
    }, [client]);
    
    const cancelWorkflow = useCallback(async (executionId) => {
        try {
            await client.cancelWorkflow(executionId);
            
            // Stop progress monitoring
            if (progressIntervals.current[executionId]) {
                clearInterval(progressIntervals.current[executionId]);
                delete progressIntervals.current[executionId];
            }
            
            // Update status
            setExecutions(prev => ({
                ...prev,
                [executionId]: {
                    ...prev[executionId],
                    status: 'cancelled'
                }
            }));
        } catch (error) {
            console.error('Cancel workflow error:', error);
            throw error;
        }
    }, [client]);
    
    // Cleanup on unmount
    useEffect(() => {
        return () => {
            Object.values(progressIntervals.current).forEach(interval => {
                clearInterval(interval);
            });
        };
    }, []);
    
    return {
        client,
        executions,
        isLoading,
        submitWorkflow,
        cancelWorkflow
    };
};

// React component example
import React, { useState } from 'react';
import { useComfyUI } from './react-comfyui-hook';
import { WorkflowTemplates } from './comfyui-client';

const ImageGenerator = () => {
    const [prompt, setPrompt] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('blurry, low quality');
    const [seed, setSeed] = useState(42);
    const [steps, setSteps] = useState(20);
    
    const { executions, isLoading, submitWorkflow, cancelWorkflow } = useComfyUI({
        apiUrl: process.env.REACT_APP_COMFYUI_API_URL,
        email: process.env.REACT_APP_COMFYUI_EMAIL,
        password: process.env.REACT_APP_COMFYUI_PASSWORD
    });
    
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!prompt.trim()) return;
        
        try {
            const workflow = WorkflowTemplates.textToImage(prompt, negativePrompt, {
                seed: parseInt(seed),
                steps: parseInt(steps)
            });
            
            await submitWorkflow(workflow, {
                metadata: { prompt, timestamp: Date.now() }
            });
        } catch (error) {
            alert('Error submitting workflow: ' + error.message);
        }
    };
    
    return (
        <div className="image-generator">
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Prompt:</label>
                    <textarea 
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Describe the image you want to generate..."
                        rows={3}
                    />
                </div>
                
                <div>
                    <label>Negative Prompt:</label>
                    <input 
                        value={negativePrompt}
                        onChange={(e) => setNegativePrompt(e.target.value)}
                    />
                </div>
                
                <div>
                    <label>Seed:</label>
                    <input 
                        type="number" 
                        value={seed}
                        onChange={(e) => setSeed(e.target.value)}
                    />
                </div>
                
                <div>
                    <label>Steps:</label>
                    <input 
                        type="number" 
                        value={steps}
                        min={1}
                        max={50}
                        onChange={(e) => setSteps(e.target.value)}
                    />
                </div>
                
                <button type="submit" disabled={isLoading}>
                    {isLoading ? 'Submitting...' : 'Generate Image'}
                </button>
            </form>
            
            <div className="executions">
                {Object.values(executions).map(execution => (
                    <div key={execution.id} className="execution-card">
                        <h3>Execution: {execution.id.substring(0, 8)}...</h3>
                        <p>Status: {execution.status}</p>
                        
                        {execution.status === 'running' && (
                            <div className="progress">
                                <progress value={execution.progress} max={100} />
                                <span>{execution.progress}%</span>
                            </div>
                        )}
                        
                        {execution.status === 'pending' && (
                            <button onClick={() => cancelWorkflow(execution.id)}>
                                Cancel
                            </button>
                        )}
                        
                        {execution.result?.outputs?.images && (
                            <div className="generated-images">
                                {execution.result.outputs.images.map((img, idx) => (
                                    <div key={idx} className="image-result">
                                        <img src={img.url} alt={`Generated ${idx}`} />
                                        <p>{img.filename}</p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ImageGenerator;
```

## curl Examples

### Basic Authentication and Workflow Execution

```bash
#!/bin/bash
# comprehensive_curl_examples.sh

# Configuration
API_URL="https://api.comfyui-serverless.com"
EMAIL="your-email@example.com"
PASSWORD="your-password"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to handle authentication
authenticate() {
    log_info "Authenticating with API..."
    
    response=$(curl -s -w "%{http_code}" -X POST "$API_URL/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")
    
    http_code="${response: -3}"
    body="${response%???}"
    
    if [ "$http_code" -eq 200 ]; then
        ACCESS_TOKEN=$(echo "$body" | jq -r '.access_token')
        REFRESH_TOKEN=$(echo "$body" | jq -r '.refresh_token')
        
        # Save tokens
        echo "ACCESS_TOKEN=\"$ACCESS_TOKEN\"" > .tokens
        echo "REFRESH_TOKEN=\"$REFRESH_TOKEN\"" >> .tokens
        
        log_info "Authentication successful"
        return 0
    else
        log_error "Authentication failed: $body"
        return 1
    fi
}

# Function to refresh tokens
refresh_tokens() {
    if [ -f ".tokens" ]; then
        source .tokens
        
        response=$(curl -s -w "%{http_code}" -X POST "$API_URL/auth/refresh" \
            -H "Content-Type: application/json" \
            -d "{\"refresh_token\":\"$REFRESH_TOKEN\"}")
        
        http_code="${response: -3}"
        body="${response%???}"
        
        if [ "$http_code" -eq 200 ]; then
            ACCESS_TOKEN=$(echo "$body" | jq -r '.access_token')
            REFRESH_TOKEN=$(echo "$body" | jq -r '.refresh_token')
            
            echo "ACCESS_TOKEN=\"$ACCESS_TOKEN\"" > .tokens
            echo "REFRESH_TOKEN=\"$REFRESH_TOKEN\"" >> .tokens
            
            log_info "Tokens refreshed"
            return 0
        else
            log_warn "Token refresh failed, re-authenticating..."
            authenticate
        fi
    else
        authenticate
    fi
}

# Function to make authenticated API calls
api_call() {
    local method="$1"
    local endpoint="$2"
    shift 2
    
    # Ensure we have valid tokens
    if [ ! -f ".tokens" ]; then
        authenticate || return 1
    fi
    
    source .tokens
    
    response=$(curl -s -w "%{http_code}" -X "$method" "$API_URL$endpoint" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        "$@")
    
    http_code="${response: -3}"
    body="${response%???}"
    
    # Handle token expiration
    if [ "$http_code" -eq 401 ]; then
        log_warn "Access token expired, refreshing..."
        refresh_tokens
        source .tokens
        
        response=$(curl -s -w "%{http_code}" -X "$method" "$API_URL$endpoint" \
            -H "Authorization: Bearer $ACCESS_TOKEN" \
            -H "Content-Type: application/json" \
            "$@")
        
        http_code="${response: -3}"
        body="${response%???}"
    fi
    
    echo "$http_code|$body"
}

# Function to submit text-to-image workflow
submit_txt2img() {
    local prompt="$1"
    local negative_prompt="${2:-blurry, low quality}"
    local seed="${3:-42}"
    local steps="${4:-20}"
    local cfg="${5:-7.0}"
    local width="${6:-512}"
    local height="${7:-512}"
    
    local workflow=$(cat << EOF
{
  "workflow": {
    "nodes": {
      "1": {
        "id": "1",
        "class_type": "CheckpointLoaderSimple",
        "inputs": [
          {"name": "ckpt_name", "type": "STRING", "value": "v1-5-pruned-emaonly.ckpt", "required": true}
        ]
      },
      "2": {
        "id": "2",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {"name": "text", "type": "STRING", "value": "$prompt", "required": true},
          {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": true}
        ]
      },
      "3": {
        "id": "3",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {"name": "text", "type": "STRING", "value": "$negative_prompt", "required": true},
          {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": true}
        ]
      },
      "4": {
        "id": "4",
        "class_type": "EmptyLatentImage",
        "inputs": [
          {"name": "width", "type": "INT", "value": $width, "required": true},
          {"name": "height", "type": "INT", "value": $height, "required": true},
          {"name": "batch_size", "type": "INT", "value": 1, "required": true}
        ]
      },
      "5": {
        "id": "5",
        "class_type": "KSampler",
        "inputs": [
          {"name": "seed", "type": "INT", "value": $seed, "required": true},
          {"name": "steps", "type": "INT", "value": $steps, "required": true},
          {"name": "cfg", "type": "FLOAT", "value": $cfg, "required": true},
          {"name": "sampler_name", "type": "STRING", "value": "euler", "required": true},
          {"name": "scheduler", "type": "STRING", "value": "normal", "required": true},
          {"name": "denoise", "type": "FLOAT", "value": 1.0, "required": true},
          {"name": "model", "type": "MODEL", "value": ["1", 0], "required": true},
          {"name": "positive", "type": "CONDITIONING", "value": ["2", 0], "required": true},
          {"name": "negative", "type": "CONDITIONING", "value": ["3", 0], "required": true},
          {"name": "latent_image", "type": "LATENT", "value": ["4", 0], "required": true}
        ]
      },
      "6": {
        "id": "6",
        "class_type": "VAEDecode",
        "inputs": [
          {"name": "samples", "type": "LATENT", "value": ["5", 0], "required": true},
          {"name": "vae", "type": "VAE", "value": ["1", 2], "required": true}
        ]
      },
      "7": {
        "id": "7",
        "class_type": "SaveImage",
        "inputs": [
          {"name": "images", "type": "IMAGE", "value": ["6", 0], "required": true},
          {"name": "filename_prefix", "type": "STRING", "value": "generated", "required": false}
        ]
      }
    },
    "metadata": {
      "description": "Text-to-image generation",
      "prompt": "$prompt",
      "created_by": "curl_script"
    }
  },
  "priority": "normal",
  "timeout_minutes": 30
}
EOF
    )
    
    log_info "Submitting text-to-image workflow..."
    log_info "Prompt: $prompt"
    
    result=$(api_call POST "/workflows/execute" -d "$workflow")
    http_code=$(echo "$result" | cut -d'|' -f1)
    body=$(echo "$result" | cut -d'|' -f2-)
    
    if [ "$http_code" -eq 200 ]; then
        execution_id=$(echo "$body" | jq -r '.execution_id')
        log_info "Workflow submitted successfully: $execution_id"
        echo "$execution_id"
        return 0
    else
        log_error "Workflow submission failed: $body"
        return 1
    fi
}

# Function to monitor workflow progress
monitor_workflow() {
    local execution_id="$1"
    local timeout_seconds="${2:-1800}" # 30 minutes default
    local start_time=$(date +%s)
    
    log_info "Monitoring workflow: $execution_id"
    
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout_seconds ]; then
            log_error "Timeout reached after ${timeout_seconds} seconds"
            return 1
        fi
        
        result=$(api_call GET "/workflows/$execution_id/status")
        http_code=$(echo "$result" | cut -d'|' -f1)
        body=$(echo "$result" | cut -d'|' -f2-)
        
        if [ "$http_code" -eq 200 ]; then
            status=$(echo "$body" | jq -r '.status')
            progress=$(echo "$body" | jq -r '.progress.percentage // 0')
            queue_position=$(echo "$body" | jq -r '.queue_position // null')
            
            if [ "$queue_position" != "null" ]; then
                log_info "Status: $status, Queue position: $queue_position"
            else
                log_info "Status: $status, Progress: ${progress}%"
            fi
            
            case $status in
                "completed")
                    log_info "Workflow completed successfully!"
                    return 0
                    ;;
                "failed")
                    log_error "Workflow failed"
                    # Get error details
                    result=$(api_call GET "/workflows/$execution_id")
                    error=$(echo "$result" | cut -d'|' -f2- | jq -r '.error // "Unknown error"')
                    log_error "Error: $error"
                    return 1
                    ;;
                "cancelled")
                    log_warn "Workflow was cancelled"
                    return 1
                    ;;
            esac
        else
            log_error "Failed to get workflow status: $body"
        fi
        
        sleep 10
    done
}

# Function to get workflow results
get_workflow_results() {
    local execution_id="$1"
    
    log_info "Getting workflow results for: $execution_id"
    
    result=$(api_call GET "/workflows/$execution_id")
    http_code=$(echo "$result" | cut -d'|' -f1)
    body=$(echo "$result" | cut -d'|' -f2-)
    
    if [ "$http_code" -eq 200 ]; then
        echo "$body" | jq '.'
        
        # Extract image URLs if available
        images=$(echo "$body" | jq -r '.outputs.images[]?.url // empty' 2>/dev/null)
        if [ -n "$images" ]; then
            log_info "Generated images:"
            echo "$images" | while read -r url; do
                filename=$(basename "$url")
                log_info "  - $filename: $url"
            done
        fi
        
        return 0
    else
        log_error "Failed to get workflow results: $body"
        return 1
    fi
}

# Function to download generated images
download_images() {
    local execution_id="$1"
    local output_dir="${2:-.}"
    
    mkdir -p "$output_dir"
    
    result=$(api_call GET "/workflows/$execution_id")
    body=$(echo "$result" | cut -d'|' -f2-)
    
    images=$(echo "$body" | jq -r '.outputs.images[]?.url // empty' 2>/dev/null)
    
    if [ -n "$images" ]; then
        log_info "Downloading images to: $output_dir"
        echo "$images" | while read -r url; do
            filename=$(basename "$url")
            log_info "Downloading: $filename"
            
            curl -s -L -o "$output_dir/$filename" "$url"
            
            if [ $? -eq 0 ]; then
                log_info "Downloaded: $output_dir/$filename"
            else
                log_error "Failed to download: $filename"
            fi
        done
    else
        log_warn "No images found in workflow results"
    fi
}

# Function to upload file
upload_file() {
    local file_path="$1"
    local description="$2"
    
    if [ ! -f "$file_path" ]; then
        log_error "File not found: $file_path"
        return 1
    fi
    
    source .tokens
    
    log_info "Uploading file: $file_path"
    
    if [ -n "$description" ]; then
        response=$(curl -s -w "%{http_code}" -X POST "$API_URL/files/upload" \
            -H "Authorization: Bearer $ACCESS_TOKEN" \
            -F "file=@$file_path" \
            -F "description=$description")
    else
        response=$(curl -s -w "%{http_code}" -X POST "$API_URL/files/upload" \
            -H "Authorization: Bearer $ACCESS_TOKEN" \
            -F "file=@$file_path")
    fi
    
    http_code="${response: -3}"
    body="${response%???}"
    
    if [ "$http_code" -eq 200 ]; then
        file_id=$(echo "$body" | jq -r '.file_id')
        log_info "File uploaded successfully: $file_id"
        echo "$file_id"
        return 0
    else
        log_error "File upload failed: $body"
        return 1
    fi
}

# Function to list and manage models
manage_models() {
    log_info "Listing available models..."
    
    result=$(api_call GET "/models/?available_only=true")
    http_code=$(echo "$result" | cut -d'|' -f1)
    body=$(echo "$result" | cut -d'|' -f2-)
    
    if [ "$http_code" -eq 200 ]; then
        echo "$body" | jq -r '.models[] | "- \(.name) (\(.type)) - Loaded: \(.is_loaded), Memory: \(.memory_usage_mb // 0) MB"'
    else
        log_error "Failed to list models: $body"
    fi
}

# Function for health check
health_check() {
    log_info "Checking API health..."
    
    response=$(curl -s -w "%{http_code}" "$API_URL/health/detailed")
    http_code="${response: -3}"
    body="${response%???}"
    
    if [ "$http_code" -eq 200 ]; then
        status=$(echo "$body" | jq -r '.status')
        version=$(echo "$body" | jq -r '.version')
        uptime=$(echo "$body" | jq -r '.uptime_seconds')
        
        log_info "API Status: $status"
        log_info "Version: $version"
        log_info "Uptime: ${uptime}s"
        
        # Show service statuses
        echo "$body" | jq -r '.services[] | "- \(.name): \(.status) (\(.response_time_ms // 0)ms)"'
        
        return 0
    else
        log_error "Health check failed: $body"
        return 1
    fi
}

# Main execution based on command line arguments
case "${1:-help}" in
    "auth")
        authenticate
        ;;
    "txt2img")
        prompt="$2"
        if [ -z "$prompt" ]; then
            log_error "Usage: $0 txt2img \"your prompt here\" [negative_prompt] [seed] [steps] [cfg]"
            exit 1
        fi
        
        execution_id=$(submit_txt2img "$prompt" "$3" "$4" "$5" "$6")
        if [ $? -eq 0 ]; then
            if monitor_workflow "$execution_id"; then
                get_workflow_results "$execution_id"
                download_images "$execution_id" "generated_images"
            fi
        fi
        ;;
    "monitor")
        execution_id="$2"
        if [ -z "$execution_id" ]; then
            log_error "Usage: $0 monitor <execution_id>"
            exit 1
        fi
        monitor_workflow "$execution_id"
        ;;
    "results")
        execution_id="$2"
        if [ -z "$execution_id" ]; then
            log_error "Usage: $0 results <execution_id>"
            exit 1
        fi
        get_workflow_results "$execution_id"
        ;;
    "download")
        execution_id="$2"
        output_dir="$3"
        if [ -z "$execution_id" ]; then
            log_error "Usage: $0 download <execution_id> [output_dir]"
            exit 1
        fi
        download_images "$execution_id" "$output_dir"
        ;;
    "upload")
        file_path="$2"
        description="$3"
        if [ -z "$file_path" ]; then
            log_error "Usage: $0 upload <file_path> [description]"
            exit 1
        fi
        upload_file "$file_path" "$description"
        ;;
    "models")
        manage_models
        ;;
    "health")
        health_check
        ;;
    "help"|*)
        echo "ComfyUI Serverless API - curl Examples"
        echo
        echo "Usage: $0 <command> [arguments]"
        echo
        echo "Commands:"
        echo "  auth                                - Authenticate and get tokens"
        echo "  txt2img \"prompt\" [neg] [seed] [steps] - Generate text-to-image"
        echo "  monitor <execution_id>              - Monitor workflow progress"
        echo "  results <execution_id>              - Get workflow results"
        echo "  download <execution_id> [dir]       - Download generated images"
        echo "  upload <file_path> [description]    - Upload a file"
        echo "  models                              - List available models"
        echo "  health                              - Check API health"
        echo
        echo "Examples:"
        echo "  $0 txt2img \"beautiful sunset over mountains\""
        echo "  $0 txt2img \"futuristic city\" \"blurry\" 123 25 8.0"
        echo "  $0 upload input.jpg \"Reference image\""
        echo
        ;;
esac
```

## Integration Examples

### Flask Web Application

```python
# flask_integration.py
from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename
import os
import json
import tempfile
from comfyui_client import ComfyUIClient, WorkflowTemplates

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize ComfyUI client
client = ComfyUIClient()

# HTML template for simple UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ComfyUI Web Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input, textarea, select { width: 100%; padding: 8px; box-sizing: border-box; }
        button { padding: 10px 20px; background: #007cba; color: white; border: none; cursor: pointer; }
        button:hover { background: #005a8b; }
        .execution { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .completed { border-color: #4caf50; }
        .failed { border-color: #f44336; }
        .running { border-color: #ff9800; }
        .progress { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
        .progress-bar { height: 100%; background: #4caf50; transition: width 0.3s; }
        .images { display: flex; flex-wrap: wrap; gap: 10px; }
        .image { max-width: 200px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ComfyUI Serverless API</h1>
        
        <div class="form-group">
            <h2>Generate Image</h2>
            <form id="generateForm">
                <div class="form-group">
                    <label>Prompt:</label>
                    <textarea id="prompt" rows="3" placeholder="Describe the image you want to generate..."></textarea>
                </div>
                
                <div class="form-group">
                    <label>Negative Prompt:</label>
                    <input type="text" id="negativePrompt" value="blurry, low quality">
                </div>
                
                <div style="display: flex; gap: 15px;">
                    <div class="form-group" style="flex: 1;">
                        <label>Seed:</label>
                        <input type="number" id="seed" value="42">
                    </div>
                    <div class="form-group" style="flex: 1;">
                        <label>Steps:</label>
                        <input type="number" id="steps" value="20" min="1" max="50">
                    </div>
                    <div class="form-group" style="flex: 1;">
                        <label>CFG Scale:</label>
                        <input type="number" id="cfg" value="7.0" step="0.1" min="1" max="20">
                    </div>
                </div>
                
                <button type="submit">Generate Image</button>
            </form>
        </div>
        
        <div id="executions">
            <h2>Executions</h2>
            <div id="executionsList"></div>
        </div>
    </div>
    
    <script>
        let executions = {};
        
        document.getElementById('generateForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                prompt: document.getElementById('prompt').value,
                negative_prompt: document.getElementById('negativePrompt').value,
                seed: parseInt(document.getElementById('seed').value),
                steps: parseInt(document.getElementById('steps').value),
                cfg: parseFloat(document.getElementById('cfg').value)
            };
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                if (response.ok) {
                    addExecution(result.execution_id, formData.prompt);
                    startMonitoring(result.execution_id);
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
        
        function addExecution(executionId, prompt) {
            executions[executionId] = {
                id: executionId,
                prompt: prompt,
                status: 'pending',
                progress: 0
            };
            updateExecutionsDisplay();
        }
        
        function startMonitoring(executionId) {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/status/${executionId}`);
                    const status = await response.json();
                    
                    if (executions[executionId]) {
                        executions[executionId].status = status.status;
                        executions[executionId].progress = status.progress || 0;
                        updateExecutionsDisplay();
                        
                        if (['completed', 'failed', 'cancelled'].includes(status.status)) {
                            clearInterval(interval);
                            
                            if (status.status === 'completed') {
                                // Get results
                                const resultResponse = await fetch(`/api/results/${executionId}`);
                                const results = await resultResponse.json();
                                executions[executionId].results = results;
                                updateExecutionsDisplay();
                            }
                        }
                    }
                } catch (error) {
                    console.error('Monitoring error:', error);
                }
            }, 2000);
        }
        
        function updateExecutionsDisplay() {
            const container = document.getElementById('executionsList');
            container.innerHTML = '';
            
            Object.values(executions).reverse().forEach(execution => {
                const div = document.createElement('div');
                div.className = `execution ${execution.status}`;
                
                let html = `
                    <h3>Execution: ${execution.id.substring(0, 8)}...</h3>
                    <p><strong>Prompt:</strong> ${execution.prompt}</p>
                    <p><strong>Status:</strong> ${execution.status}</p>
                `;
                
                if (execution.status === 'running' && execution.progress > 0) {
                    html += `
                        <div class="progress">
                            <div class="progress-bar" style="width: ${execution.progress}%"></div>
                        </div>
                        <p>Progress: ${execution.progress}%</p>
                    `;
                }
                
                if (execution.results && execution.results.outputs && execution.results.outputs.images) {
                    html += '<div class="images">';
                    execution.results.outputs.images.forEach(img => {
                        html += `<img src="/api/image/${execution.id}/${img.filename}" class="image" alt="Generated image">`;
                    });
                    html += '</div>';
                }
                
                div.innerHTML = html;
                container.appendChild(div);
            });
        }
        
        // Auto-refresh executions on page load
        fetch('/api/executions')
            .then(response => response.json())
            .then(data => {
                data.forEach(execution => {
                    executions[execution.execution_id] = {
                        id: execution.execution_id,
                        prompt: execution.metadata?.prompt || 'Unknown',
                        status: execution.status,
                        progress: 100,
                        results: execution
                    };
                });
                updateExecutionsDisplay();
            });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/generate', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        
        workflow = WorkflowTemplates.text_to_image(
            prompt=data['prompt'],
            negative_prompt=data.get('negative_prompt', 'blurry, low quality'),
            seed=data.get('seed', 42),
            steps=data.get('steps', 20),
            cfg=data.get('cfg', 7.0)
        )
        
        execution_id = client.execute_workflow(
            workflow, 
            metadata={
                'prompt': data['prompt'],
                'source': 'flask_web_ui',
                'timestamp': time.time()
            }
        )
        
        return jsonify({
            'execution_id': execution_id,
            'status': 'submitted'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<execution_id>')
def get_status(execution_id):
    try:
        status = client.get_workflow_status(execution_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<execution_id>')
def get_results(execution_id):
    try:
        result = client.get_workflow_result(execution_id)
        return jsonify(result.__dict__)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/executions')
def list_executions():
    try:
        # This would typically come from a database
        # For now, return empty list
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<execution_id>/<filename>')
def get_image(execution_id, filename):
    try:
        # In a real implementation, you'd retrieve the image from storage
        # This is a placeholder that would download from the actual URL
        result = client.get_workflow_result(execution_id)
        
        if result.outputs and 'images' in result.outputs:
            for img in result.outputs['images']:
                if img.get('filename') == filename and img.get('url'):
                    # Download and serve the image
                    import requests
                    response = requests.get(img['url'])
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        tmp.write(response.content)
                        return send_file(tmp.name, mimetype='image/png')
        
        return jsonify({'error': 'Image not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### FastAPI WebSocket Integration

```python
# fastapi_websocket.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import time
from typing import Dict, Set
from comfyui_client import AsyncComfyUIClient, WorkflowTemplates

app = FastAPI()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.execution_subscribers: Dict[str, Set[str]] = {}  # execution_id -> set of connection_ids
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Remove from all subscriptions
        for execution_id in list(self.execution_subscribers.keys()):
            if connection_id in self.execution_subscribers[execution_id]:
                self.execution_subscribers[execution_id].discard(connection_id)
                if not self.execution_subscribers[execution_id]:
                    del self.execution_subscribers[execution_id]
    
    def subscribe_to_execution(self, connection_id: str, execution_id: str):
        if execution_id not in self.execution_subscribers:
            self.execution_subscribers[execution_id] = set()
        self.execution_subscribers[execution_id].add(connection_id)
    
    async def send_to_connection(self, connection_id: str, message: dict):
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
            except:
                self.disconnect(connection_id)
    
    async def broadcast_to_execution_subscribers(self, execution_id: str, message: dict):
        if execution_id in self.execution_subscribers:
            for connection_id in list(self.execution_subscribers[execution_id]):
                await self.send_to_connection(connection_id, message)

manager = ConnectionManager()

# Background task to monitor executions
async def monitor_executions():
    """Background task to monitor workflow executions and send updates via WebSocket."""
    monitored_executions = set()
    
    while True:
        try:
            # Check all subscribed executions
            for execution_id in list(manager.execution_subscribers.keys()):
                if execution_id not in monitored_executions:
                    monitored_executions.add(execution_id)
                    # Start monitoring this execution
                    asyncio.create_task(monitor_single_execution(execution_id))
            
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Error in monitor_executions: {e}")
            await asyncio.sleep(10)

async def monitor_single_execution(execution_id: str):
    """Monitor a single execution and send updates."""
    try:
        async with AsyncComfyUIClient(
            api_url=os.getenv('COMFYUI_API_URL'),
            access_token=os.getenv('COMFYUI_TOKEN')
        ) as client:
            
            while True:
                try:
                    # Get execution status
                    async with client.session.get(f"/workflows/{execution_id}/status") as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Send update to subscribers
                            await manager.broadcast_to_execution_subscribers(execution_id, {
                                "type": "execution_update",
                                "execution_id": execution_id,
                                "status": data["status"],
                                "progress": data.get("progress", {}),
                                "timestamp": time.time()
                            })
                            
                            # Stop monitoring if completed
                            if data["status"] in ["completed", "failed", "cancelled"]:
                                # Send final result
                                if data["status"] == "completed":
                                    async with client.session.get(f"/workflows/{execution_id}") as result_response:
                                        if result_response.status == 200:
                                            result_data = await result_response.json()
                                            await manager.broadcast_to_execution_subscribers(execution_id, {
                                                "type": "execution_completed",
                                                "execution_id": execution_id,
                                                "result": result_data,
                                                "timestamp": time.time()
                                            })
                                break
                        else:
                            # Execution not found or error
                            break
                            
                except Exception as e:
                    print(f"Error monitoring execution {execution_id}: {e}")
                    break
                
                await asyncio.sleep(3)
                
    except Exception as e:
        print(f"Error in monitor_single_execution {execution_id}: {e}")

# Start background monitoring
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_executions())

@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    await manager.connect(websocket, connection_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "subscribe_execution":
                execution_id = data["execution_id"]
                manager.subscribe_to_execution(connection_id, execution_id)
                
                await manager.send_to_connection(connection_id, {
                    "type": "subscription_confirmed",
                    "execution_id": execution_id
                })
            
            elif data["type"] == "submit_workflow":
                # Submit new workflow
                try:
                    workflow = WorkflowTemplates.text_to_image(
                        prompt=data["prompt"],
                        negative_prompt=data.get("negative_prompt", "blurry"),
                        seed=data.get("seed", 42),
                        steps=data.get("steps", 20)
                    )
                    
                    async with AsyncComfyUIClient(
                        api_url=os.getenv('COMFYUI_API_URL'),
                        access_token=os.getenv('COMFYUI_TOKEN')
                    ) as client:
                        execution_id = await client.execute_workflow(workflow)
                        
                        # Automatically subscribe to this execution
                        manager.subscribe_to_execution(connection_id, execution_id)
                        
                        await manager.send_to_connection(connection_id, {
                            "type": "workflow_submitted",
                            "execution_id": execution_id,
                            "status": "submitted"
                        })
                        
                except Exception as e:
                    await manager.send_to_connection(connection_id, {
                        "type": "error",
                        "message": str(e)
                    })
            
    except WebSocketDisconnect:
        manager.disconnect(connection_id)

# Serve static files and HTML
@app.get("/")
async def get():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>ComfyUI WebSocket Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1000px; margin: 0 auto; }
        .panel { border: 1px solid #ddd; margin: 10px; padding: 15px; border-radius: 5px; }
        .execution { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
        .progress { width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }
        .progress-bar { height: 100%; background: #4caf50; transition: width 0.3s; }
        .log { background: #1e1e1e; color: #fff; padding: 10px; height: 200px; overflow-y: scroll; font-family: monospace; }
        input, textarea, button { margin: 5px; padding: 8px; }
        button { background: #007cba; color: white; border: none; cursor: pointer; }
        button:hover { background: #005a8b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ComfyUI WebSocket Real-time Demo</h1>
        
        <div class="panel">
            <h2>Generate Image</h2>
            <div>
                <label>Prompt:</label><br>
                <textarea id="prompt" rows="3" cols="50" placeholder="beautiful landscape..."></textarea><br>
                <label>Negative Prompt:</label><br>
                <input type="text" id="negativePrompt" value="blurry, low quality"><br>
                <label>Seed:</label>
                <input type="number" id="seed" value="42">
                <label>Steps:</label>
                <input type="number" id="steps" value="20" min="1" max="50"><br>
                <button onclick="submitWorkflow()">Generate</button>
            </div>
        </div>
        
        <div class="panel">
            <h2>Active Executions</h2>
            <div id="executions"></div>
        </div>
        
        <div class="panel">
            <h2>Connection Log</h2>
            <div id="log" class="log"></div>
        </div>
    </div>
    
    <script>
        const connectionId = 'conn_' + Math.random().toString(36).substr(2, 9);
        const executions = {};
        let ws;
        
        function log(message) {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}\\n`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function connect() {
            ws = new WebSocket(`ws://localhost:8000/ws/${connectionId}`);
            
            ws.onopen = function(event) {
                log('Connected to WebSocket server');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
            
            ws.onclose = function(event) {
                log('WebSocket connection closed');
                setTimeout(connect, 3000); // Reconnect after 3 seconds
            };
            
            ws.onerror = function(error) {
                log('WebSocket error: ' + error);
            };
        }
        
        function handleMessage(data) {
            log(`Received: ${data.type}`);
            
            switch (data.type) {
                case 'workflow_submitted':
                    executions[data.execution_id] = {
                        id: data.execution_id,
                        status: 'pending',
                        progress: 0,
                        prompt: document.getElementById('prompt').value
                    };
                    updateExecutionsDisplay();
                    break;
                    
                case 'execution_update':
                    if (executions[data.execution_id]) {
                        executions[data.execution_id].status = data.status;
                        executions[data.execution_id].progress = data.progress.percentage || 0;
                        updateExecutionsDisplay();
                    }
                    break;
                    
                case 'execution_completed':
                    if (executions[data.execution_id]) {
                        executions[data.execution_id].status = 'completed';
                        executions[data.execution_id].progress = 100;
                        executions[data.execution_id].result = data.result;
                        updateExecutionsDisplay();
                    }
                    break;
                    
                case 'error':
                    log('Error: ' + data.message);
                    break;
            }
        }
        
        function submitWorkflow() {
            const payload = {
                type: 'submit_workflow',
                prompt: document.getElementById('prompt').value,
                negative_prompt: document.getElementById('negativePrompt').value,
                seed: parseInt(document.getElementById('seed').value),
                steps: parseInt(document.getElementById('steps').value)
            };
            
            ws.send(JSON.stringify(payload));
            log('Submitted workflow');
        }
        
        function updateExecutionsDisplay() {
            const container = document.getElementById('executions');
            container.innerHTML = '';
            
            Object.values(executions).reverse().forEach(execution => {
                const div = document.createElement('div');
                div.className = 'execution';
                
                let html = `
                    <h3>Execution: ${execution.id.substring(0, 8)}...</h3>
                    <p><strong>Status:</strong> ${execution.status}</p>
                    <p><strong>Prompt:</strong> ${execution.prompt}</p>
                `;
                
                if (execution.status === 'running' && execution.progress > 0) {
                    html += `
                        <div class="progress">
                            <div class="progress-bar" style="width: ${execution.progress}%"></div>
                        </div>
                        <p>Progress: ${execution.progress}%</p>
                    `;
                }
                
                if (execution.result && execution.result.outputs && execution.result.outputs.images) {
                    html += '<h4>Generated Images:</h4>';
                    execution.result.outputs.images.forEach(img => {
                        html += `<p>📷 ${img.filename} - <a href="${img.url}" target="_blank">View</a></p>`;
                    });
                }
                
                div.innerHTML = html;
                container.appendChild(div);
            });
        }
        
        // Connect on page load
        connect();
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

This comprehensive SDK documentation provides complete examples for integrating with the ComfyUI Serverless API across multiple languages and platforms. The examples include error handling, authentication management, progress monitoring, and real-time updates through WebSockets.