"""ComfyUI integration client."""
import asyncio
import json
import uuid
from typing import Dict, Any, Optional, List
import aiohttp
import structlog
from datetime import datetime

from src.config.settings import settings

logger = structlog.get_logger()


class ComfyUIError(Exception):
    """Custom error for ComfyUI client operations."""
    pass


class ComfyUIClient:
    """Client for interacting with ComfyUI API."""

    def __init__(self, base_url: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None):
        self.base_url = base_url or settings.COMFYUI_API_URL
        self.websocket_url = self.base_url.replace("http", "ws") + "/ws"
        self._session: Optional[aiohttp.ClientSession] = session
        self._ws_connection = None

    async def __aenter__(self):
        """Async context manager entry."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def health_check(self) -> bool:
        """Check if ComfyUI is healthy and responsive."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
            async with self._session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error("ComfyUI health check failed", error=str(e))
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status from ComfyUI."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            async with self._session.get(f"{self.base_url}/queue") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error("Failed to get queue status", status=response.status)
                        return {}
        except Exception as e:
            logger.error("Error getting queue status", error=str(e))
            return {}
    
    async def execute_workflow(
        self,
        execution_id: str,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow on ComfyUI and return results."""
        try:
            logger.info("Starting workflow execution", execution_id=execution_id)
            
            # Validate workflow format
            if not self._validate_workflow(workflow_data):
                raise ComfyUIError("Invalid workflow format")
            
            # Convert workflow to ComfyUI format
            comfyui_workflow = self._convert_to_comfyui_format(workflow_data)
            
            # Submit workflow to ComfyUI queue
            prompt_id, submit_payload = await self._queue_workflow(comfyui_workflow)
            # If ComfyUI responded with node_errors, report them
            if submit_payload and submit_payload.get("node_errors"):
                raise ComfyUIError("Node errors in workflow")
            if not prompt_id:
                raise ComfyUIError("Failed to queue workflow in ComfyUI")
            
            logger.info("Workflow queued", execution_id=execution_id, prompt_id=prompt_id)
            
            # Monitor execution progress
            result = await self._monitor_execution(execution_id, prompt_id)
            
            logger.info("Workflow execution completed", execution_id=execution_id)
            return result
            
        except Exception as e:
            logger.error("Workflow execution failed", execution_id=execution_id, error=str(e))
            raise
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            async with self._session.post(
                f"{self.base_url}/interrupt",
                json={"execution_id": execution_id}
            ) as response:
                if response.status == 200:
                    logger.info("Execution cancelled", execution_id=execution_id)
                    return True
            return False
            
        except Exception as e:
            logger.error("Failed to cancel execution", execution_id=execution_id, error=str(e))
            return False
    
    async def get_execution_progress(self, prompt_id: str) -> Dict[str, Any]:
        """Get progress information for an execution."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            async with self._session.get(f"{self.base_url}/history/{prompt_id}") as response:
                    if response.status == 200:
                        history = await response.json()
                        if prompt_id in history:
                            return self._parse_progress(history[prompt_id])
                    
            return {"progress": 0.0, "status": "unknown"}
            
        except Exception as e:
            logger.error("Failed to get execution progress", prompt_id=prompt_id, error=str(e))
            return {"progress": 0.0, "status": "error", "error": str(e)}
    
    async def get_models_list(self) -> Dict[str, List[str]]:
        """Get list of available models from ComfyUI."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/object_info") as response:
                    if response.status == 200:
                        object_info = await response.json()
                        return self._extract_models_from_object_info(object_info)
                    
            return {}
            
        except Exception as e:
            logger.error("Failed to get models list", error=str(e))
            return {}
    
    async def interrupt_execution(self) -> bool:
        """Interrupt currently running execution."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            async with self._session.post(f"{self.base_url}/interrupt") as response:
                    if response.status == 200:
                        logger.info("Execution interrupted successfully")
                        return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to interrupt execution", error=str(e))
            return False
    
    def _validate_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """Validate workflow data format."""
        try:
            # Check if workflow has nodes
            if "nodes" not in workflow_data:
                return False
            
            nodes = workflow_data["nodes"]
            if not isinstance(nodes, dict) or not nodes:
                return False
            
            # Validate each node has required fields
            for node_id, node_data in nodes.items():
                if not isinstance(node_data, dict):
                    return False
                
                if "class_type" not in node_data:
                    return False
                
                if "inputs" not in node_data:
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Workflow validation error", error=str(e))
            return False
    
    def _convert_to_comfyui_format(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert our workflow format to ComfyUI format."""
        try:
            comfyui_workflow = {}
            
            for node_id, node_data in workflow_data["nodes"].items():
                # Convert inputs from our format to ComfyUI format
                inputs = {}
                # Accept either list-of-dicts or dict mapping
                raw_inputs = node_data.get("inputs", {})
                if isinstance(raw_inputs, dict):
                    inputs = raw_inputs
                else:
                    for input_item in raw_inputs:
                        inputs[input_item["name"]] = input_item["value"]
                
                comfyui_workflow[node_id] = {
                    "class_type": node_data["class_type"],
                    "inputs": inputs
                }
            
            return comfyui_workflow
            
        except Exception as e:
            logger.error("Workflow conversion error", error=str(e))
            raise
    
    async def _queue_workflow(self, workflow: Dict[str, Any]) -> (Optional[str], Optional[Dict[str, Any]]):
        """Submit workflow to ComfyUI queue."""
        try:
            prompt_id = str(uuid.uuid4())
            
            queue_data = {
                "prompt": workflow,
                "client_id": prompt_id
            }
            
            if self._session is None:
                self._session = aiohttp.ClientSession()
            async with self._session.post(
                f"{self.base_url}/prompt",
                json=queue_data
            ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("prompt_id", prompt_id), result
                    else:
                        logger.error("Failed to queue workflow", status=response.status)
                        return None, None
                        
        except Exception as e:
            logger.error("Error queueing workflow", error=str(e))
            return None, None
    
    async def _monitor_execution(
        self, 
        execution_id: str, 
        prompt_id: str,
        timeout_seconds: int = 1800  # 30 minutes
    ) -> Dict[str, Any]:
        """Monitor execution progress and return results."""
        start_time = datetime.utcnow()
        logs = []
        
        try:
            # Connect to WebSocket for real-time updates
            ws_task = asyncio.create_task(
                self._monitor_websocket(execution_id, prompt_id, logs)
            )
            
            # Poll for completion and return raw outputs for compatibility with tests
            while True:
                # Check timeout
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    ws_task.cancel()
                    raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds")
                
                # Check execution status
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/history/{prompt_id}") as response:
                        if response.status == 200:
                            history = await response.json()
                            
                            if prompt_id in history:
                                execution_data = history[prompt_id]
                                
                                # Check if execution is complete
                                if execution_data.get("status", {}).get("completed", False):
                                    ws_task.cancel()
                                    
                                    # Return outputs as provided by ComfyUI history
                                    return {
                                        "outputs": execution_data.get("outputs", {}),
                                        "logs": logs,
                                        "execution_time": elapsed,
                                        "status": "completed"
                                    }
                                
                                # Check for errors
                                if "error" in execution_data.get("status", {}):
                                    ws_task.cancel()
                                    error_info = execution_data["status"]["error"]
                                    
                                    return {
                                        "error": error_info,
                                        "logs": logs,
                                        "execution_time": elapsed,
                                        "status": "failed"
                                    }
                
                # Wait before next check
                await asyncio.sleep(2)
                
        except asyncio.CancelledError:
            logger.info("Execution monitoring cancelled", execution_id=execution_id)
            raise
        except Exception as e:
            logger.error("Error monitoring execution", execution_id=execution_id, error=str(e))
            raise
    
    async def _monitor_websocket(
        self, 
        execution_id: str, 
        prompt_id: str, 
        logs: List[Dict[str, Any]]
    ):
        """Monitor WebSocket for real-time updates."""
        try:
            import websockets
            
            async with websockets.connect(f"{self.websocket_url}?clientId={prompt_id}") as websocket:
                logger.info("WebSocket connected for monitoring", execution_id=execution_id)
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Log progress updates
                        if data.get("type") == "progress":
                            logs.append({
                                "timestamp": datetime.utcnow().isoformat(),
                                "level": "INFO",
                                "message": f"Progress: {data.get('data', {})}",
                                "component": "comfyui"
                            })
                        
                        # Log execution updates
                        elif data.get("type") == "executing":
                            node_id = data.get("data", {}).get("node")
                            if node_id:
                                logs.append({
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "level": "INFO",
                                    "message": f"Executing node: {node_id}",
                                    "component": "comfyui"
                                })
                        
                        # Log other events
                        else:
                            logs.append({
                                "timestamp": datetime.utcnow().isoformat(),
                                "level": "DEBUG",
                                "message": f"WebSocket message: {data.get('type', 'unknown')}",
                                "component": "comfyui"
                            })
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error("WebSocket monitoring error", execution_id=execution_id, error=str(e))
    
    def _extract_outputs(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract outputs from ComfyUI execution data."""
        try:
            outputs = {}
            
            # Extract generated images
            if "outputs" in execution_data:
                for node_id, node_output in execution_data["outputs"].items():
                    if "images" in node_output:
                        images = []
                        for img_data in node_output["images"]:
                            images.append({
                                "filename": img_data.get("filename"),
                                "subfolder": img_data.get("subfolder", ""),
                                "type": img_data.get("type", "output")
                            })
                        outputs[f"node_{node_id}_images"] = images
                    
                    # Extract other outputs
                    for key, value in node_output.items():
                        if key != "images":
                            outputs[f"node_{node_id}_{key}"] = value
            
            return outputs
            
        except Exception as e:
            logger.error("Error extracting outputs", error=str(e))
            return {}
    
    def _parse_progress(self, history_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse progress information from history data."""
        try:
            status_data = history_data.get("status", {})
            
            if status_data.get("completed", False):
                return {"progress": 1.0, "status": "completed"}
            
            if "error" in status_data:
                return {"progress": 0.0, "status": "failed", "error": status_data["error"]}
            
            # Estimate progress based on executed nodes
            if "outputs" in history_data:
                executed_nodes = len(history_data["outputs"])
                # This is a rough estimation - in practice you'd need workflow analysis
                total_nodes = executed_nodes + 5  # Assume some remaining nodes
                progress = min(0.9, executed_nodes / total_nodes)  # Cap at 90% until complete
                
                return {"progress": progress, "status": "running", "executed_nodes": executed_nodes}
            
            return {"progress": 0.1, "status": "starting"}
            
        except Exception as e:
            logger.error("Error parsing progress", error=str(e))
            return {"progress": 0.0, "status": "unknown"}
    
    def _extract_models_from_object_info(self, object_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract available models from ComfyUI object info."""
        try:
            models = {
                "checkpoints": [],
                "loras": [],
                "vaes": [],
                "controlnets": [],
                "upscalers": []
            }
            
            for class_name, class_info in object_info.items():
                inputs = class_info.get("input", {})
                required = inputs.get("required", {})
                
                # Extract checkpoint models
                if "ckpt_name" in required:
                    models["checkpoints"].extend(required["ckpt_name"][0])
                
                # Extract LoRA models
                if "lora_name" in required:
                    models["loras"].extend(required["lora_name"][0])
                
                # Extract VAE models
                if "vae_name" in required:
                    models["vaes"].extend(required["vae_name"][0])
                
                # Extract ControlNet models
                if "control_net_name" in required:
                    models["controlnets"].extend(required["control_net_name"][0])
                
                # Extract upscaler models
                if "model_name" in required and "upscale" in class_name.lower():
                    models["upscalers"].extend(required["model_name"][0])
            
            # Remove duplicates
            for key in models:
                models[key] = list(set(models[key]))
            
            return models
            
        except Exception as e:
            logger.error("Error extracting models", error=str(e))
            return {}
