"""Workflow execution service."""
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import structlog
from celery import Celery
import redis
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.core.database import get_db, SessionLocal
from src.models.database import WorkflowExecution, WorkflowStatus, Priority
from src.models.schemas import WorkflowDefinition
from src.services.comfyui import ComfyUIClient
from src.services.model import model_service
from src.services.storage import storage_service
from src.services.webhook import webhook_service
from src.utils.validation import validate_workflow

logger = structlog.get_logger()

# Initialize Celery
celery_app = Celery(
    "workflow",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_routes=settings.CELERY_TASK_ROUTES,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True
)


class WorkflowService:
    """Workflow execution service."""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.comfyui_client = ComfyUIClient()
        self._queue_counters = {}
    
    async def submit_workflow(
        self,
        execution_id: str,
        workflow: WorkflowDefinition,
        priority: Priority = Priority.NORMAL,
        user_id: int = None,
        webhook_url: Optional[str] = None,
        timeout_minutes: int = 30
    ) -> int:
        """Submit workflow for execution."""
        try:
            # Validate workflow
            validation_result = await validate_workflow(workflow)
            if not validation_result.is_valid:
                raise ValueError(f"Invalid workflow: {validation_result.errors}")
            
            # Check required models
            required_models = self._extract_required_models(workflow)
            for model_name, model_type in required_models.items():
                if not await model_service.is_model_available(model_name, model_type):
                    # Queue model download
                    await model_service.queue_download(model_name, model_type)
            
            # Submit to Celery queue
            task = execute_workflow_task.apply_async(
                args=[execution_id],
                kwargs={
                    "workflow_data": workflow.dict(),
                    "user_id": user_id,
                    "webhook_url": webhook_url,
                    "timeout_minutes": timeout_minutes
                },
                queue=self._get_queue_name(priority),
                countdown=0 if priority == Priority.HIGH else 1
            )
            
            # Update execution with task ID
            with SessionLocal() as db:
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.id == execution_id
                ).first()
                if execution:
                    execution.execution_metadata = {
                        **(execution.execution_metadata or {}),
                        "celery_task_id": task.id
                    }
                    db.commit()
            
            # Get queue position
            queue_position = await self._get_queue_position(execution_id, priority)
            
            logger.info(
                "Workflow submitted",
                execution_id=execution_id,
                task_id=task.id,
                queue_position=queue_position,
                priority=priority
            )
            
            return queue_position
            
        except Exception as e:
            logger.error(
                "Failed to submit workflow",
                execution_id=execution_id,
                error=str(e)
            )
            raise
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a workflow execution."""
        try:
            with SessionLocal() as db:
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.id == execution_id
                ).first()
                
                if not execution:
                    return False
                
                # Get Celery task ID
                task_id = execution.execution_metadata.get("celery_task_id") if execution.execution_metadata else None
                
                if task_id:
                    # Cancel Celery task
                    celery_app.control.revoke(task_id, terminate=True)
                
                # Cancel in ComfyUI if running
                if execution.status == WorkflowStatus.RUNNING:
                    await self.comfyui_client.cancel_execution(execution_id)
                
                # Update status
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.utcnow()
                db.commit()
                
                logger.info("Workflow cancelled", execution_id=execution_id)
                return True
                
        except Exception as e:
            logger.error(
                "Failed to cancel workflow",
                execution_id=execution_id,
                error=str(e)
            )
            return False
    
    async def get_progress(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution progress."""
        try:
            # Check Redis for real-time progress
            progress_key = f"workflow:progress:{execution_id}"
            progress_data = self.redis_client.get(progress_key)
            
            if progress_data:
                return json.loads(progress_data)
            
            # Fall back to database status
            with SessionLocal() as db:
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.id == execution_id
                ).first()
                
                if not execution:
                    return {}
                
                base_progress = {
                    "status": execution.status,
                    "queue_position": execution.queue_position,
                    "created_at": execution.created_at.isoformat() if execution.created_at else None,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None
                }
                
                # Add progress estimates
                if execution.status == WorkflowStatus.PENDING:
                    estimated_wait = await self._estimate_queue_wait_time(execution.queue_position)
                    base_progress["estimated_wait_seconds"] = estimated_wait
                
                elif execution.status == WorkflowStatus.RUNNING:
                    elapsed = (datetime.utcnow() - execution.started_at).total_seconds() if execution.started_at else 0
                    estimated_total = await self.estimate_duration(execution.workflow_definition)
                    
                    base_progress.update({
                        "elapsed_seconds": elapsed,
                        "estimated_total_seconds": estimated_total,
                        "progress_percent": min(90, (elapsed / estimated_total * 100)) if estimated_total > 0 else 0,
                        "eta_seconds": max(0, estimated_total - elapsed) if estimated_total > elapsed else 0
                    })
                
                return base_progress
                
        except Exception as e:
            logger.error(
                "Failed to get workflow progress",
                execution_id=execution_id,
                error=str(e)
            )
            return {}
    
    async def get_live_logs(
        self, 
        execution_id: str, 
        level: Optional[str] = None, 
        tail: int = 100
    ) -> List[Dict[str, Any]]:
        """Get live workflow execution logs."""
        try:
            logs_key = f"workflow:logs:{execution_id}"
            log_entries = self.redis_client.lrange(logs_key, -tail, -1)
            
            logs = []
            for entry in log_entries:
                try:
                    log_data = json.loads(entry.decode("utf-8"))
                    if level and log_data.get("level", "").upper() != level.upper():
                        continue
                    logs.append(log_data)
                except json.JSONDecodeError:
                    continue
            
            return logs
            
        except Exception as e:
            logger.error(
                "Failed to get live logs",
                execution_id=execution_id,
                error=str(e)
            )
            return []
    
    async def estimate_duration(self, workflow: Dict[str, Any]) -> int:
        """Estimate workflow execution duration in seconds."""
        try:
            # Basic estimation based on node count and types
            nodes = workflow.get("nodes", {})
            node_count = len(nodes)
            
            # Base time per node
            base_time = node_count * 2  # 2 seconds per node
            
            # Add time for specific node types
            heavy_nodes = [
                "KSampler", "KSamplerAdvanced", "ESRGAN_UPSCALER",
                "ControlNetApply", "VAEDecode", "VAEEncode"
            ]
            
            for node_id, node_data in nodes.items():
                class_type = node_data.get("class_type", "")
                if class_type in heavy_nodes:
                    base_time += 30  # Add 30 seconds for heavy operations
                
                # Add time based on image size
                inputs = node_data.get("inputs", {})
                width = inputs.get("width", 512)
                height = inputs.get("height", 512)
                
                if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                    pixel_factor = (width * height) / (512 * 512)
                    base_time += int(pixel_factor * 10)  # Scale with resolution
            
            return max(30, min(base_time, 1800))  # Between 30 seconds and 30 minutes
            
        except Exception:
            return 300  # Default 5 minutes
    
    def _extract_required_models(self, workflow: WorkflowDefinition) -> Dict[str, str]:
        """Extract required models from workflow."""
        required_models = {}
        
        for node_id, node in workflow.nodes.items():
            # Check for checkpoint models
            if node.class_type in ["CheckpointLoaderSimple", "CheckpointLoader"]:
                for input_item in node.inputs:
                    if input_item.name == "ckpt_name":
                        required_models[input_item.value] = "checkpoint"
            
            # Check for LoRA models
            elif node.class_type in ["LoraLoader", "LoRALoader"]:
                for input_item in node.inputs:
                    if input_item.name == "lora_name":
                        required_models[input_item.value] = "lora"
            
            # Check for VAE models
            elif node.class_type in ["VAELoader"]:
                for input_item in node.inputs:
                    if input_item.name == "vae_name":
                        required_models[input_item.value] = "vae"
        
        return required_models
    
    def _get_queue_name(self, priority: Priority) -> str:
        """Get queue name based on priority."""
        priority_map = {
            Priority.HIGH: "workflow_high",
            Priority.NORMAL: "workflow",
            Priority.LOW: "workflow_low"
        }
        return priority_map.get(priority, "workflow")
    
    async def _get_queue_position(self, execution_id: str, priority: Priority) -> int:
        """Get position in execution queue."""
        queue_name = self._get_queue_name(priority)
        
        # Get active tasks from Celery
        active_tasks = celery_app.control.inspect().active()
        reserved_tasks = celery_app.control.inspect().reserved()
        
        # Count tasks in queue
        position = 0
        for worker_tasks in (active_tasks or {}).values():
            position += len([task for task in worker_tasks if task.get("queue") == queue_name])
        
        for worker_tasks in (reserved_tasks or {}).values():
            position += len([task for task in worker_tasks if task.get("queue") == queue_name])
        
        return position + 1
    
    async def _estimate_queue_wait_time(self, queue_position: int) -> int:
        """Estimate wait time in queue."""
        # Assume average execution time of 5 minutes
        average_execution_time = 300
        return queue_position * average_execution_time


# Celery task for workflow execution
@celery_app.task(bind=True, name="workflow.execute")
def execute_workflow_task(
    self,
    execution_id: str,
    workflow_data: Dict[str, Any],
    user_id: int,
    webhook_url: Optional[str] = None,
    timeout_minutes: int = 30
):
    """Execute workflow task."""
    start_time = datetime.utcnow()
    
    logger.info(
        "Starting workflow execution",
        execution_id=execution_id,
        task_id=self.request.id
    )
    
    # Update status to running
    with SessionLocal() as db:
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.id == execution_id
        ).first()
        
        if not execution:
            logger.error("Execution not found", execution_id=execution_id)
            return
        
        execution.status = WorkflowStatus.RUNNING
        execution.started_at = start_time
        execution.worker_id = self.request.hostname
        db.commit()
    
    try:
        # Execute workflow with ComfyUI
        comfyui_client = ComfyUIClient()
        result = asyncio.run(
            comfyui_client.execute_workflow(execution_id, workflow_data)
        )
        
        # Update execution with results
        with SessionLocal() as db:
            execution = db.query(WorkflowExecution).filter(
                WorkflowExecution.id == execution_id
            ).first()
            
            if execution:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                execution.outputs = result.get("outputs", {})
                execution.logs = result.get("logs", [])
                db.commit()
        
        # Send webhook notification
        if webhook_url:
            asyncio.run(
                webhook_service.send_completion_webhook(
                    webhook_url, execution_id, True, result
                )
            )
        
        logger.info(
            "Workflow execution completed",
            execution_id=execution_id,
            duration_seconds=(datetime.utcnow() - start_time).total_seconds()
        )
        
    except Exception as e:
        logger.error(
            "Workflow execution failed",
            execution_id=execution_id,
            error=str(e)
        )
        
        # Update execution with error
        with SessionLocal() as db:
            execution = db.query(WorkflowExecution).filter(
                WorkflowExecution.id == execution_id
            ).first()
            
            if execution:
                execution.status = WorkflowStatus.FAILED
                execution.completed_at = datetime.utcnow()
                execution.error_message = str(e)
                db.commit()
        
        # Send webhook notification
        if webhook_url:
            asyncio.run(
                webhook_service.send_completion_webhook(
                    webhook_url, execution_id, False, {"error": str(e)}
                )
            )
        
        raise


# Global workflow service instance
workflow_service = WorkflowService()
