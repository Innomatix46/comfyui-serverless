"""Workflow management API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime, timedelta

from src.core.database import get_db
from src.models.schemas import (
    WorkflowExecutionRequest,
    WorkflowExecutionResponse, 
    WorkflowResult,
    WorkflowStatus,
    Priority
)
from src.models.database import User, WorkflowExecution
from src.services.workflow import workflow_service
from src.services.auth import get_current_user
from src.utils.pagination import paginate

router = APIRouter()


@router.post("/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Execute a ComfyUI workflow asynchronously."""
    try:
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Create workflow execution record
        execution = WorkflowExecution(
            id=execution_id,
            user_id=current_user.id,
            workflow_definition=request.workflow.dict(),
            priority=request.priority,
            status=WorkflowStatus.PENDING,
            webhook_url=request.webhook_url,
            execution_metadata=request.metadata,
            timeout_at=datetime.utcnow() + timedelta(minutes=request.timeout_minutes)
        )
        
        db.add(execution)
        db.commit()
        db.refresh(execution)
        
        # Submit to workflow service
        queue_position = await workflow_service.submit_workflow(
            execution_id=execution_id,
            workflow=request.workflow,
            priority=request.priority,
            user_id=current_user.id,
            webhook_url=request.webhook_url,
            timeout_minutes=request.timeout_minutes
        )
        
        # Update queue position
        execution.queue_position = queue_position
        db.commit()
        
        return WorkflowExecutionResponse(
            execution_id=execution_id,
            status=WorkflowStatus.PENDING,
            created_at=execution.created_at,
            estimated_duration=await workflow_service.estimate_duration(request.workflow),
            queue_position=queue_position
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute workflow: {str(e)}"
        )


@router.get("/{execution_id}", response_model=WorkflowResult)
async def get_workflow_result(
    execution_id: str = Path(..., description="Workflow execution ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get workflow execution result."""
    execution = db.query(WorkflowExecution).filter(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.user_id == current_user.id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=404,
            detail="Workflow execution not found"
        )
    
    return WorkflowResult(
        execution_id=execution.id,
        status=execution.status,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        duration_seconds=(
            (execution.completed_at - execution.started_at).total_seconds()
            if execution.started_at and execution.completed_at else None
        ),
        outputs=execution.outputs,
        error=execution.error_message,
        logs=execution.logs or [],
        metadata=execution.execution_metadata
    )


@router.get("/", response_model=List[WorkflowResult])
async def list_workflow_executions(
    status: Optional[WorkflowStatus] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user's workflow executions."""
    query = db.query(WorkflowExecution).filter(
        WorkflowExecution.user_id == current_user.id
    )
    
    if status:
        query = query.filter(WorkflowExecution.status == status)
    
    executions = query.order_by(
        WorkflowExecution.created_at.desc()
    ).offset(offset).limit(limit).all()
    
    return [
        WorkflowResult(
            execution_id=execution.id,
            status=execution.status,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_seconds=(
                (execution.completed_at - execution.started_at).total_seconds()
                if execution.started_at and execution.completed_at else None
            ),
            outputs=execution.outputs,
            error=execution.error_message,
            logs=execution.logs or [],
            metadata=execution.execution_metadata
        )
        for execution in executions
    ]


@router.post("/{execution_id}/cancel")
async def cancel_workflow(
    execution_id: str = Path(..., description="Workflow execution ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel a running or pending workflow."""
    execution = db.query(WorkflowExecution).filter(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.user_id == current_user.id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=404,
            detail="Workflow execution not found"
        )
    
    if execution.status not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel workflow in {execution.status} status"
        )
    
    # Cancel the workflow
    success = await workflow_service.cancel_workflow(execution_id)
    
    if success:
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        db.commit()
        
        return {"message": "Workflow cancelled successfully"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel workflow"
        )


@router.get("/{execution_id}/status")
async def get_workflow_status(
    execution_id: str = Path(..., description="Workflow execution ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get workflow execution status and progress."""
    execution = db.query(WorkflowExecution).filter(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.user_id == current_user.id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=404,
            detail="Workflow execution not found"
        )
    
    # Get live progress from workflow service
    progress = await workflow_service.get_progress(execution_id)
    
    return {
        "execution_id": execution.id,
        "status": execution.status,
        "created_at": execution.created_at,
        "started_at": execution.started_at,
        "queue_position": execution.queue_position,
        "progress": progress,
        "estimated_completion": (
            datetime.utcnow() + timedelta(seconds=progress.get("eta_seconds", 0))
            if progress and progress.get("eta_seconds") else None
        )
    }


@router.get("/{execution_id}/logs")
async def get_workflow_logs(
    execution_id: str = Path(..., description="Workflow execution ID"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    tail: int = Query(100, ge=1, le=1000, description="Number of recent logs"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get workflow execution logs."""
    execution = db.query(WorkflowExecution).filter(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.user_id == current_user.id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=404,
            detail="Workflow execution not found"
        )
    
    # Get logs from workflow service (real-time) or database (completed)
    if execution.status in [WorkflowStatus.RUNNING, WorkflowStatus.PENDING]:
        logs = await workflow_service.get_live_logs(execution_id, level=level, tail=tail)
    else:
        logs = execution.logs or []
        if level:
            logs = [log for log in logs if log.get("level") == level.upper()]
        logs = logs[-tail:] if tail else logs
    
    return {
        "execution_id": execution.id,
        "logs": logs,
        "total_count": len(logs),
        "filtered": bool(level)
    }


@router.post("/{execution_id}/retry")
async def retry_workflow(
    execution_id: str = Path(..., description="Workflow execution ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retry a failed workflow execution."""
    execution = db.query(WorkflowExecution).filter(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.user_id == current_user.id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=404,
            detail="Workflow execution not found"
        )
    
    if execution.status != WorkflowStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed workflows, current status: {execution.status}"
        )
    
    # Create new execution with same parameters
    new_execution_id = str(uuid.uuid4())
    
    new_execution = WorkflowExecution(
        id=new_execution_id,
        user_id=current_user.id,
        workflow_definition=execution.workflow_definition,
        priority=execution.priority,
        status=WorkflowStatus.PENDING,
        webhook_url=execution.webhook_url,
        execution_metadata={
            **(execution.execution_metadata or {}),
            "retried_from": execution_id,
            "retry_count": (execution.execution_metadata or {}).get("retry_count", 0) + 1
        },
        timeout_at=datetime.utcnow() + timedelta(minutes=30)
    )
    
    db.add(new_execution)
    db.commit()
    db.refresh(new_execution)
    
    # Submit to workflow service
    queue_position = await workflow_service.submit_workflow(
        execution_id=new_execution_id,
        workflow=execution.workflow_definition,
        priority=execution.priority,
        user_id=current_user.id,
        webhook_url=execution.webhook_url
    )
    
    new_execution.queue_position = queue_position
    db.commit()
    
    return {
        "execution_id": new_execution_id,
        "status": WorkflowStatus.PENDING,
        "queue_position": queue_position,
        "retried_from": execution_id
    }
