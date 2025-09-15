"""Metrics and monitoring API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional

from src.core.database import get_db
from src.models.schemas import SystemMetrics, ExecutionMetrics
from src.models.database import User
from src.services.monitoring import monitoring_service
from src.services.auth import get_current_user

router = APIRouter()


@router.get("/prometheus", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    try:
        metrics_text = await monitoring_service.export_prometheus_metrics()
        return metrics_text
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export metrics: {str(e)}"
        )


@router.get("/system", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get current system metrics."""
    try:
        metrics = await monitoring_service.get_system_metrics()
        
        return SystemMetrics(
            cpu_usage_percent=metrics.get("cpu_usage_percent", 0.0),
            memory_usage_percent=metrics.get("memory_usage_percent", 0.0),
            gpu_usage_percent=metrics.get("gpu_usage_percent"),
            gpu_memory_usage_percent=metrics.get("gpu_memory_usage_percent"),
            disk_usage_percent=metrics.get("disk_usage_percent", 0.0),
            active_executions=metrics.get("active_executions", 0),
            queue_size=metrics.get("queue_size", 0),
            total_executions=metrics.get("total_executions", 0),
            average_execution_time_seconds=metrics.get("average_execution_time_seconds", 0.0)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system metrics: {str(e)}"
        )


@router.get("/executions", response_model=ExecutionMetrics)
async def get_execution_metrics(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    current_user: User = Depends(get_current_user)
):
    """Get workflow execution metrics."""
    try:
        metrics = await monitoring_service.get_execution_metrics(
            start_time=datetime.utcnow() - timedelta(hours=hours),
            end_time=datetime.utcnow()
        )
        
        return ExecutionMetrics(
            total_executions=metrics.get("total_executions", 0),
            completed_executions=metrics.get("completed_executions", 0),
            failed_executions=metrics.get("failed_executions", 0),
            average_duration_seconds=metrics.get("average_duration_seconds", 0.0),
            executions_per_minute=metrics.get("executions_per_minute", 0.0),
            queue_wait_time_seconds=metrics.get("queue_wait_time_seconds", 0.0)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get execution metrics: {str(e)}"
        )


@router.get("/usage")
async def get_usage_statistics(
    days: int = Query(7, ge=1, le=30, description="Time window in days"),
    current_user: User = Depends(get_current_user)
):
    """Get API usage statistics for the user."""
    try:
        usage_stats = await monitoring_service.get_user_usage_stats(
            user_id=current_user.id,
            days=days
        )
        
        return {
            "user_id": current_user.id,
            "period_days": days,
            "statistics": usage_stats,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get usage statistics: {str(e)}"
        )


@router.get("/performance")
async def get_performance_metrics(
    component: Optional[str] = Query(None, description="Filter by component"),
    current_user: User = Depends(get_current_user)
):
    """Get detailed performance metrics."""
    try:
        performance_data = await monitoring_service.get_performance_metrics(component)
        
        return {
            "component": component or "all",
            "metrics": performance_data,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.get("/alerts")
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    current_user: User = Depends(get_current_user)
):
    """Get active system alerts."""
    try:
        alerts = await monitoring_service.get_active_alerts(severity)
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alerts: {str(e)}"
        )


@router.get("/capacity")
async def get_capacity_info(
    current_user: User = Depends(get_current_user)
):
    """Get system capacity and resource utilization."""
    try:
        capacity_info = await monitoring_service.get_capacity_info()
        
        return {
            "capacity": capacity_info,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get capacity info: {str(e)}"
        )


@router.get("/trends")
async def get_performance_trends(
    metric: str = Query(..., description="Metric name"),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    current_user: User = Depends(get_current_user)
):
    """Get performance trends for a specific metric."""
    try:
        trends = await monitoring_service.get_performance_trends(
            metric=metric,
            start_time=datetime.utcnow() - timedelta(hours=hours),
            end_time=datetime.utcnow()
        )
        
        return {
            "metric": metric,
            "period_hours": hours,
            "trends": trends,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trends: {str(e)}"
        )


@router.post("/reset")
async def reset_metrics(
    confirm: bool = Query(False, description="Confirm reset operation"),
    current_user: User = Depends(get_current_user)
):
    """Reset metrics counters (admin only)."""
    # Note: In a real implementation, you'd check for admin privileges
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Please set confirm=true to reset metrics"
        )
    
    try:
        await monitoring_service.reset_metrics()
        
        return {
            "message": "Metrics reset successfully",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset metrics: {str(e)}"
        )