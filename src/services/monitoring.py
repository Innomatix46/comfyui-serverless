"""Monitoring and metrics service."""
import asyncio
import time
import json
import psutil
import structlog
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import func
from sqlalchemy.orm import Session
import redis
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest

from src.config.settings import settings
from src.core.database import SessionLocal
from src.models.database import WorkflowExecution, WorkflowStatus, SystemMetrics, ExecutionLog
from src.utils.gpu import get_gpu_memory_info, get_gpu_utilization

logger = structlog.get_logger()


class MonitoringService:
    """Service for system monitoring and metrics collection."""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.start_time = time.time()
        self.is_running = False
        self.monitoring_task = None
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Set up Prometheus metrics."""
        # System metrics
        self.cpu_usage_gauge = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage_gauge = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.gpu_usage_gauge = Gauge(
            'system_gpu_usage_percent',
            'GPU usage percentage',
            registry=self.registry
        )
        
        self.gpu_memory_gauge = Gauge(
            'system_gpu_memory_usage_percent',
            'GPU memory usage percentage',
            registry=self.registry
        )
        
        # Application metrics
        self.active_executions_gauge = Gauge(
            'workflow_active_executions',
            'Number of active workflow executions',
            registry=self.registry
        )
        
        self.queue_size_gauge = Gauge(
            'workflow_queue_size',
            'Size of workflow execution queue',
            registry=self.registry
        )
        
        self.execution_duration_histogram = Histogram(
            'workflow_execution_duration_seconds',
            'Workflow execution duration in seconds',
            registry=self.registry
        )
        
        self.execution_counter = Counter(
            'workflow_executions_total',
            'Total number of workflow executions',
            ['status'],
            registry=self.registry
        )
        
        self.api_requests_counter = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_request_duration_histogram = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
    
    def start(self):
        """Start monitoring service."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring service started")
    
    def stop(self):
        """Stop monitoring service."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_metrics(self):
        """Collect and store system metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get GPU metrics if available
            gpu_info = get_gpu_memory_info()
            gpu_util = get_gpu_utilization()
            
            # Get application metrics
            app_metrics = await self._get_application_metrics()
            
            # Update Prometheus metrics
            self.cpu_usage_gauge.set(cpu_percent)
            self.memory_usage_gauge.set(memory.percent)
            
            if gpu_util:
                self.gpu_usage_gauge.set(gpu_util.get('utilization', 0))
            
            if gpu_info:
                gpu_memory_percent = (gpu_info['used_mb'] / gpu_info['total_mb']) * 100
                self.gpu_memory_gauge.set(gpu_memory_percent)
            
            self.active_executions_gauge.set(app_metrics['active_executions'])
            self.queue_size_gauge.set(app_metrics['queue_size'])
            
            # Store metrics in database
            with SessionLocal() as db:
                metrics = SystemMetrics(
                    cpu_usage_percent=cpu_percent,
                    memory_usage_percent=memory.percent,
                    disk_usage_percent=disk.percent,
                    gpu_usage_percent=gpu_util.get('utilization') if gpu_util else None,
                    gpu_memory_usage_percent=gpu_memory_percent if gpu_info else None,
                    gpu_temperature=gpu_util.get('temperature') if gpu_util else None,
                    active_executions=app_metrics['active_executions'],
                    queue_size=app_metrics['queue_size'],
                    total_executions=app_metrics['total_executions'],
                    failed_executions=app_metrics['failed_executions'],
                    average_execution_time=app_metrics['average_execution_time'],
                    loaded_models_count=app_metrics['loaded_models_count'],
                    model_memory_usage_mb=app_metrics['model_memory_usage_mb']
                )
                
                db.add(metrics)
                db.commit()
            
            # Store in Redis for fast access
            metrics_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'gpu_usage_percent': gpu_util.get('utilization') if gpu_util else None,
                'gpu_memory_usage_percent': gpu_memory_percent if gpu_info else None,
                'active_executions': app_metrics['active_executions'],
                'queue_size': app_metrics['queue_size']
            }
            
            self.redis_client.setex(
                'system_metrics:latest',
                300,  # 5 minutes TTL
                json.dumps(metrics_data)
            )
            
        except Exception as e:
            logger.error("Error collecting metrics", error=str(e))
    
    async def _get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics."""
        try:
            with SessionLocal() as db:
                # Active executions
                active_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.status.in_([WorkflowStatus.PENDING, WorkflowStatus.RUNNING])
                ).count()
                
                # Queue size (pending executions)
                queue_size = db.query(WorkflowExecution).filter(
                    WorkflowExecution.status == WorkflowStatus.PENDING
                ).count()
                
                # Total executions
                total_executions = db.query(WorkflowExecution).count()
                
                # Failed executions
                failed_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.status == WorkflowStatus.FAILED
                ).count()
                
                # Average execution time (last 24 hours)
                yesterday = datetime.utcnow() - timedelta(days=1)
                avg_time_result = db.query(
                    func.avg(
                        func.extract('epoch', WorkflowExecution.completed_at) -
                        func.extract('epoch', WorkflowExecution.started_at)
                    )
                ).filter(
                    WorkflowExecution.status == WorkflowStatus.COMPLETED,
                    WorkflowExecution.completed_at >= yesterday
                ).scalar()
                
                average_execution_time = float(avg_time_result) if avg_time_result else 0.0
            
            # Get model metrics
            from src.services.model import model_service
            try:
                loaded_models_count = len(model_service._loaded_models)
                model_memory_usage_mb = sum(model_service._model_memory.values())
            except:
                loaded_models_count = 0
                model_memory_usage_mb = 0.0
            
            return {
                'active_executions': active_executions,
                'queue_size': queue_size,
                'total_executions': total_executions,
                'failed_executions': failed_executions,
                'average_execution_time': average_execution_time,
                'loaded_models_count': loaded_models_count,
                'model_memory_usage_mb': model_memory_usage_mb
            }
            
        except Exception as e:
            logger.error("Error getting application metrics", error=str(e))
            return {
                'active_executions': 0,
                'queue_size': 0,
                'total_executions': 0,
                'failed_executions': 0,
                'average_execution_time': 0.0,
                'loaded_models_count': 0,
                'model_memory_usage_mb': 0.0
            }
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # Try to get from Redis first (fast)
            cached_metrics = self.redis_client.get('system_metrics:latest')
            if cached_metrics:
                return json.loads(cached_metrics.decode())
            
            # Fall back to live collection
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            gpu_info = get_gpu_memory_info()
            gpu_util = get_gpu_utilization()
            
            app_metrics = await self._get_application_metrics()
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'gpu_usage_percent': gpu_util.get('utilization') if gpu_util else None,
                'gpu_memory_usage_percent': (gpu_info['used_mb'] / gpu_info['total_mb']) * 100 if gpu_info else None,
                'active_executions': app_metrics['active_executions'],
                'queue_size': app_metrics['queue_size'],
                'total_executions': app_metrics['total_executions'],
                'average_execution_time_seconds': app_metrics['average_execution_time']
            }
            
        except Exception as e:
            logger.error("Error getting system metrics", error=str(e))
            return {}
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                'platform': psutil.WINDOWS if psutil.WINDOWS else ('MACOS' if psutil.MACOS else 'LINUX'),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
                'python_version': psutil.version_info,
                'uptime_seconds': time.time() - self.start_time
            }
        except Exception as e:
            logger.error("Error getting system info", error=str(e))
            return {}
    
    async def get_health_metrics(self) -> Dict[str, Any]:
        """Get health-related metrics."""
        try:
            with SessionLocal() as db:
                # Recent errors
                recent_errors = db.query(ExecutionLog).filter(
                    ExecutionLog.level == 'ERROR',
                    ExecutionLog.timestamp >= datetime.utcnow() - timedelta(hours=1)
                ).count()
                
                # Recent failures
                recent_failures = db.query(WorkflowExecution).filter(
                    WorkflowExecution.status == WorkflowStatus.FAILED,
                    WorkflowExecution.created_at >= datetime.utcnow() - timedelta(hours=1)
                ).count()
            
            return {
                'recent_errors_1h': recent_errors,
                'recent_failures_1h': recent_failures,
                'uptime_seconds': time.time() - self.start_time,
                'is_healthy': recent_errors < 10 and recent_failures < 5  # Simple health check
            }
            
        except Exception as e:
            logger.error("Error getting health metrics", error=str(e))
            return {'is_healthy': False, 'error': str(e)}
    
    async def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
        user_id: Optional[int] = None
    ):
        """Record API request metrics."""
        try:
            # Update Prometheus metrics
            self.api_requests_counter.labels(
                method=method,
                endpoint=self._normalize_endpoint(path),
                status=str(status_code)
            ).inc()
            
            self.api_request_duration_histogram.labels(
                method=method,
                endpoint=self._normalize_endpoint(path)
            ).observe(duration_seconds)
            
            # Store detailed request log if needed
            if settings.DEBUG:
                request_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'method': method,
                    'path': path,
                    'status_code': status_code,
                    'duration_seconds': duration_seconds,
                    'user_id': user_id
                }
                
                # Store in Redis with TTL
                self.redis_client.lpush('request_logs', str(request_data))
                self.redis_client.ltrim('request_logs', 0, 1000)  # Keep only last 1000 requests
                
        except Exception as e:
            logger.error("Error recording request metrics", error=str(e))
    
    async def record_workflow_execution(
        self,
        execution_id: str,
        status: WorkflowStatus,
        duration_seconds: Optional[float] = None
    ):
        """Record workflow execution metrics."""
        try:
            # Update Prometheus metrics
            self.execution_counter.labels(status=status.value).inc()
            
            if duration_seconds:
                self.execution_duration_histogram.observe(duration_seconds)
            
        except Exception as e:
            logger.error("Error recording workflow metrics", error=str(e))
    
    async def get_execution_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get execution metrics for a time period."""
        try:
            with SessionLocal() as db:
                # Total executions in period
                total_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.created_at >= start_time,
                    WorkflowExecution.created_at <= end_time
                ).count()
                
                # Completed executions
                completed_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.created_at >= start_time,
                    WorkflowExecution.created_at <= end_time,
                    WorkflowExecution.status == WorkflowStatus.COMPLETED
                ).count()
                
                # Failed executions
                failed_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.created_at >= start_time,
                    WorkflowExecution.created_at <= end_time,
                    WorkflowExecution.status == WorkflowStatus.FAILED
                ).count()
                
                # Average duration
                avg_duration = db.query(
                    func.avg(
                        func.extract('epoch', WorkflowExecution.completed_at) -
                        func.extract('epoch', WorkflowExecution.started_at)
                    )
                ).filter(
                    WorkflowExecution.created_at >= start_time,
                    WorkflowExecution.created_at <= end_time,
                    WorkflowExecution.status == WorkflowStatus.COMPLETED
                ).scalar()
                
                # Executions per minute
                period_minutes = (end_time - start_time).total_seconds() / 60
                executions_per_minute = total_executions / period_minutes if period_minutes > 0 else 0
                
                return {
                    'total_executions': total_executions,
                    'completed_executions': completed_executions,
                    'failed_executions': failed_executions,
                    'average_duration_seconds': float(avg_duration) if avg_duration else 0.0,
                    'executions_per_minute': executions_per_minute,
                    'success_rate': completed_executions / total_executions if total_executions > 0 else 0.0,
                    'failure_rate': failed_executions / total_executions if total_executions > 0 else 0.0,
                    'queue_wait_time_seconds': 0.0  # Placeholder - would need more complex calculation
                }
                
        except Exception as e:
            logger.error("Error getting execution metrics", error=str(e))
            return {}
    
    async def get_user_usage_stats(self, user_id: int, days: int) -> Dict[str, Any]:
        """Get usage statistics for a user."""
        try:
            start_time = datetime.utcnow() - timedelta(days=days)
            
            with SessionLocal() as db:
                # User's executions
                user_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.user_id == user_id,
                    WorkflowExecution.created_at >= start_time
                ).all()
                
                total_executions = len(user_executions)
                completed = sum(1 for e in user_executions if e.status == WorkflowStatus.COMPLETED)
                failed = sum(1 for e in user_executions if e.status == WorkflowStatus.FAILED)
                
                # Calculate total execution time
                total_time = 0
                for execution in user_executions:
                    if execution.started_at and execution.completed_at:
                        total_time += (execution.completed_at - execution.started_at).total_seconds()
                
                return {
                    'total_executions': total_executions,
                    'completed_executions': completed,
                    'failed_executions': failed,
                    'total_execution_time_seconds': total_time,
                    'average_execution_time_seconds': total_time / completed if completed > 0 else 0,
                    'success_rate': completed / total_executions if total_executions > 0 else 0
                }
                
        except Exception as e:
            logger.error("Error getting user usage stats", user_id=user_id, error=str(e))
            return {}
    
    async def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error("Error exporting Prometheus metrics", error=str(e))
            return ""
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics."""
        # Replace IDs with placeholders to avoid high cardinality
        import re
        
        # Replace UUIDs
        path = re.sub(r'/[0-9a-f-]{36}', '/{id}', path)
        
        # Replace other numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path
    
    async def get_performance_metrics(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics."""
        # Placeholder implementation
        return {'status': 'not_implemented'}
    
    async def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts."""
        # Placeholder implementation
        return []
    
    async def get_capacity_info(self) -> Dict[str, Any]:
        """Get capacity information."""
        # Placeholder implementation
        return {'status': 'not_implemented'}
    
    async def get_performance_trends(
        self,
        metric: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get performance trends."""
        # Placeholder implementation
        return {'status': 'not_implemented'}
    
    async def reset_metrics(self):
        """Reset metrics counters."""
        # Placeholder implementation
        pass


# Global monitoring service instance
monitoring_service = MonitoringService()
