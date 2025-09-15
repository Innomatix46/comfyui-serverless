"""Cleanup service for maintaining system health."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import structlog
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.core.database import SessionLocal
from src.models.database import WorkflowExecution, FileUpload, SystemMetrics, ExecutionLog, WebhookLog
from src.services.storage import storage_service
from src.services.model import model_service
from src.services.webhook import webhook_service

logger = structlog.get_logger()


class CleanupService:
    """Service for cleaning up temporary files, logs, and old data."""
    
    def __init__(self):
        self.is_running = False
        self.cleanup_task = None
        self.cleanup_interval = settings.CLEANUP_INTERVAL_HOURS * 3600  # Convert to seconds
    
    def start(self):
        """Start cleanup service."""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Cleanup service started")
    
    def stop(self):
        """Stop cleanup service."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
        logger.info("Cleanup service stopped")
    
    async def _cleanup_loop(self):
        """Main cleanup loop."""
        while self.is_running:
            try:
                await self._run_cleanup_tasks()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _run_cleanup_tasks(self):
        """Run all cleanup tasks."""
        logger.info("Starting cleanup tasks")
        
        start_time = datetime.utcnow()
        results = {}
        
        try:
            # Cleanup expired files
            results['expired_files'] = await storage_service.cleanup_expired_files()
            
            # Cleanup temporary files
            results['temp_files'] = await self._cleanup_temp_files()
            
            # Cleanup old execution logs
            results['old_logs'] = await self._cleanup_old_logs()
            
            # Cleanup old metrics
            results['old_metrics'] = await self._cleanup_old_metrics()
            
            # Cleanup old webhook logs
            results['webhook_logs'] = await webhook_service.cleanup_old_webhook_logs()
            
            # Cleanup unused models
            results['unused_models'] = await model_service.cleanup_unused_models(
                max_age_hours=settings.TEMP_FILE_RETENTION_HOURS
            )
            
            # Retry failed webhooks
            results['retried_webhooks'] = await webhook_service.retry_failed_webhooks()
            
            # Cleanup completed workflow executions (keep for result retention period)
            results['old_executions'] = await self._cleanup_old_executions()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                "Cleanup tasks completed",
                duration_seconds=duration,
                results=results
            )
            
        except Exception as e:
            logger.error("Error running cleanup tasks", error=str(e))
    
    async def _cleanup_temp_files(self) -> int:
        """Clean up temporary files older than retention period."""
        try:
            cleaned_count = 0
            cutoff_time = datetime.utcnow() - timedelta(hours=settings.TEMP_FILE_RETENTION_HOURS)
            
            with SessionLocal() as db:
                # Find old temporary file records
                old_temp_files = db.query(FileUpload).filter(
                    FileUpload.created_at < cutoff_time,
                    FileUpload.file_metadata['is_temporary'].astext.cast(bool) == True
                ).all()
                
                for file_record in old_temp_files:
                    try:
                        # Delete from storage
                        if file_record.storage_type == "s3":
                            from src.services.storage import storage_service
                            await storage_service._delete_from_s3(file_record.storage_path)
                        else:
                            await storage_service._delete_from_local(file_record.storage_path)
                        
                        # Delete from database
                        db.delete(file_record)
                        cleaned_count += 1
                        
                    except Exception as e:
                        logger.error(
                            "Failed to delete temp file",
                            file_id=file_record.id,
                            error=str(e)
                        )
                
                db.commit()
            
            if cleaned_count > 0:
                logger.info("Cleaned up temporary files", count=cleaned_count)
            
            return cleaned_count
            
        except Exception as e:
            logger.error("Error cleaning up temp files", error=str(e))
            return 0
    
    async def _cleanup_old_logs(self) -> int:
        """Clean up old execution logs."""
        try:
            # Keep logs for 7 days
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            
            with SessionLocal() as db:
                deleted_count = db.query(ExecutionLog).filter(
                    ExecutionLog.timestamp < cutoff_time
                ).delete()
                
                db.commit()
                
                if deleted_count > 0:
                    logger.info("Cleaned up old execution logs", count=deleted_count)
                
                return deleted_count
                
        except Exception as e:
            logger.error("Error cleaning up old logs", error=str(e))
            return 0
    
    async def _cleanup_old_metrics(self) -> int:
        """Clean up old system metrics."""
        try:
            # Keep detailed metrics for 30 days
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            
            with SessionLocal() as db:
                deleted_count = db.query(SystemMetrics).filter(
                    SystemMetrics.timestamp < cutoff_time
                ).delete()
                
                db.commit()
                
                if deleted_count > 0:
                    logger.info("Cleaned up old system metrics", count=deleted_count)
                
                return deleted_count
                
        except Exception as e:
            logger.error("Error cleaning up old metrics", error=str(e))
            return 0
    
    async def _cleanup_old_executions(self) -> int:
        """Clean up old workflow executions."""
        try:
            # Keep executions based on result retention days
            cutoff_time = datetime.utcnow() - timedelta(days=settings.RESULT_RETENTION_DAYS)
            
            with SessionLocal() as db:
                # Only delete completed or failed executions, keep others for debugging
                deleted_count = db.query(WorkflowExecution).filter(
                    WorkflowExecution.created_at < cutoff_time,
                    WorkflowExecution.status.in_(['completed', 'failed', 'cancelled'])
                ).delete()
                
                db.commit()
                
                if deleted_count > 0:
                    logger.info("Cleaned up old workflow executions", count=deleted_count)
                
                return deleted_count
                
        except Exception as e:
            logger.error("Error cleaning up old executions", error=str(e))
            return 0
    
    async def run_manual_cleanup(self) -> Dict[str, Any]:
        """Run cleanup tasks manually and return results."""
        logger.info("Starting manual cleanup")
        
        start_time = datetime.utcnow()
        results = {}
        
        try:
            # Run all cleanup tasks
            results['expired_files'] = await storage_service.cleanup_expired_files()
            results['temp_files'] = await self._cleanup_temp_files()
            results['old_logs'] = await self._cleanup_old_logs()
            results['old_metrics'] = await self._cleanup_old_metrics()
            results['webhook_logs'] = await webhook_service.cleanup_old_webhook_logs()
            results['unused_models'] = await model_service.cleanup_unused_models(max_age_hours=1)
            results['retried_webhooks'] = await webhook_service.retry_failed_webhooks()
            results['old_executions'] = await self._cleanup_old_executions()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            results['cleanup_duration_seconds'] = duration
            results['cleanup_timestamp'] = start_time.isoformat()
            results['status'] = 'completed'
            
            logger.info("Manual cleanup completed", results=results)
            
            return results
            
        except Exception as e:
            logger.error("Error in manual cleanup", error=str(e))
            
            return {
                'status': 'failed',
                'error': str(e),
                'cleanup_timestamp': start_time.isoformat(),
                'cleanup_duration_seconds': (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup service statistics."""
        try:
            with SessionLocal() as db:
                # Count items that could be cleaned up
                now = datetime.utcnow()
                
                # Expired files
                expired_files = db.query(FileUpload).filter(
                    FileUpload.expires_at < now
                ).count()
                
                # Old logs (> 7 days)
                old_logs = db.query(ExecutionLog).filter(
                    ExecutionLog.timestamp < now - timedelta(days=7)
                ).count()
                
                # Old metrics (> 30 days)
                old_metrics = db.query(SystemMetrics).filter(
                    SystemMetrics.timestamp < now - timedelta(days=30)
                ).count()
                
                # Old executions
                old_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.created_at < now - timedelta(days=settings.RESULT_RETENTION_DAYS),
                    WorkflowExecution.status.in_(['completed', 'failed', 'cancelled'])
                ).count()
                
                # Temp files
                temp_files = db.query(FileUpload).filter(
                    FileUpload.created_at < now - timedelta(hours=settings.TEMP_FILE_RETENTION_HOURS),
                    FileUpload.file_metadata['is_temporary'].astext.cast(bool) == True
                ).count()
                
                # Failed webhooks pending retry
                failed_webhooks = db.query(WebhookLog).filter(
                    WebhookLog.is_successful == False,
                    WebhookLog.next_retry_at <= now,
                    WebhookLog.attempt_number < settings.WEBHOOK_RETRY_ATTEMPTS
                ).count()
                
                return {
                    'service_running': self.is_running,
                    'cleanup_interval_hours': settings.CLEANUP_INTERVAL_HOURS,
                    'items_pending_cleanup': {
                        'expired_files': expired_files,
                        'old_logs': old_logs,
                        'old_metrics': old_metrics,
                        'old_executions': old_executions,
                        'temp_files': temp_files,
                        'failed_webhooks': failed_webhooks
                    },
                    'retention_policies': {
                        'temp_files_hours': settings.TEMP_FILE_RETENTION_HOURS,
                        'result_retention_days': settings.RESULT_RETENTION_DAYS,
                        'log_retention_days': 7,
                        'metrics_retention_days': 30,
                        'webhook_log_retention_days': 30
                    }
                }
                
        except Exception as e:
            logger.error("Error getting cleanup stats", error=str(e))
            return {'error': str(e)}


# Global cleanup service instance
cleanup_service = CleanupService()
