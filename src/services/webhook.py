"""Webhook service for sending notifications."""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import aiohttp
import structlog
import redis
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.core.database import SessionLocal
from src.models.database import WebhookLog

logger = structlog.get_logger()


class WebhookError(Exception):
    """Custom error for webhook service operations."""
    pass


class WebhookService:
    """Service for sending webhook notifications."""
    
    def __init__(self):
        self.session = None
        self.max_retries = settings.WEBHOOK_RETRY_ATTEMPTS
        self.retry_delay = settings.WEBHOOK_RETRY_DELAY_SECONDS
        self.timeout = settings.WEBHOOK_TIMEOUT_SECONDS
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
        except Exception:
            self.redis_client = None

    async def send_webhook(self, webhook_url: str, payload: Dict[str, Any], timeout_seconds: Optional[int] = None) -> bool:
        """Send a generic webhook with the raw payload. Returns True on <400 status."""
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds or self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(webhook_url, json=payload, headers={'Content-Type': 'application/json'}) as resp:
                    return resp.status < 400
        except Exception as e:
            logger.error("send_webhook error", url=webhook_url, error=str(e))
            return False

    async def send_webhook_with_retry(self, webhook_url: str, payload: Dict[str, Any], max_retries: Optional[int] = None) -> bool:
        """Send a webhook with simple retry/backoff; tracks attempts in Redis if available."""
        attempts = 0
        retries = self.max_retries if max_retries is None else max_retries
        key = None
        if self.redis_client:
            try:
                import hashlib
                key = f"webhook:retries:{hashlib.sha256((webhook_url+json.dumps(payload, sort_keys=True)).encode()).hexdigest()[:16]}"
            except Exception:
                key = None

        while True:
            ok = await self.send_webhook(webhook_url, payload)
            if ok:
                return True
            if attempts >= retries:
                return False
            attempts += 1
            if key and self.redis_client:
                try:
                    self.redis_client.set(key, attempts, ex=3600)
                except Exception:
                    pass
            await asyncio.sleep(self.retry_delay * attempts)
    
    async def send_completion_webhook(
        self,
        webhook_url: str,
        execution_id: str,
        success: bool,
        result_data: Dict[str, Any],
        attempt: int = 1
    ) -> bool:
        """Send workflow completion webhook."""
        try:
            payload = {
                "type": "workflow.completed",
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat(),
                "success": success,
                "data": result_data
            }
            
            return await self._send_webhook(
                webhook_url=webhook_url,
                execution_id=execution_id,
                payload=payload,
                attempt=attempt
            )
            
        except Exception as e:
            logger.error(
                "Failed to send completion webhook",
                execution_id=execution_id,
                webhook_url=webhook_url,
                error=str(e)
            )
            return False
    
    async def send_progress_webhook(
        self,
        webhook_url: str,
        execution_id: str,
        progress_data: Dict[str, Any],
        attempt: int = 1
    ) -> bool:
        """Send workflow progress webhook."""
        try:
            payload = {
                "type": "workflow.progress",
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat(),
                "progress": progress_data
            }
            
            return await self._send_webhook(
                webhook_url=webhook_url,
                execution_id=execution_id,
                payload=payload,
                attempt=attempt
            )
            
        except Exception as e:
            logger.error(
                "Failed to send progress webhook",
                execution_id=execution_id,
                webhook_url=webhook_url,
                error=str(e)
            )
            return False
    
    async def send_error_webhook(
        self,
        webhook_url: str,
        execution_id: str,
        error_message: str,
        attempt: int = 1
    ) -> bool:
        """Send workflow error webhook."""
        try:
            payload = {
                "event": "workflow_error",
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": error_message
            }
            
            return await self._send_webhook(
                webhook_url=webhook_url,
                execution_id=execution_id,
                payload=payload,
                attempt=attempt
            )
            
        except Exception as e:
            logger.error(
                "Failed to send error webhook",
                execution_id=execution_id,
                webhook_url=webhook_url,
                error=str(e)
            )
            return False
    
    async def _send_webhook(
        self,
        webhook_url: str,
        execution_id: str,
        payload: Dict[str, Any],
        attempt: int = 1
    ) -> bool:
        """Send webhook with retry logic."""
        start_time = datetime.utcnow()
        
        try:
            # Create webhook log entry
            log_entry = None
            with SessionLocal() as db:
                log_entry = WebhookLog(
                    execution_id=execution_id,
                    webhook_url=webhook_url,
                    request_payload=payload,
                    attempt_number=attempt
                )
                db.add(log_entry)
                db.commit()
                db.refresh(log_entry)
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': f'{settings.API_TITLE}/{settings.API_VERSION}',
                'X-Webhook-Event': payload.get('type') or payload.get('event', 'unknown'),
                'X-Execution-ID': execution_id
            }
            
            # Send webhook
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers=headers
                ) as response:
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    response_text = await response.text()
                    
                    # Update webhook log
                    with SessionLocal() as db:
                        log_entry = db.query(WebhookLog).filter(
                            WebhookLog.id == log_entry.id
                        ).first()
                        
                        if log_entry:
                            log_entry.response_status = response.status
                            log_entry.response_body = response_text[:1000]  # Limit size
                            log_entry.response_time_ms = response_time
                            log_entry.is_successful = response.status < 400
                            log_entry.delivered_at = datetime.utcnow()
                            
                            if not log_entry.is_successful:
                                log_entry.error_message = f"HTTP {response.status}: {response_text[:200]}"
                                
                                # Schedule retry if not max attempts
                                if attempt < self.max_retries:
                                    log_entry.next_retry_at = (
                                        datetime.utcnow() + 
                                        timedelta(seconds=self.retry_delay * attempt)
                                    )
                            
                            db.commit()
                    
                    # Check if successful
                    if response.status < 400:
                        logger.info(
                            "Webhook sent successfully",
                            execution_id=execution_id,
                            webhook_url=webhook_url,
                            status=response.status,
                            attempt=attempt,
                            response_time_ms=response_time
                        )
                        return True
                    else:
                        logger.warning(
                            "Webhook failed",
                            execution_id=execution_id,
                            webhook_url=webhook_url,
                            status=response.status,
                            attempt=attempt,
                            response_time_ms=response_time
                        )
                        
                        # Retry if not max attempts
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_delay * attempt)
                            return await self._send_webhook(
                                webhook_url=webhook_url,
                                execution_id=execution_id,
                                payload=payload,
                                attempt=attempt + 1
                            )
                        
                        return False
        
        except asyncio.TimeoutError:
            error_msg = f"Webhook timeout after {self.timeout} seconds"
            logger.warning(
                "Webhook timeout",
                execution_id=execution_id,
                webhook_url=webhook_url,
                attempt=attempt
            )
            
            # Update webhook log with timeout error
            with SessionLocal() as db:
                if log_entry:
                    log_entry = db.query(WebhookLog).filter(
                        WebhookLog.id == log_entry.id
                    ).first()
                    
                    if log_entry:
                        log_entry.error_message = error_msg
                        log_entry.is_successful = False
                        
                        if attempt < self.max_retries:
                            log_entry.next_retry_at = (
                                datetime.utcnow() +
                                timedelta(seconds=self.retry_delay * attempt)
                            )
                        
                        db.commit()
            
            # Retry if not max attempts
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * attempt)
                return await self._send_webhook(
                    webhook_url=webhook_url,
                    execution_id=execution_id,
                    payload=payload,
                    attempt=attempt + 1
                )
            
            return False
        
        except Exception as e:
            error_msg = f"Webhook error: {str(e)}"
            logger.error(
                "Webhook error",
                execution_id=execution_id,
                webhook_url=webhook_url,
                attempt=attempt,
                error=str(e)
            )
            
            # Update webhook log with error
            with SessionLocal() as db:
                if log_entry:
                    log_entry = db.query(WebhookLog).filter(
                        WebhookLog.id == log_entry.id
                    ).first()
                    
                    if log_entry:
                        log_entry.error_message = error_msg
                        log_entry.is_successful = False
                        
                        if attempt < self.max_retries:
                            log_entry.next_retry_at = (
                                datetime.utcnow() + 
                                timedelta(seconds=self.retry_delay * attempt)
                            )
                        
                        db.commit()
            
            # Retry if not max attempts
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * attempt)
                return await self._send_webhook(
                    webhook_url=webhook_url,
                    execution_id=execution_id,
                    payload=payload,
                    attempt=attempt + 1
                )
            
            return False
    
    async def retry_failed_webhooks(self) -> int:
        """Retry failed webhooks that are scheduled for retry."""
        try:
            retried_count = 0
            
            with SessionLocal() as db:
                # Find webhooks that need retry
                failed_webhooks = db.query(WebhookLog).filter(
                    WebhookLog.is_successful == False,
                    WebhookLog.next_retry_at <= datetime.utcnow(),
                    WebhookLog.attempt_number < self.max_retries
                ).all()
                
                for webhook_log in failed_webhooks:
                    try:
                        # Recreate the payload
                        payload = webhook_log.request_payload or {}
                        
                        # Retry the webhook
                        success = await self._send_webhook(
                            webhook_url=webhook_log.webhook_url,
                            execution_id=webhook_log.execution_id,
                            payload=payload,
                            attempt=webhook_log.attempt_number + 1
                        )
                        
                        if success:
                            retried_count += 1
                    
                    except Exception as e:
                        logger.error(
                            "Error retrying webhook",
                            webhook_id=webhook_log.id,
                            error=str(e)
                        )
            
            if retried_count > 0:
                logger.info("Retried failed webhooks", count=retried_count)
            
            return retried_count
        
        except Exception as e:
            logger.error("Error in retry failed webhooks", error=str(e))
            return 0
    
    async def get_webhook_stats(self, execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Get webhook delivery statistics."""
        try:
            with SessionLocal() as db:
                query = db.query(WebhookLog)
                
                if execution_id:
                    query = query.filter(WebhookLog.execution_id == execution_id)
                
                # Get stats for last 24 hours
                yesterday = datetime.utcnow() - timedelta(days=1)
                query = query.filter(WebhookLog.created_at >= yesterday)
                
                webhooks = query.all()
                
                total_webhooks = len(webhooks)
                successful_webhooks = sum(1 for w in webhooks if w.is_successful)
                failed_webhooks = total_webhooks - successful_webhooks
                
                # Average response time
                response_times = [w.response_time_ms for w in webhooks if w.response_time_ms]
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                
                return {
                    'total_webhooks': total_webhooks,
                    'successful_webhooks': successful_webhooks,
                    'failed_webhooks': failed_webhooks,
                    'success_rate': successful_webhooks / total_webhooks if total_webhooks > 0 else 0,
                    'average_response_time_ms': avg_response_time,
                    'period_hours': 24
                }
        
        except Exception as e:
            logger.error("Error getting webhook stats", error=str(e))
            return {}
    
    async def cleanup_old_webhook_logs(self, days_to_keep: int = 30) -> int:
        """Clean up old webhook logs."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            with SessionLocal() as db:
                deleted_count = db.query(WebhookLog).filter(
                    WebhookLog.created_at < cutoff_date
                ).delete()
                
                db.commit()
                
                if deleted_count > 0:
                    logger.info("Cleaned up old webhook logs", count=deleted_count)
                
                return deleted_count
        
        except Exception as e:
            logger.error("Error cleaning up webhook logs", error=str(e))
            return 0


# Global webhook service instance
webhook_service = WebhookService()
