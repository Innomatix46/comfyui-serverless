# Advanced Features Documentation

This document covers advanced features of the ComfyUI Serverless API including rate limiting, webhook implementation, monitoring, and enterprise-grade functionality.

## Table of Contents

- [Rate Limiting](#rate-limiting)
- [Webhook Implementation](#webhook-implementation)
- [API Versioning](#api-versioning)
- [Monitoring and Observability](#monitoring-and-observability)
- [Caching Strategies](#caching-strategies)
- [Security Features](#security-features)
- [Performance Optimization](#performance-optimization)
- [Enterprise Features](#enterprise-features)

## Rate Limiting

The API implements sophisticated rate limiting to ensure fair usage and prevent abuse while maintaining high performance for legitimate users.

### Rate Limiting Strategy

The system uses a multi-tier rate limiting approach:

1. **Global Rate Limits**: Applied to all users
2. **User-Specific Limits**: Based on subscription tier
3. **Endpoint-Specific Limits**: Different limits for different operations
4. **Burst Allowance**: Short-term burst capacity for bursty workloads

### Rate Limit Configuration

```python
# Rate limit configuration
RATE_LIMITS = {
    # Workflow execution endpoints
    "/workflows/execute": {
        "requests_per_minute": 10,
        "requests_per_hour": 100,
        "burst_capacity": 5,
        "priority_multiplier": {
            "free": 1.0,
            "pro": 2.0,
            "enterprise": 5.0
        }
    },
    
    # Model management endpoints
    "/models/": {
        "requests_per_minute": 30,
        "requests_per_hour": 300,
        "burst_capacity": 10
    },
    
    # File operations
    "/files/upload": {
        "requests_per_minute": 20,
        "requests_per_hour": 200,
        "burst_capacity": 5,
        "size_limit_mb": 100
    },
    
    # Default for other endpoints
    "default": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "burst_capacity": 20
    }
}
```

### Implementation Details

```python
# services/rate_limiting.py
import redis
import time
import json
from typing import Dict, Optional, Tuple
from enum import Enum
import hashlib

class RateLimitResult(Enum):
    ALLOWED = "allowed"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"

class RateLimitService:
    """Advanced rate limiting service with Redis backend."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_window = 60  # 1 minute
        
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        tier: str = "free"
    ) -> Tuple[RateLimitResult, Dict[str, int]]:
        """Check rate limit with sliding window and burst detection."""
        
        # Get rate limit configuration for endpoint
        config = self.get_rate_limit_config(endpoint, tier)
        
        # Check multiple time windows
        windows = [
            ("minute", 60, config["requests_per_minute"]),
            ("hour", 3600, config["requests_per_hour"]),
        ]
        
        current_time = int(time.time())
        results = {}
        
        for window_name, window_size, limit in windows:
            key = f"rate_limit:{user_id}:{endpoint}:{window_name}"
            
            # Use sliding window counter
            allowed, remaining, reset_time = await self.sliding_window_check(
                key, window_size, limit, current_time
            )
            
            results[window_name] = {
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time,
                "allowed": allowed
            }
            
            if not allowed:
                return RateLimitResult.RATE_LIMITED, results
        
        # Check burst capacity
        burst_allowed = await self.check_burst_limit(
            user_id, endpoint, config["burst_capacity"]
        )
        
        if not burst_allowed:
            return RateLimitResult.RATE_LIMITED, results
        
        # Record the request
        await self.record_request(user_id, endpoint, current_time)
        
        return RateLimitResult.ALLOWED, results
    
    async def sliding_window_check(
        self,
        key: str,
        window_size: int,
        limit: int,
        current_time: int
    ) -> Tuple[bool, int, int]:
        """Sliding window rate limit check."""
        
        # Remove expired entries
        cutoff = current_time - window_size
        
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, cutoff)
        pipe.zcard(key)
        pipe.expire(key, window_size)
        
        results = await pipe.execute()
        current_count = results[1]
        
        # Check if under limit
        if current_count < limit:
            # Add current request
            pipe = self.redis.pipeline()
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, window_size)
            await pipe.execute()
            
            remaining = limit - current_count - 1
            reset_time = current_time + window_size
            return True, remaining, reset_time
        
        remaining = 0
        reset_time = current_time + window_size
        return False, remaining, reset_time
    
    async def check_burst_limit(
        self,
        user_id: str,
        endpoint: str,
        burst_capacity: int
    ) -> bool:
        """Check burst capacity using token bucket algorithm."""
        bucket_key = f"burst:{user_id}:{endpoint}"
        
        # Get current bucket state
        bucket_data = await self.redis.get(bucket_key)
        
        if bucket_data:
            bucket = json.loads(bucket_data)
            tokens = bucket["tokens"]
            last_refill = bucket["last_refill"]
        else:
            tokens = burst_capacity
            last_refill = time.time()
        
        # Refill tokens based on time passed
        now = time.time()
        time_passed = now - last_refill
        
        # Refill at 1 token per second
        tokens_to_add = int(time_passed)
        tokens = min(burst_capacity, tokens + tokens_to_add)
        
        if tokens > 0:
            # Consume a token
            tokens -= 1
            
            # Update bucket
            bucket_data = json.dumps({
                "tokens": tokens,
                "last_refill": now
            })
            await self.redis.setex(bucket_key, 300, bucket_data)  # 5 minute TTL
            
            return True
        
        return False
    
    async def record_request(self, user_id: str, endpoint: str, timestamp: int):
        """Record request for analytics and monitoring."""
        analytics_key = f"analytics:requests:{user_id}"
        
        pipe = self.redis.pipeline()
        pipe.hincrby(analytics_key, f"{endpoint}:{timestamp//3600*3600}", 1)  # Hourly buckets
        pipe.expire(analytics_key, 86400 * 7)  # Keep for 7 days
        await pipe.execute()
    
    def get_rate_limit_config(self, endpoint: str, tier: str) -> Dict[str, int]:
        """Get rate limit configuration for endpoint and tier."""
        base_config = RATE_LIMITS.get(endpoint, RATE_LIMITS["default"]).copy()
        
        # Apply tier multiplier if available
        if "priority_multiplier" in base_config and tier in base_config["priority_multiplier"]:
            multiplier = base_config["priority_multiplier"][tier]
            base_config["requests_per_minute"] = int(base_config["requests_per_minute"] * multiplier)
            base_config["requests_per_hour"] = int(base_config["requests_per_hour"] * multiplier)
        
        return base_config

# FastAPI dependency for rate limiting
from fastapi import HTTPException, Request, Depends
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

async def rate_limit_dependency(
    request: Request,
    current_user = Depends(get_current_user),
    rate_limit_service: RateLimitService = Depends()
):
    """FastAPI dependency for rate limiting."""
    
    endpoint = request.url.path
    user_tier = current_user.subscription_tier if current_user else "free"
    
    result, details = await rate_limit_service.check_rate_limit(
        user_id=str(current_user.id if current_user else request.client.host),
        endpoint=endpoint,
        tier=user_tier
    )
    
    if result != RateLimitResult.ALLOWED:
        # Add rate limit headers
        headers = {}
        if "minute" in details:
            minute_details = details["minute"]
            headers.update({
                "X-RateLimit-Limit": str(minute_details["limit"]),
                "X-RateLimit-Remaining": str(minute_details["remaining"]),
                "X-RateLimit-Reset": str(minute_details["reset"])
            })
        
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please slow down and try again later.",
                "retry_after": details.get("minute", {}).get("reset", 60)
            },
            headers=headers
        )
    
    # Add success headers
    if "minute" in details:
        minute_details = details["minute"]
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(minute_details["limit"]),
            "X-RateLimit-Remaining": str(minute_details["remaining"]),
            "X-RateLimit-Reset": str(minute_details["reset"])
        }
```

### Rate Limiting Headers

All API responses include rate limiting headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
X-RateLimit-Retry-After: 3600
```

## Webhook Implementation

Webhooks provide real-time notifications about workflow status changes, enabling seamless integration with external systems.

### Webhook Events

| Event | Description | Trigger |
|-------|-------------|---------|
| `workflow.queued` | Workflow added to queue | After successful submission |
| `workflow.started` | Workflow execution began | When worker picks up task |
| `workflow.progress` | Execution progress update | During processing (optional) |
| `workflow.completed` | Workflow completed successfully | After successful completion |
| `workflow.failed` | Workflow execution failed | On execution error |
| `workflow.cancelled` | Workflow was cancelled | When manually cancelled |

### Webhook Service Implementation

```python
# services/webhook.py
import aiohttp
import asyncio
import hashlib
import hmac
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class WebhookEvent:
    """Webhook event data structure."""
    event_type: str
    execution_id: str
    user_id: int
    timestamp: datetime
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type,
            "execution_id": self.execution_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }

class WebhookService:
    """Service for managing webhook deliveries."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.max_retries = 5
        self.retry_delays = [1, 5, 15, 60, 300]  # Exponential backoff
        self.timeout = 30  # Request timeout in seconds
        
    async def send_webhook(
        self,
        webhook_url: str,
        event: WebhookEvent,
        secret: Optional[str] = None
    ) -> bool:
        """Send webhook with retry logic."""
        
        webhook_id = self.generate_webhook_id(event)
        
        # Check if already sent successfully
        if await self.is_webhook_delivered(webhook_id):
            return True
        
        payload = event.to_dict()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-Serverless-Webhook/1.0",
            "X-Webhook-ID": webhook_id,
            "X-Webhook-Timestamp": str(int(time.time())),
        }
        
        # Add signature if secret provided
        if secret:
            signature = self.generate_signature(payload, secret)
            headers["X-Webhook-Signature"] = signature
        
        # Attempt delivery with retries
        for attempt in range(self.max_retries):
            try:
                success = await self.attempt_delivery(
                    webhook_url, payload, headers, attempt
                )
                
                if success:
                    await self.mark_webhook_delivered(webhook_id)
                    logger.info(
                        "Webhook delivered successfully",
                        webhook_id=webhook_id,
                        url=webhook_url,
                        event_type=event.event_type,
                        attempt=attempt + 1
                    )
                    return True
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(
                    "Webhook delivery attempt failed",
                    webhook_id=webhook_id,
                    url=webhook_url,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    await asyncio.sleep(delay)
        
        # Mark as failed after all retries
        await self.mark_webhook_failed(webhook_id, event)
        logger.error(
            "Webhook delivery failed after all retries",
            webhook_id=webhook_id,
            url=webhook_url,
            event_type=event.event_type
        )
        return False
    
    async def attempt_delivery(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        attempt: int
    ) -> bool:
        """Attempt single webhook delivery."""
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json=payload,
                headers=headers
            ) as response:
                
                # Log response for debugging
                response_text = await response.text()
                
                logger.info(
                    "Webhook delivery attempt",
                    url=url,
                    status_code=response.status,
                    attempt=attempt + 1,
                    response_size=len(response_text)
                )
                
                # Consider 2xx status codes as success
                return 200 <= response.status < 300
    
    def generate_webhook_id(self, event: WebhookEvent) -> str:
        """Generate unique webhook ID."""
        data = f"{event.execution_id}:{event.event_type}:{event.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate HMAC signature for webhook verification."""
        payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    async def is_webhook_delivered(self, webhook_id: str) -> bool:
        """Check if webhook was already delivered."""
        key = f"webhook:delivered:{webhook_id}"
        return await self.redis.exists(key) > 0
    
    async def mark_webhook_delivered(self, webhook_id: str):
        """Mark webhook as successfully delivered."""
        key = f"webhook:delivered:{webhook_id}"
        await self.redis.setex(key, 86400 * 7, "1")  # Keep for 7 days
    
    async def mark_webhook_failed(self, webhook_id: str, event: WebhookEvent):
        """Mark webhook as failed and store for retry."""
        failed_key = f"webhook:failed:{webhook_id}"
        failed_data = {
            "webhook_id": webhook_id,
            "event": event.to_dict(),
            "failed_at": datetime.utcnow().isoformat(),
            "retry_count": self.max_retries
        }
        
        await self.redis.setex(
            failed_key,
            86400 * 30,  # Keep failed webhooks for 30 days
            json.dumps(failed_data)
        )
    
    async def send_workflow_event(
        self,
        execution_id: str,
        event_type: str,
        webhook_url: str,
        user_id: int,
        data: Dict[str, Any],
        secret: Optional[str] = None
    ) -> bool:
        """Send workflow-related webhook event."""
        
        event = WebhookEvent(
            event_type=event_type,
            execution_id=execution_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            data=data
        )
        
        return await self.send_webhook(webhook_url, event, secret)
    
    async def get_webhook_logs(
        self,
        execution_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get webhook delivery logs for execution."""
        # This would query webhook delivery logs from storage
        # Implementation depends on your logging/storage system
        pass
    
    async def retry_failed_webhooks(self, max_age_hours: int = 24):
        """Retry failed webhooks within specified age."""
        pattern = "webhook:failed:*"
        failed_keys = await self.redis.keys(pattern)
        
        retry_count = 0
        
        for key in failed_keys:
            try:
                data = await self.redis.get(key)
                if not data:
                    continue
                
                webhook_data = json.loads(data)
                failed_at = datetime.fromisoformat(webhook_data["failed_at"])
                
                # Check if within retry window
                if datetime.utcnow() - failed_at > timedelta(hours=max_age_hours):
                    continue
                
                # Reconstruct event and retry
                event_data = webhook_data["event"]
                event = WebhookEvent(
                    event_type=event_data["event"],
                    execution_id=event_data["execution_id"],
                    user_id=event_data["user_id"],
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    data=event_data["data"]
                )
                
                # Get webhook URL from execution record
                # This would query the database for webhook_url
                webhook_url = await self.get_webhook_url_for_execution(
                    event.execution_id
                )
                
                if webhook_url:
                    success = await self.send_webhook(webhook_url, event)
                    if success:
                        await self.redis.delete(key)
                        retry_count += 1
                
            except Exception as e:
                logger.error(
                    "Failed to retry webhook",
                    key=key,
                    error=str(e)
                )
        
        return retry_count

# Celery task for async webhook delivery
@celery_app.task(bind=True, max_retries=3)
def send_webhook_task(
    self,
    webhook_url: str,
    event_data: Dict[str, Any],
    secret: Optional[str] = None
):
    """Async task for webhook delivery."""
    
    webhook_service = WebhookService(redis.from_url(os.getenv("REDIS_URL")))
    
    event = WebhookEvent(
        event_type=event_data["event"],
        execution_id=event_data["execution_id"],
        user_id=event_data["user_id"],
        timestamp=datetime.fromisoformat(event_data["timestamp"]),
        data=event_data["data"]
    )
    
    try:
        # Run async webhook delivery
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        success = loop.run_until_complete(
            webhook_service.send_webhook(webhook_url, event, secret)
        )
        
        if not success:
            raise Exception("Webhook delivery failed")
        
        return {"status": "delivered", "webhook_id": webhook_service.generate_webhook_id(event)}
        
    except Exception as exc:
        logger.error(
            "Webhook task failed",
            webhook_url=webhook_url,
            event_type=event.event_type,
            error=str(exc),
            retry_count=self.request.retries
        )
        
        if self.request.retries < self.max_retries:
            # Retry with exponential backoff
            countdown = 2 ** self.request.retries * 60  # 1, 2, 4 minutes
            raise self.retry(countdown=countdown, exc=exc)
        
        raise
```

### Webhook Security

#### Signature Verification

All webhooks include an HMAC signature for verification:

```python
def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify webhook signature."""
    if not signature.startswith('sha256='):
        return False
    
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    received_signature = signature[7:]  # Remove 'sha256=' prefix
    
    return hmac.compare_digest(expected_signature, received_signature)

# Usage in webhook endpoint
@app.post("/webhook")
async def handle_webhook(
    request: Request,
    signature: str = Header(None, alias="X-Webhook-Signature")
):
    """Handle incoming webhook."""
    payload = await request.body()
    
    if not verify_webhook_signature(payload, signature, webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Process webhook data
    data = await request.json()
    # ... handle event
```

#### Webhook Best Practices

1. **Idempotency**: Handle duplicate webhook deliveries gracefully
2. **Response Time**: Respond quickly (< 30 seconds) to avoid retries
3. **Status Codes**: Return appropriate HTTP status codes
4. **Logging**: Log all webhook events for debugging
5. **Security**: Always verify signatures in production

### Webhook Event Examples

#### Workflow Started Event

```json
{
  "event": "workflow.started",
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": 12345,
  "timestamp": "2024-01-01T10:30:00Z",
  "data": {
    "workflow": {
      "node_count": 7,
      "estimated_duration": 120
    },
    "priority": "normal",
    "queue_position": 2
  }
}
```

#### Workflow Completed Event

```json
{
  "event": "workflow.completed",
  "execution_id": "550e8400-e29b-41d4-a716-446655440000", 
  "user_id": 12345,
  "timestamp": "2024-01-01T10:32:30Z",
  "data": {
    "duration_seconds": 150,
    "outputs": {
      "images": [
        {
          "filename": "output_001.png",
          "url": "https://storage.example.com/outputs/output_001.png",
          "width": 512,
          "height": 512,
          "size_bytes": 892431
        }
      ]
    },
    "metadata": {
      "model_used": "v1-5-pruned-emaonly.ckpt",
      "sampler": "euler_a",
      "steps": 20
    }
  }
}
```

## API Versioning

The API supports multiple versioning strategies to ensure backward compatibility and smooth migrations.

### Versioning Strategies

1. **URL Path Versioning**: `/v1/workflows/execute`, `/v2/workflows/execute`
2. **Header Versioning**: `Accept: application/vnd.comfyui.v1+json`
3. **Query Parameter**: `?version=1.0`

### Implementation

```python
# api/versioning.py
from fastapi import Request, HTTPException
from typing import Optional
import semver

class APIVersionManager:
    """Manage API versioning and compatibility."""
    
    SUPPORTED_VERSIONS = ["1.0", "1.1", "2.0"]
    DEFAULT_VERSION = "2.0"
    
    def __init__(self):
        self.version_mappings = {
            "1.0": "v1",
            "1.1": "v1",  # Backward compatible
            "2.0": "v2"
        }
    
    def extract_version(self, request: Request) -> str:
        """Extract API version from request."""
        
        # Check URL path first
        if request.url.path.startswith('/v'):
            path_parts = request.url.path.split('/')
            if len(path_parts) > 1 and path_parts[1].startswith('v'):
                version_part = path_parts[1]
                try:
                    version = version_part[1:]  # Remove 'v' prefix
                    if version in self.SUPPORTED_VERSIONS:
                        return version
                except:
                    pass
        
        # Check Accept header
        accept_header = request.headers.get('accept', '')
        if 'vnd.comfyui.' in accept_header:
            try:
                # Extract version from Accept header like: application/vnd.comfyui.v1+json
                version_part = accept_header.split('vnd.comfyui.')[1].split('+')[0]
                version = version_part[1:]  # Remove 'v' prefix
                if version in self.SUPPORTED_VERSIONS:
                    return version
            except:
                pass
        
        # Check query parameter
        version = request.query_params.get('version')
        if version and version in self.SUPPORTED_VERSIONS:
            return version
        
        return self.DEFAULT_VERSION
    
    def get_router_prefix(self, version: str) -> str:
        """Get router prefix for version."""
        return self.version_mappings.get(version, "v2")
    
    def check_deprecation(self, version: str) -> Optional[Dict[str, Any]]:
        """Check if version is deprecated."""
        deprecated_versions = {
            "1.0": {
                "deprecated": True,
                "sunset_date": "2024-12-31",
                "migration_guide": "/docs/v1-to-v2-migration",
                "replacement_version": "2.0"
            }
        }
        
        return deprecated_versions.get(version)
    
    def validate_version(self, version: str):
        """Validate requested API version."""
        if version not in self.SUPPORTED_VERSIONS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unsupported_api_version",
                    "message": f"API version {version} is not supported",
                    "supported_versions": self.SUPPORTED_VERSIONS
                }
            )
        
        # Check for deprecation warnings
        deprecation_info = self.check_deprecation(version)
        if deprecation_info and deprecation_info.get("deprecated"):
            # Add deprecation headers (handled by middleware)
            pass

# Middleware for version handling
class VersioningMiddleware(BaseHTTPMiddleware):
    """Middleware to handle API versioning."""
    
    def __init__(self, app):
        super().__init__(app)
        self.version_manager = APIVersionManager()
    
    async def dispatch(self, request: Request, call_next):
        # Extract and validate version
        version = self.version_manager.extract_version(request)
        self.version_manager.validate_version(version)
        
        # Store version in request state
        request.state.api_version = version
        
        # Process request
        response = await call_next(request)
        
        # Add version headers
        response.headers["X-API-Version"] = version
        
        # Add deprecation warnings if needed
        deprecation_info = self.version_manager.check_deprecation(version)
        if deprecation_info and deprecation_info.get("deprecated"):
            response.headers["X-API-Deprecated"] = "true"
            response.headers["X-API-Sunset"] = deprecation_info["sunset_date"]
            response.headers["X-API-Migration-Guide"] = deprecation_info["migration_guide"]
        
        return response
```

## Monitoring and Observability

### Metrics Collection

```python
# services/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import psutil
import time
from typing import Dict, Any
import asyncio

class MetricsCollector:
    """Collect application and system metrics."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status', 'version'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint', 'version'],
            registry=self.registry
        )
        
        # Workflow metrics
        self.workflow_executions_total = Counter(
            'workflow_executions_total',
            'Total workflow executions',
            ['status', 'priority', 'user_tier'],
            registry=self.registry
        )
        
        self.workflow_duration = Histogram(
            'workflow_duration_seconds',
            'Workflow execution duration',
            ['status', 'node_count_bucket'],
            registry=self.registry,
            buckets=[1, 5, 15, 30, 60, 120, 300, 600, 1800, float('inf')]
        )
        
        self.active_workflows = Gauge(
            'active_workflows',
            'Currently active workflows',
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'workflow_queue_size',
            'Current workflow queue size',
            ['priority'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'system_gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        # Model metrics
        self.model_load_duration = Histogram(
            'model_load_duration_seconds',
            'Model loading duration',
            ['model_name', 'model_type'],
            registry=self.registry
        )
        
        self.model_memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Model memory usage in bytes',
            ['model_name', 'model_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_rate = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        version: str
    ):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=status_code,
            version=version
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint,
            version=version
        ).observe(duration)
    
    def record_workflow_execution(
        self,
        status: str,
        priority: str,
        user_tier: str,
        duration_seconds: float,
        node_count: int
    ):
        """Record workflow execution metrics."""
        self.workflow_executions_total.labels(
            status=status,
            priority=priority,
            user_tier=user_tier
        ).inc()
        
        # Bucket node count for histogram
        node_count_bucket = self.get_node_count_bucket(node_count)
        
        self.workflow_duration.labels(
            status=status,
            node_count_bucket=node_count_bucket
        ).observe(duration_seconds)
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.percent)
        
        # GPU metrics (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                self.gpu_usage.labels(gpu_id=str(i)).set(gpu.load * 100)
        except ImportError:
            pass
    
    def get_node_count_bucket(self, node_count: int) -> str:
        """Convert node count to bucket for metrics."""
        if node_count <= 5:
            return "small"
        elif node_count <= 15:
            return "medium"
        elif node_count <= 30:
            return "large"
        else:
            return "xlarge"
    
    def get_metrics(self) -> str:
        """Get Prometheus formatted metrics."""
        return generate_latest(self.registry)

# Background task for metrics collection
@celery_app.task
def collect_system_metrics():
    """Periodic system metrics collection."""
    metrics_collector = MetricsCollector()
    metrics_collector.update_system_metrics()
    
    # Collect workflow queue metrics
    from core.database import SessionLocal
    from models.database import WorkflowExecution, WorkflowStatus
    
    db = SessionLocal()
    try:
        # Active workflows
        active_count = db.query(WorkflowExecution).filter(
            WorkflowExecution.status.in_([WorkflowStatus.RUNNING, WorkflowStatus.PENDING])
        ).count()
        
        metrics_collector.active_workflows.set(active_count)
        
        # Queue size by priority
        for priority in ["low", "normal", "high"]:
            queue_count = db.query(WorkflowExecution).filter(
                WorkflowExecution.status == WorkflowStatus.PENDING,
                WorkflowExecution.priority == priority
            ).count()
            
            metrics_collector.queue_size.labels(priority=priority).set(queue_count)
    
    finally:
        db.close()
```

### Structured Logging

```python
# utils/logging.py
import structlog
import logging
import sys
from typing import Dict, Any
import json
from datetime import datetime

def configure_logging(log_level: str = "INFO", log_format: str = "json"):
    """Configure structured logging."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_format == "json" else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

class LoggerMixin:
    """Mixin to add structured logging to classes."""
    
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = structlog.get_logger(self.__class__.__name__)
        return self._logger
```

## Caching Strategies

### Multi-Level Caching

```python
# services/caching.py
import redis
import json
import pickle
import hashlib
from typing import Any, Optional, Union, Dict
from functools import wraps
from datetime import timedelta
import asyncio

class CacheManager:
    """Multi-level cache manager."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.local_cache = {}  # In-memory cache
        self.local_cache_max_size = 1000
        
        # Cache TTL defaults
        self.default_ttls = {
            "model_metadata": 3600,      # 1 hour
            "workflow_templates": 1800,   # 30 minutes
            "user_sessions": 900,         # 15 minutes
            "api_responses": 300,         # 5 minutes
            "system_metrics": 60          # 1 minute
        }
    
    async def get(self, key: str, cache_level: str = "both") -> Optional[Any]:
        """Get value from cache with fallback strategy."""
        
        if cache_level in ["local", "both"]:
            # Check local cache first
            if key in self.local_cache:
                return self.local_cache[key]["value"]
        
        if cache_level in ["redis", "both"]:
            # Check Redis cache
            try:
                cached_data = await self.redis.get(f"cache:{key}")
                if cached_data:
                    value = pickle.loads(cached_data)
                    
                    # Populate local cache
                    if cache_level == "both":
                        self.set_local_cache(key, value)
                    
                    return value
            except Exception as e:
                structlog.get_logger().error("Redis cache get failed", key=key, error=str(e))
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_level: str = "both",
        cache_type: str = "default"
    ):
        """Set value in cache."""
        
        if ttl is None:
            ttl = self.default_ttls.get(cache_type, 300)
        
        if cache_level in ["local", "both"]:
            self.set_local_cache(key, value)
        
        if cache_level in ["redis", "both"]:
            try:
                serialized_value = pickle.dumps(value)
                await self.redis.setex(f"cache:{key}", ttl, serialized_value)
            except Exception as e:
                structlog.get_logger().error("Redis cache set failed", key=key, error=str(e))
    
    def set_local_cache(self, key: str, value: Any):
        """Set value in local cache with LRU eviction."""
        if len(self.local_cache) >= self.local_cache_max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
        
        self.local_cache[key] = {
            "value": value,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    async def delete(self, key: str, cache_level: str = "both"):
        """Delete key from cache."""
        if cache_level in ["local", "both"] and key in self.local_cache:
            del self.local_cache[key]
        
        if cache_level in ["redis", "both"]:
            try:
                await self.redis.delete(f"cache:{key}")
            except Exception as e:
                structlog.get_logger().error("Redis cache delete failed", key=key, error=str(e))
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = ":".join(key_parts)
        
        # Hash long keys
        if len(key_string) > 250:
            key_string = hashlib.sha256(key_string.encode()).hexdigest()
        
        return key_string
    
    def cached(
        self,
        ttl: Optional[int] = None,
        cache_type: str = "default",
        cache_level: str = "both",
        key_func: Optional[callable] = None
    ):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{self.cache_key(*args, **kwargs)}"
                
                # Try to get from cache
                cached_result = await self.get(cache_key, cache_level)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                await self.set(cache_key, result, ttl, cache_level, cache_type)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, use sync Redis operations
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{self.cache_key(*args, **kwargs)}"
                
                # Try local cache first
                if cache_level in ["local", "both"] and cache_key in self.local_cache:
                    return self.local_cache[cache_key]["value"]
                
                # Try Redis cache
                if cache_level in ["redis", "both"]:
                    try:
                        cached_data = self.redis.get(f"cache:{cache_key}")
                        if cached_data:
                            value = pickle.loads(cached_data)
                            if cache_level == "both":
                                self.set_local_cache(cache_key, value)
                            return value
                    except:
                        pass
                
                # Execute function and cache
                result = func(*args, **kwargs)
                
                if cache_level in ["local", "both"]:
                    self.set_local_cache(cache_key, result)
                
                if cache_level in ["redis", "both"]:
                    try:
                        cache_ttl = ttl or self.default_ttls.get(cache_type, 300)
                        serialized = pickle.dumps(result)
                        self.redis.setex(f"cache:{cache_key}", cache_ttl, serialized)
                    except:
                        pass
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

# Usage examples
cache_manager = CacheManager(redis.from_url(os.getenv("REDIS_URL")))

# Cache model metadata
@cache_manager.cached(ttl=3600, cache_type="model_metadata")
async def get_model_info(model_name: str, model_type: str):
    """Get model information (cached for 1 hour)."""
    # Expensive operation to get model info
    return {
        "name": model_name,
        "type": model_type,
        "size_mb": 4200,
        "hash": "sha256:abc123..."
    }

# Cache workflow templates
@cache_manager.cached(ttl=1800, cache_type="workflow_templates")
def load_workflow_template(template_name: str):
    """Load workflow template from file (cached for 30 minutes)."""
    with open(f"templates/{template_name}.json") as f:
        return json.load(f)
```

This advanced features documentation covers the sophisticated functionality that makes the ComfyUI Serverless API production-ready and enterprise-grade. Each feature includes detailed implementation examples and best practices for optimal performance and reliability.