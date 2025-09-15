"""FastAPI middleware implementations."""
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import uuid
import structlog
from typing import Optional
import redis
import json
from datetime import datetime, timedelta

from src.config.settings import settings
from src.services.auth import auth_service
from src.services.monitoring import monitoring_service

logger = structlog.get_logger()
security = HTTPBearer(auto_error=False)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with logging."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None
        )
        
        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful response
            logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2)
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                duration_ms=round(duration * 1000, 2)
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "message": "An internal error occurred",
                    "request_id": request_id,
                    "timestamp": time.time()
                },
                headers={"X-Request-ID": request_id}
            )


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware."""
    
    # Routes that don't require authentication
    PUBLIC_ROUTES = {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/auth/login",
        "/auth/register",
        "/health",
        "/health/",
        "/metrics"  # Metrics endpoint for monitoring
    }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication."""
        path = request.url.path
        
        # Skip authentication for public routes
        if path in self.PUBLIC_ROUTES or path.startswith("/health/") or path.startswith("/metrics"):
            return await call_next(request)
        
        # Extract authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "authentication_required",
                    "message": "Authorization header is required"
                }
            )
        
        # Validate token format
        if not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "invalid_token_format",
                    "message": "Authorization header must be in format 'Bearer <token>'"
                }
            )
        
        token = authorization.split(" ")[1]
        
        # Validate token - try API key first, then JWT
        try:
            # Import here to avoid circular imports
            from src.services.auth import verify_api_key
            
            # Try API key authentication first
            user = await verify_api_key(token)
            
            # If API key fails, try JWT token
            if not user:
                user = await auth_service.verify_token(token)
            
            if not user:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "invalid_token",
                        "message": "Invalid or expired token"
                    }
                )
            
            # Add user to request state
            request.state.user = user
            
            return await call_next(request)
            
        except Exception as e:
            logger.error("Authentication error", error=str(e))
            return JSONResponse(
                status_code=401,
                content={
                    "error": "authentication_failed",
                    "message": "Authentication failed"
                }
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis."""
    
    def __init__(self, app):
        super().__init__(app)
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
            # Test connection
            self.redis_client.ping()
        except:
            self.redis_client = None
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client identifier (user ID or IP)
        client_id = self._get_client_id(request)
        
        # Check rate limits
        if self.redis_client and not await self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use user ID if authenticated
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limits."""
        now = datetime.utcnow()
        
        # Check per-minute limit
        minute_key = f"rate_limit:minute:{client_id}:{now.strftime('%Y%m%d%H%M')}"
        minute_count = self.redis_client.incr(minute_key)
        if minute_count == 1:
            self.redis_client.expire(minute_key, 60)
        
        if minute_count > settings.RATE_LIMIT_PER_MINUTE:
            return False
        
        # Check per-hour limit
        hour_key = f"rate_limit:hour:{client_id}:{now.strftime('%Y%m%d%H')}"
        hour_count = self.redis_client.incr(hour_key)
        if hour_count == 1:
            self.redis_client.expire(hour_key, 3600)
        
        if hour_count > settings.RATE_LIMIT_PER_HOUR:
            return False
        
        # Check per-day limit
        day_key = f"rate_limit:day:{client_id}:{now.strftime('%Y%m%d')}"
        day_count = self.redis_client.incr(day_key)
        if day_count == 1:
            self.redis_client.expire(day_key, 86400)
        
        if day_count > settings.RATE_LIMIT_PER_DAY:
            return False
        
        return True


class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics collection middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with metrics collection."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        duration = time.time() - start_time
        
        # Record metrics
        await monitoring_service.record_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration,
            user_id=getattr(request.state, 'user', None).id if hasattr(request.state, 'user') and request.state.user else None
        )
        
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with security headers."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
