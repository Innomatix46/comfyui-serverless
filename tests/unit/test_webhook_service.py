"""Unit tests for webhook service."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json
import aiohttp
from typing import Dict, Any

from src.services.webhook import WebhookService, WebhookError


@pytest.fixture
def webhook_service():
    """Create webhook service instance."""
    return WebhookService()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = Mock()
    mock.lpush.return_value = 1
    mock.llen.return_value = 0
    mock.rpop.return_value = None
    mock.set.return_value = True
    mock.get.return_value = None
    mock.incr.return_value = 1
    mock.expire.return_value = True
    return mock


@pytest.fixture
def sample_webhook_data():
    """Sample webhook payload."""
    return {
        "execution_id": "test-execution-123",
        "status": "completed",
        "timestamp": datetime.utcnow().isoformat(),
        "result": {
            "outputs": {
                "images": [{"filename": "output.png", "type": "output"}]
            }
        },
        "metadata": {
            "duration_seconds": 45.2,
            "queue_time_seconds": 12.1
        }
    }


class TestWebhookService:
    """Test webhook service functionality."""
    
    @pytest.mark.asyncio
    async def test_send_webhook_success(self, webhook_service, sample_webhook_data):
        """Test successful webhook delivery."""
        webhook_url = "https://example.com/webhook"
        
        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"received": True}
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await webhook_service.send_webhook(webhook_url, sample_webhook_data)
        
        assert result is True
        mock_post.assert_called_once()
        
        # Verify request details
        call_args = mock_post.call_args
        assert call_args[0][0] == webhook_url
        assert "json" in call_args[1]
        assert call_args[1]["json"] == sample_webhook_data
    
    @pytest.mark.asyncio
    async def test_send_webhook_failure(self, webhook_service, sample_webhook_data):
        """Test webhook delivery failure."""
        webhook_url = "https://example.com/webhook"
        
        # Mock failed HTTP response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await webhook_service.send_webhook(webhook_url, sample_webhook_data)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_completion_webhook(self, webhook_service):
        """Test sending workflow completion webhook."""
        webhook_url = "https://example.com/completion"
        execution_id = "test-execution-456"
        success = True
        result_data = {
            "outputs": {"images": [{"filename": "test.png"}]},
            "execution_time": 30.5
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            sent = await webhook_service.send_completion_webhook(
                webhook_url, execution_id, success, result_data
            )
        
        assert sent is True
        
        # Verify payload structure
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        
        assert payload["type"] == "workflow.completed"
        assert payload["execution_id"] == execution_id
        assert payload["success"] is True
        assert payload["data"] == result_data
        assert "timestamp" in payload
    
    @pytest.mark.asyncio
    async def test_send_progress_webhook(self, webhook_service):
        """Test sending progress update webhook."""
        webhook_url = "https://example.com/progress"
        execution_id = "test-execution-789"
        progress_data = {
            "progress_percent": 45.0,
            "current_node": "KSampler",
            "eta_seconds": 120
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            sent = await webhook_service.send_progress_webhook(
                webhook_url, execution_id, progress_data
            )
        
        assert sent is True
        
        # Verify payload structure
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        
        assert payload["type"] == "workflow.progress"
        assert payload["execution_id"] == execution_id
        assert payload["progress"] == progress_data
    
    @pytest.mark.asyncio
    async def test_webhook_retry_mechanism(self, webhook_service, mock_redis, sample_webhook_data):
        """Test webhook retry mechanism."""
        webhook_url = "https://example.com/webhook"
        max_retries = 3
        
        # Mock failed responses for retries
        mock_response = AsyncMock()
        mock_response.status = 503  # Service Unavailable
        mock_response.text.return_value = "Service Unavailable"
        
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch.object(webhook_service, 'redis_client', mock_redis), \
             patch('asyncio.sleep', return_value=None):  # Speed up test
            
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await webhook_service.send_webhook_with_retry(
                webhook_url, sample_webhook_data, max_retries=max_retries
            )
        
        assert result is False
        assert mock_post.call_count == max_retries + 1  # Initial attempt + retries
        
        # Verify retry delays were applied (exponential backoff)
        mock_redis.set.assert_called()  # Retry count tracking
    
    @pytest.mark.asyncio
    async def test_webhook_queue_processing(self, webhook_service, mock_redis):
        """Test webhook queue processing."""
        # Mock queued webhook
        queued_webhook = json.dumps({
            "url": "https://example.com/webhook",
            "payload": {"execution_id": "test", "status": "completed"},
            "created_at": datetime.utcnow().isoformat(),
            "retry_count": 0
        })
        
        mock_redis.rpop.side_effect = [queued_webhook.encode(), None]  # One item, then empty
        
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch.object(webhook_service, 'redis_client', mock_redis):
            
            mock_post.return_value.__aenter__.return_value = mock_response
            
            processed = await webhook_service.process_webhook_queue()
        
        assert processed == 1
        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_webhook_validation(self, webhook_service):
        """Test webhook URL and payload validation."""
        # Test valid webhook URL
        valid_urls = [
            "https://example.com/webhook",
            "http://localhost:8080/callback",
            "https://webhook.site/unique-id"
        ]
        
        for url in valid_urls:
            assert webhook_service.validate_webhook_url(url) is True
        
        # Test invalid webhook URLs
        invalid_urls = [
            "ftp://example.com/webhook",
            "not-a-url",
            "javascript:alert('xss')",
            "",
            None
        ]
        
        for url in invalid_urls:
            assert webhook_service.validate_webhook_url(url) is False
        
        # Test payload validation
        valid_payload = {"execution_id": "test", "status": "completed"}
        assert webhook_service.validate_payload(valid_payload) is True
        
        invalid_payloads = [
            None,
            "",
            [],
            {"circular": None}  # Will be modified to create circular reference
        ]
        
        # Create circular reference
        invalid_payloads[3]["circular"] = invalid_payloads[3]
        
        for payload in invalid_payloads:
            assert webhook_service.validate_payload(payload) is False
    
    @pytest.mark.asyncio
    async def test_webhook_security_headers(self, webhook_service, sample_webhook_data):
        """Test security headers in webhook requests."""
        webhook_url = "https://example.com/webhook"
        secret = "webhook-secret-key"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await webhook_service.send_secure_webhook(
                webhook_url, sample_webhook_data, secret=secret
            )
        
        # Verify security headers
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        
        assert "X-Webhook-Signature" in headers
        assert "X-Webhook-Timestamp" in headers
        assert "User-Agent" in headers
        assert headers["User-Agent"].startswith("ComfyUI-Serverless-Webhook")
    
    @pytest.mark.asyncio
    async def test_webhook_rate_limiting(self, webhook_service, mock_redis):
        """Test webhook rate limiting."""
        webhook_url = "https://example.com/webhook"
        
        # Mock rate limit exceeded
        mock_redis.get.return_value = b"10"  # 10 requests in current window
        
        with patch.object(webhook_service, 'redis_client', mock_redis):
            result = await webhook_service.check_rate_limit(webhook_url, max_requests=5)
        
        assert result is False  # Rate limit exceeded
        
        # Mock within rate limit
        mock_redis.get.return_value = b"3"  # 3 requests in current window
        
        result = await webhook_service.check_rate_limit(webhook_url, max_requests=5)
        assert result is True  # Within rate limit
    
    @pytest.mark.asyncio
    async def test_webhook_timeout_handling(self, webhook_service, sample_webhook_data):
        """Test webhook timeout handling."""
        webhook_url = "https://slow-webhook.example.com/webhook"
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()
            
            result = await webhook_service.send_webhook(
                webhook_url, sample_webhook_data, timeout=5.0
            )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_webhook_filtering(self, webhook_service):
        """Test webhook event filtering."""
        webhook_url = "https://example.com/webhook"
        
        # Configure filters
        filters = {
            "events": ["workflow.completed", "workflow.failed"],
            "statuses": ["completed", "failed"]
        }
        
        # Test matching event
        matching_data = {
            "type": "workflow.completed",
            "status": "completed",
            "execution_id": "test"
        }
        
        should_send = webhook_service.should_send_webhook(matching_data, filters)
        assert should_send is True
        
        # Test non-matching event
        non_matching_data = {
            "type": "workflow.progress",
            "status": "running",
            "execution_id": "test"
        }
        
        should_send = webhook_service.should_send_webhook(non_matching_data, filters)
        assert should_send is False
    
    @pytest.mark.asyncio
    async def test_webhook_batch_delivery(self, webhook_service, mock_redis):
        """Test batch webhook delivery."""
        webhook_urls = [
            "https://webhook1.example.com/callback",
            "https://webhook2.example.com/callback",
            "https://webhook3.example.com/callback"
        ]
        
        payload = {"execution_id": "batch-test", "status": "completed"}
        
        # Mock successful responses
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            results = await webhook_service.send_batch_webhooks(webhook_urls, payload)
        
        assert len(results) == 3
        assert all(result is True for result in results)
        assert mock_post.call_count == 3


class TestWebhookServiceEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_malformed_webhook_response(self, webhook_service, sample_webhook_data):
        """Test handling of malformed webhook responses."""
        webhook_url = "https://example.com/webhook"
        
        # Mock response with invalid JSON
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        mock_response.text.return_value = "Not JSON"
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Should still be considered successful (status 200)
            result = await webhook_service.send_webhook(webhook_url, sample_webhook_data)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_webhook_redirect_handling(self, webhook_service, sample_webhook_data):
        """Test handling of webhook redirects."""
        webhook_url = "https://example.com/webhook"
        
        # Mock redirect response
        mock_response = AsyncMock()
        mock_response.status = 302
        mock_response.headers = {"Location": "https://example.com/new-webhook"}
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await webhook_service.send_webhook(webhook_url, sample_webhook_data)
        
        # 3xx status codes should be considered successful for webhooks
        assert result is True
    
    @pytest.mark.asyncio
    async def test_webhook_payload_size_limit(self, webhook_service):
        """Test webhook payload size limitations."""
        webhook_url = "https://example.com/webhook"
        
        # Create large payload
        large_payload = {
            "execution_id": "large-payload-test",
            "large_data": "x" * (10 * 1024 * 1024)  # 10MB string
        }
        
        result = await webhook_service.send_webhook(webhook_url, large_payload)
        
        # Should reject oversized payloads
        assert result is False
    
    @pytest.mark.asyncio
    async def test_webhook_duplicate_delivery_prevention(self, webhook_service, mock_redis):
        """Test prevention of duplicate webhook deliveries."""
        webhook_url = "https://example.com/webhook"
        execution_id = "duplicate-test"
        payload = {"execution_id": execution_id, "status": "completed"}
        
        # Mock Redis to show webhook already sent
        mock_redis.get.return_value = b"sent"
        
        with patch.object(webhook_service, 'redis_client', mock_redis):
            result = await webhook_service.send_webhook_once(webhook_url, payload)
        
        assert result is False  # Already sent
        
        # Mock Redis to show webhook not sent yet
        mock_redis.get.return_value = None
        
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await webhook_service.send_webhook_once(webhook_url, payload)
        
        assert result is True
        mock_redis.set.assert_called()  # Mark as sent
    
    @pytest.mark.asyncio
    async def test_webhook_circuit_breaker(self, webhook_service, mock_redis):
        """Test webhook circuit breaker functionality."""
        webhook_url = "https://failing-webhook.example.com/callback"
        
        # Mock high failure rate
        mock_redis.get.side_effect = [
            b"10",  # failure_count
            b"5",   # success_count (high failure rate)
            b"open" # circuit_breaker_state
        ]
        
        with patch.object(webhook_service, 'redis_client', mock_redis):
            result = await webhook_service.should_send_webhook_with_circuit_breaker(webhook_url)
        
        assert result is False  # Circuit breaker is open
    
    @pytest.mark.asyncio
    async def test_webhook_custom_headers(self, webhook_service, sample_webhook_data):
        """Test webhook with custom headers."""
        webhook_url = "https://example.com/webhook"
        custom_headers = {
            "Authorization": "Bearer token123",
            "X-Custom-Header": "custom-value",
            "Content-Type": "application/json"
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await webhook_service.send_webhook(
                webhook_url, sample_webhook_data, headers=custom_headers
            )
        
        # Verify custom headers were included
        call_args = mock_post.call_args
        sent_headers = call_args[1]["headers"]
        
        for key, value in custom_headers.items():
            assert sent_headers[key] == value
    
    @pytest.mark.asyncio
    async def test_webhook_ssl_verification(self, webhook_service, sample_webhook_data):
        """Test webhook SSL certificate verification."""
        webhook_url = "https://self-signed.example.com/webhook"
        
        # Mock SSL error
        ssl_error = aiohttp.ClientSSLError(
            connection_key=None,
            os_error=None
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = ssl_error
            
            result = await webhook_service.send_webhook(webhook_url, sample_webhook_data)
        
        assert result is False
        
        # Test with SSL verification disabled
        result = await webhook_service.send_webhook(
            webhook_url, sample_webhook_data, verify_ssl=False
        )
        
        # Should succeed when SSL verification is disabled
        # (mock_post would need to be set up differently for this test)