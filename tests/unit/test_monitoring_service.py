"""Unit tests for monitoring service."""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json

from src.services.monitoring import MonitoringService, MetricsCollector


@pytest.fixture
def monitoring_service():
    """Create monitoring service instance."""
    return MonitoringService()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = Mock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    mock.hset.return_value = 1
    mock.hget.return_value = None
    mock.hincrby.return_value = 1
    mock.incr.return_value = 1
    mock.expire.return_value = True
    mock.lpush.return_value = 1
    mock.lrange.return_value = []
    return mock


@pytest.fixture
def mock_database():
    """Mock database session."""
    mock = Mock()
    mock.query.return_value.filter.return_value.count.return_value = 5
    mock.query.return_value.filter.return_value.all.return_value = []
    return mock


class TestMonitoringService:
    """Test monitoring service functionality."""
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self, monitoring_service):
        """Test system metrics collection."""
        with patch('psutil.cpu_percent', return_value=45.2), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('src.utils.gpu.get_gpu_memory_info') as mock_gpu:
            
            # Mock memory info
            mock_memory.return_value = Mock(
                total=32 * 1024**3,  # 32GB
                available=16 * 1024**3,  # 16GB available
                percent=50.0
            )
            
            # Mock disk info
            mock_disk.return_value = Mock(
                total=1024 * 1024**3,  # 1TB
                used=512 * 1024**3,    # 512GB used
                free=512 * 1024**3     # 512GB free
            )
            
            # Mock GPU info
            mock_gpu.return_value = {
                'total_mb': 24000,
                'used_mb': 8000,
                'free_mb': 16000,
                'utilization_percent': 33.3
            }
            
            metrics = await monitoring_service.collect_system_metrics()
        
        assert metrics["cpu_usage_percent"] == 45.2
        assert metrics["memory_usage_percent"] == 50.0
        assert metrics["disk_usage_percent"] == 50.0
        assert metrics["gpu_memory_usage_percent"] == 33.3
        assert "timestamp" in metrics
    
    @pytest.mark.asyncio
    async def test_collect_application_metrics(self, monitoring_service, mock_database):
        """Test application metrics collection."""
        with patch.object(monitoring_service, '_get_db_session', return_value=mock_database):
            metrics = await monitoring_service.collect_application_metrics()
        
        assert "active_executions" in metrics
        assert "total_executions" in metrics
        assert "queue_length" in metrics
        assert "average_execution_time" in metrics
    
    @pytest.mark.asyncio
    async def test_health_check_all_services(self, monitoring_service, mock_redis):
        """Test comprehensive health check."""
        # Mock ComfyUI health check
        mock_comfyui = AsyncMock()
        mock_comfyui.health_check.return_value = True
        
        # Mock S3 health check
        mock_s3 = Mock()
        mock_s3.head_bucket.return_value = {}
        
        with patch('src.services.comfyui.ComfyUIClient', return_value=mock_comfyui), \
             patch('boto3.client', return_value=mock_s3), \
             patch.object(monitoring_service, 'redis_client', mock_redis), \
             patch('sqlalchemy.create_engine') as mock_engine:
            
            mock_engine.return_value.connect.return_value.__enter__.return_value = Mock()
            
            health_status = await monitoring_service.check_health()
        
        assert health_status["status"] == "healthy"
        assert "services" in health_status
        assert len(health_status["services"]) >= 4  # Redis, DB, ComfyUI, S3
        
        # Check individual service statuses
        service_names = [s["name"] for s in health_status["services"]]
        assert "redis" in service_names
        assert "database" in service_names
        assert "comfyui" in service_names
        assert "storage" in service_names
    
    @pytest.mark.asyncio
    async def test_track_execution_metrics(self, monitoring_service, mock_redis):
        """Test execution metrics tracking."""
        execution_id = "test-execution-123"
        metrics_data = {
            "duration_seconds": 45.2,
            "queue_time_seconds": 12.1,
            "nodes_processed": 8,
            "memory_peak_mb": 1200.0,
            "gpu_utilization_peak": 85.5
        }
        
        with patch.object(monitoring_service, 'redis_client', mock_redis):
            await monitoring_service.track_execution_metrics(execution_id, metrics_data)
        
        # Verify Redis calls for metrics storage
        mock_redis.hset.assert_called()
        mock_redis.lpush.assert_called()
        mock_redis.expire.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_execution_statistics(self, monitoring_service, mock_redis):
        """Test getting execution statistics."""
        # Mock Redis data
        mock_redis.lrange.return_value = [
            json.dumps({"duration_seconds": 45.0, "timestamp": datetime.utcnow().isoformat()}).encode(),
            json.dumps({"duration_seconds": 60.0, "timestamp": datetime.utcnow().isoformat()}).encode(),
            json.dumps({"duration_seconds": 30.0, "timestamp": datetime.utcnow().isoformat()}).encode()
        ]
        
        with patch.object(monitoring_service, 'redis_client', mock_redis):
            stats = await monitoring_service.get_execution_statistics(timeframe_hours=24)
        
        assert stats["total_executions"] == 3
        assert stats["average_duration_seconds"] == 45.0
        assert stats["min_duration_seconds"] == 30.0
        assert stats["max_duration_seconds"] == 60.0
    
    @pytest.mark.asyncio
    async def test_detect_anomalies(self, monitoring_service, mock_redis):
        """Test anomaly detection."""
        # Mock historical metrics showing normal behavior
        normal_metrics = [
            {"cpu_usage_percent": 30.0, "memory_usage_percent": 40.0},
            {"cpu_usage_percent": 35.0, "memory_usage_percent": 45.0},
            {"cpu_usage_percent": 32.0, "memory_usage_percent": 38.0}
        ]
        
        # Current metrics showing anomaly
        current_metrics = {
            "cpu_usage_percent": 95.0,  # Much higher than normal
            "memory_usage_percent": 85.0,  # Much higher than normal
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with patch.object(monitoring_service, '_get_historical_metrics', return_value=normal_metrics):
            anomalies = await monitoring_service.detect_anomalies(current_metrics)
        
        assert len(anomalies) == 2  # CPU and memory anomalies
        assert any("cpu_usage" in anomaly["metric"] for anomaly in anomalies)
        assert any("memory_usage" in anomaly["metric"] for anomaly in anomalies)
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, monitoring_service):
        """Test alert generation and notification."""
        # Mock alert condition
        alert_data = {
            "type": "high_cpu_usage",
            "metric": "cpu_usage_percent",
            "current_value": 95.0,
            "threshold": 80.0,
            "severity": "high",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        mock_webhook = AsyncMock(return_value=True)
        
        with patch.object(monitoring_service, 'send_alert_webhook', mock_webhook):
            result = await monitoring_service.generate_alert(alert_data)
        
        assert result is True
        mock_webhook.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, monitoring_service):
        """Test performance regression detection."""
        # Mock historical performance data
        historical_data = [
            {"average_execution_time": 45.0, "date": "2024-08-29"},
            {"average_execution_time": 47.0, "date": "2024-08-30"},
            {"average_execution_time": 46.0, "date": "2024-08-31"}
        ]
        
        # Current performance data showing regression
        current_data = {"average_execution_time": 85.0}
        
        with patch.object(monitoring_service, '_get_historical_performance', return_value=historical_data):
            regression = await monitoring_service.detect_performance_regression(current_data)
        
        assert regression["detected"] is True
        assert regression["severity"] == "high"
        assert regression["performance_degradation_percent"] > 50
    
    @pytest.mark.asyncio
    async def test_custom_metrics_collection(self, monitoring_service, mock_redis):
        """Test custom metrics collection and storage."""
        custom_metric = {
            "name": "workflow_complexity_score",
            "value": 7.5,
            "tags": {"workflow_type": "image_generation", "user_tier": "premium"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with patch.object(monitoring_service, 'redis_client', mock_redis):
            await monitoring_service.record_custom_metric(custom_metric)
        
        # Verify metric was stored
        mock_redis.hset.assert_called()
        mock_redis.expire.assert_called()
    
    @pytest.mark.asyncio
    async def test_metrics_aggregation(self, monitoring_service, mock_redis):
        """Test metrics aggregation and rollup."""
        # Mock minute-level metrics
        minute_metrics = [
            {"cpu_usage": 30.0, "timestamp": datetime.utcnow() - timedelta(minutes=1)},
            {"cpu_usage": 35.0, "timestamp": datetime.utcnow() - timedelta(minutes=2)},
            {"cpu_usage": 40.0, "timestamp": datetime.utcnow() - timedelta(minutes=3)}
        ]
        
        with patch.object(monitoring_service, '_get_minute_metrics', return_value=minute_metrics):
            hourly_aggregate = await monitoring_service.aggregate_metrics_to_hourly()
        
        assert hourly_aggregate["cpu_usage_avg"] == 35.0
        assert hourly_aggregate["cpu_usage_min"] == 30.0
        assert hourly_aggregate["cpu_usage_max"] == 40.0
    
    @pytest.mark.asyncio
    async def test_resource_usage_forecasting(self, monitoring_service):
        """Test resource usage forecasting."""
        # Mock historical resource usage data
        historical_usage = [
            {"timestamp": datetime.utcnow() - timedelta(hours=i), "cpu_usage": 30 + i * 2}
            for i in range(24)  # 24 hours of data
        ]
        
        with patch.object(monitoring_service, '_get_historical_usage', return_value=historical_usage):
            forecast = await monitoring_service.forecast_resource_usage(hours_ahead=6)
        
        assert "cpu_usage_forecast" in forecast
        assert len(forecast["cpu_usage_forecast"]) == 6
        assert "confidence_interval" in forecast


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        
        assert collector.metrics == {}
        assert collector.start_time is not None
    
    def test_counter_metrics(self):
        """Test counter metrics."""
        collector = MetricsCollector()
        
        # Increment counter
        collector.increment_counter("api_requests")
        collector.increment_counter("api_requests", value=5)
        
        assert collector.get_metric("api_requests") == 6
    
    def test_gauge_metrics(self):
        """Test gauge metrics."""
        collector = MetricsCollector()
        
        # Set gauge value
        collector.set_gauge("queue_length", 10)
        assert collector.get_metric("queue_length") == 10
        
        # Update gauge
        collector.set_gauge("queue_length", 15)
        assert collector.get_metric("queue_length") == 15
    
    def test_histogram_metrics(self):
        """Test histogram metrics."""
        collector = MetricsCollector()
        
        # Record histogram values
        values = [10, 20, 30, 40, 50]
        for value in values:
            collector.record_histogram("execution_time", value)
        
        histogram = collector.get_metric("execution_time")
        assert histogram["count"] == 5
        assert histogram["sum"] == 150
        assert histogram["avg"] == 30
        assert histogram["min"] == 10
        assert histogram["max"] == 50
    
    def test_summary_metrics(self):
        """Test summary metrics."""
        collector = MetricsCollector()
        
        # Record summary values
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for value in values:
            collector.record_summary("response_time", value)
        
        summary = collector.get_metric("response_time")
        assert summary["count"] == 10
        assert summary["sum"] == 55
        assert "percentiles" in summary
        assert summary["percentiles"]["50"] == 5.5  # Median
        assert summary["percentiles"]["95"] == 9.5  # 95th percentile
    
    def test_metrics_with_labels(self):
        """Test metrics with labels."""
        collector = MetricsCollector()
        
        # Record metrics with labels
        collector.increment_counter("api_requests", labels={"endpoint": "/workflows", "method": "POST"})
        collector.increment_counter("api_requests", labels={"endpoint": "/models", "method": "GET"})
        collector.increment_counter("api_requests", labels={"endpoint": "/workflows", "method": "POST"})
        
        workflows_requests = collector.get_metric("api_requests", {"endpoint": "/workflows", "method": "POST"})
        models_requests = collector.get_metric("api_requests", {"endpoint": "/models", "method": "GET"})
        
        assert workflows_requests == 2
        assert models_requests == 1
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        collector = MetricsCollector()
        
        # Add various metrics
        collector.increment_counter("requests_total", value=100)
        collector.set_gauge("active_connections", 25)
        collector.record_histogram("request_duration", 0.5)
        
        # Export metrics
        exported = collector.export_metrics()
        
        assert "requests_total" in exported
        assert "active_connections" in exported
        assert "request_duration" in exported
        assert "timestamp" in exported
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        collector = MetricsCollector()
        
        # Add some metrics
        collector.increment_counter("test_counter", value=10)
        collector.set_gauge("test_gauge", 20)
        
        # Verify metrics exist
        assert collector.get_metric("test_counter") == 10
        assert collector.get_metric("test_gauge") == 20
        
        # Reset metrics
        collector.reset_metrics()
        
        # Verify metrics are cleared
        assert collector.get_metric("test_counter") == 0
        assert collector.get_metric("test_gauge") is None


class TestMonitoringServiceEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_health_check_with_service_failures(self, monitoring_service, mock_redis):
        """Test health check when some services are failing."""
        # Mock failing services
        mock_comfyui = AsyncMock()
        mock_comfyui.health_check.side_effect = Exception("ComfyUI connection failed")
        
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        
        with patch('src.services.comfyui.ComfyUIClient', return_value=mock_comfyui), \
             patch.object(monitoring_service, 'redis_client', mock_redis):
            
            health_status = await monitoring_service.check_health()
        
        assert health_status["status"] == "degraded"
        
        # Check that failed services are marked as unhealthy
        failed_services = [s for s in health_status["services"] if s["status"] == "unhealthy"]
        assert len(failed_services) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_collection_with_missing_dependencies(self, monitoring_service):
        """Test metrics collection when some dependencies are missing."""
        with patch('psutil.cpu_percent', side_effect=ImportError("psutil not available")), \
             patch('src.utils.gpu.get_gpu_memory_info', side_effect=Exception("GPU not available")):
            
            metrics = await monitoring_service.collect_system_metrics()
        
        # Should still collect available metrics
        assert "timestamp" in metrics
        # Missing metrics should have default/null values
        assert metrics.get("cpu_usage_percent") is None
        assert metrics.get("gpu_memory_usage_percent") is None
    
    @pytest.mark.asyncio
    async def test_alert_deduplication(self, monitoring_service, mock_redis):
        """Test alert deduplication to prevent spam."""
        alert_data = {
            "type": "high_memory_usage",
            "metric": "memory_usage_percent",
            "current_value": 90.0,
            "threshold": 80.0
        }
        
        # Mock Redis to show alert recently sent
        mock_redis.get.return_value = datetime.utcnow().isoformat().encode()
        
        with patch.object(monitoring_service, 'redis_client', mock_redis):
            result = await monitoring_service.generate_alert(alert_data)
        
        # Should not send duplicate alert
        assert result is False
    
    @pytest.mark.asyncio
    async def test_metrics_cleanup_and_retention(self, monitoring_service, mock_redis):
        """Test metrics cleanup and retention policies."""
        # Mock old metrics data
        old_timestamp = (datetime.utcnow() - timedelta(days=31)).isoformat()
        mock_redis.lrange.return_value = [
            json.dumps({"timestamp": old_timestamp, "value": 30}).encode()
        ]
        
        with patch.object(monitoring_service, 'redis_client', mock_redis):
            cleaned = await monitoring_service.cleanup_old_metrics(retention_days=30)
        
        assert cleaned >= 0  # Should clean up old metrics
        mock_redis.ltrim.assert_called()  # Verify cleanup operation
    
    @pytest.mark.asyncio
    async def test_high_frequency_metrics_sampling(self, monitoring_service):
        """Test high-frequency metrics sampling to reduce overhead."""
        # Mock high-frequency metrics collection
        high_freq_metrics = []
        for i in range(1000):  # Simulate 1000 data points
            high_freq_metrics.append({
                "timestamp": datetime.utcnow() - timedelta(seconds=i),
                "cpu_usage": 30 + (i % 20)  # Varying CPU usage
            })
        
        with patch.object(monitoring_service, '_get_high_frequency_metrics', return_value=high_freq_metrics):
            sampled = await monitoring_service.sample_metrics(sample_rate=0.1)  # 10% sampling
        
        # Should significantly reduce the number of data points
        assert len(sampled) <= 100  # Should be roughly 10% of original
        assert len(sampled) > 0  # Should still have some data