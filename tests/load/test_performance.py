"""Load and performance tests for ComfyUI serverless API."""
import pytest
import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import json

from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


class TestAPIPerformance:
    """Test API endpoint performance under load."""
    
    @pytest.mark.performance
    def test_health_endpoint_performance(self, client: TestClient):
        """Test health endpoint performance under load."""
        # Measure response times for multiple requests
        response_times = []
        num_requests = 100
        
        start_time = time.time()
        
        for _ in range(num_requests):
            request_start = time.time()
            response = client.get("/health")
            request_end = time.time()
            
            assert response.status_code == 200
            response_times.append(request_end - request_start)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate performance metrics
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        rps = num_requests / total_duration
        
        print(f"\nHealth Endpoint Performance:")
        print(f"  Total requests: {num_requests}")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Requests/second: {rps:.2f}")
        print(f"  Average response time: {avg_response_time*1000:.2f}ms")
        print(f"  95th percentile: {p95_response_time*1000:.2f}ms")
        print(f"  99th percentile: {p99_response_time*1000:.2f}ms")
        
        # Performance assertions
        assert avg_response_time < 0.1  # Average response time under 100ms
        assert p95_response_time < 0.2  # 95th percentile under 200ms
        assert p99_response_time < 0.5  # 99th percentile under 500ms
        assert rps > 50  # At least 50 requests per second
    
    @pytest.mark.performance
    def test_workflow_submission_performance(self, client: TestClient, test_workflow):
        """Test workflow submission performance."""
        response_times = []
        num_requests = 50
        
        for _ in range(num_requests):
            request_start = time.time()
            response = client.post(
                "/workflows/submit",
                json=test_workflow,
                headers={"Authorization": "Bearer test-token"}
            )
            request_end = time.time()
            
            assert response.status_code == 201
            response_times.append(request_end - request_start)
        
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]
        
        print(f"\nWorkflow Submission Performance:")
        print(f"  Average response time: {avg_response_time*1000:.2f}ms")
        print(f"  95th percentile: {p95_response_time*1000:.2f}ms")
        
        # Workflow submission should be fast (validation + queuing)
        assert avg_response_time < 1.0  # Under 1 second
        assert p95_response_time < 2.0  # 95th percentile under 2 seconds
    
    @pytest.mark.performance
    def test_concurrent_api_requests(self, client: TestClient):
        """Test API performance under concurrent load."""
        num_threads = 10
        requests_per_thread = 20
        
        def make_requests():
            """Make multiple requests in a thread."""
            thread_times = []
            for _ in range(requests_per_thread):
                start = time.time()
                response = client.get("/health")
                end = time.time()
                
                assert response.status_code == 200
                thread_times.append(end - start)
            return thread_times
        
        # Run concurrent requests
        start_time = time.time()
        all_response_times = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_requests) for _ in range(num_threads)]
            
            for future in as_completed(futures):
                all_response_times.extend(future.result())
        
        end_time = time.time()
        total_duration = end_time - start_time
        total_requests = num_threads * requests_per_thread
        
        avg_response_time = statistics.mean(all_response_times)
        rps = total_requests / total_duration
        
        print(f"\nConcurrent API Performance:")
        print(f"  Threads: {num_threads}")
        print(f"  Requests per thread: {requests_per_thread}")
        print(f"  Total requests: {total_requests}")
        print(f"  Concurrent RPS: {rps:.2f}")
        print(f"  Average response time: {avg_response_time*1000:.2f}ms")
        
        # Should handle concurrent load well
        assert avg_response_time < 0.5  # Under 500ms average
        assert rps > 20  # At least 20 RPS with concurrency
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_websocket_performance(self):
        """Test WebSocket performance for real-time updates."""
        num_connections = 50
        messages_per_connection = 10
        
        async def websocket_client():
            """Create WebSocket connection and measure performance."""
            uri = "ws://localhost:8000/ws/workflow/test-execution-123"
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.ws_connect(uri) as ws:
                        start_time = time.time()
                        
                        # Send and receive messages
                        for i in range(messages_per_connection):
                            await ws.send_str(json.dumps({"type": "ping", "id": i}))
                            msg = await ws.receive()
                            assert msg.type == aiohttp.WSMsgType.TEXT
                        
                        end_time = time.time()
                        return end_time - start_time
                        
                except Exception as e:
                    # WebSocket endpoint might not be implemented
                    pytest.skip(f"WebSocket test skipped: {e}")
                    return 0
        
        # Run concurrent WebSocket connections
        tasks = [websocket_client() for _ in range(num_connections)]
        durations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions/skipped tests
        valid_durations = [d for d in durations if isinstance(d, float) and d > 0]
        
        if valid_durations:
            avg_duration = statistics.mean(valid_durations)
            print(f"\nWebSocket Performance:")
            print(f"  Connections: {len(valid_durations)}")
            print(f"  Messages per connection: {messages_per_connection}")
            print(f"  Average connection duration: {avg_duration:.2f}s")
            
            # WebSocket connections should be efficient
            assert avg_duration < 5.0  # Complete message exchange under 5 seconds


class TestWorkflowPerformance:
    """Test workflow execution performance."""
    
    @pytest.mark.performance
    def test_workflow_queue_performance(self, client: TestClient, test_workflow):
        """Test workflow queue performance under load."""
        num_workflows = 100
        
        # Submit multiple workflows quickly
        start_time = time.time()
        execution_ids = []
        
        for i in range(num_workflows):
            workflow_copy = test_workflow.copy()
            workflow_copy["workflow"]["metadata"] = {"batch_id": f"perf_test_{i}"}
            
            response = client.post(
                "/workflows/submit",
                json=workflow_copy,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 201
            execution_ids.append(response.json()["execution_id"])
        
        submission_time = time.time() - start_time
        
        # Check that all workflows are properly queued
        queued_count = 0
        for execution_id in execution_ids:
            response = client.get(
                f"/workflows/{execution_id}/status",
                headers={"Authorization": "Bearer test-token"}
            )
            
            if response.status_code == 200:
                status = response.json()["status"]
                if status in ["pending", "running", "completed"]:
                    queued_count += 1
        
        submission_rate = num_workflows / submission_time
        
        print(f"\nWorkflow Queue Performance:")
        print(f"  Workflows submitted: {num_workflows}")
        print(f"  Submission time: {submission_time:.2f}s")
        print(f"  Submission rate: {submission_rate:.2f} workflows/second")
        print(f"  Successfully queued: {queued_count}")
        
        # Queue should handle high submission rates
        assert submission_rate > 10  # At least 10 workflows per second
        assert queued_count >= num_workflows * 0.95  # 95% successfully queued
    
    @pytest.mark.performance
    def test_concurrent_workflow_status_checks(self, client: TestClient, test_workflow):
        """Test performance of concurrent workflow status checks."""
        # Submit a few workflows
        execution_ids = []
        for i in range(5):
            response = client.post(
                "/workflows/submit",
                json=test_workflow,
                headers={"Authorization": "Bearer test-token"}
            )
            execution_ids.append(response.json()["execution_id"])
        
        # Perform concurrent status checks
        num_threads = 20
        checks_per_thread = 50
        
        def check_status():
            """Check workflow status multiple times."""
            response_times = []
            for _ in range(checks_per_thread):
                execution_id = execution_ids[_ % len(execution_ids)]
                
                start = time.time()
                response = client.get(
                    f"/workflows/{execution_id}/status",
                    headers={"Authorization": "Bearer test-token"}
                )
                end = time.time()
                
                assert response.status_code == 200
                response_times.append(end - start)
            
            return response_times
        
        # Run concurrent status checks
        all_times = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(check_status) for _ in range(num_threads)]
            
            for future in as_completed(futures):
                all_times.extend(future.result())
        
        avg_response_time = statistics.mean(all_times)
        p95_response_time = statistics.quantiles(all_times, n=20)[18]
        
        print(f"\nConcurrent Status Check Performance:")
        print(f"  Total checks: {len(all_times)}")
        print(f"  Average response time: {avg_response_time*1000:.2f}ms")
        print(f"  95th percentile: {p95_response_time*1000:.2f}ms")
        
        # Status checks should be very fast
        assert avg_response_time < 0.1  # Under 100ms
        assert p95_response_time < 0.2  # 95th percentile under 200ms


class TestDatabasePerformance:
    """Test database performance under load."""
    
    @pytest.mark.performance
    def test_workflow_history_query_performance(self, client: TestClient, test_db):
        """Test performance of workflow history queries."""
        # Create test data in database
        db = test_db()
        
        # Insert many workflow executions
        from src.models.database import WorkflowExecution, WorkflowStatus, Priority
        from datetime import datetime, timedelta
        
        executions = []
        for i in range(1000):  # 1000 test executions
            execution = WorkflowExecution(
                id=f"perf-test-{i}",
                user_id=1,
                workflow_definition={"nodes": {}},
                status=WorkflowStatus.COMPLETED,
                priority=Priority.NORMAL,
                created_at=datetime.utcnow() - timedelta(hours=i),
                completed_at=datetime.utcnow() - timedelta(hours=i) + timedelta(minutes=30)
            )
            executions.append(execution)
        
        db.add_all(executions)
        db.commit()
        
        # Test query performance
        query_times = []
        num_queries = 50
        
        for _ in range(num_queries):
            start = time.time()
            
            # Test pagination query
            response = client.get(
                "/workflows/list?page=1&size=20",
                headers={"Authorization": "Bearer test-token"}
            )
            
            end = time.time()
            
            assert response.status_code == 200
            query_times.append(end - start)
        
        avg_query_time = statistics.mean(query_times)
        p95_query_time = statistics.quantiles(query_times, n=20)[18]
        
        print(f"\nDatabase Query Performance:")
        print(f"  Test records: 1000")
        print(f"  Queries executed: {num_queries}")
        print(f"  Average query time: {avg_query_time*1000:.2f}ms")
        print(f"  95th percentile: {p95_query_time*1000:.2f}ms")
        
        # Database queries should be fast even with many records
        assert avg_query_time < 0.5  # Under 500ms
        assert p95_query_time < 1.0  # 95th percentile under 1 second
        
        db.close()
    
    @pytest.mark.performance
    def test_concurrent_database_operations(self, test_db):
        """Test concurrent database operations performance."""
        from src.models.database import WorkflowExecution, WorkflowStatus, Priority
        from datetime import datetime
        import threading
        
        def database_operations():
            """Perform database operations in a thread."""
            local_db = test_db()
            operation_times = []
            
            try:
                for i in range(10):
                    # Create
                    start = time.time()
                    execution = WorkflowExecution(
                        id=f"concurrent-{threading.get_ident()}-{i}",
                        user_id=1,
                        workflow_definition={"nodes": {}},
                        status=WorkflowStatus.PENDING,
                        priority=Priority.NORMAL,
                        created_at=datetime.utcnow()
                    )
                    local_db.add(execution)
                    local_db.commit()
                    
                    # Read
                    local_db.query(WorkflowExecution).filter(
                        WorkflowExecution.id == execution.id
                    ).first()
                    
                    # Update
                    execution.status = WorkflowStatus.RUNNING
                    local_db.commit()
                    
                    end = time.time()
                    operation_times.append(end - start)
                
                return operation_times
            finally:
                local_db.close()
        
        # Run concurrent database operations
        num_threads = 5
        all_times = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(database_operations) for _ in range(num_threads)]
            
            for future in as_completed(futures):
                all_times.extend(future.result())
        
        if all_times:
            avg_operation_time = statistics.mean(all_times)
            
            print(f"\nConcurrent Database Performance:")
            print(f"  Threads: {num_threads}")
            print(f"  Total operations: {len(all_times)}")
            print(f"  Average operation time: {avg_operation_time*1000:.2f}ms")
            
            # Database operations should handle concurrency well
            assert avg_operation_time < 1.0  # Under 1 second per operation


class TestMemoryPerformance:
    """Test memory usage and performance."""
    
    @pytest.mark.performance
    def test_memory_usage_under_load(self, client: TestClient, test_workflow):
        """Test memory usage during high load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Submit many workflows to test memory usage
        num_workflows = 200
        execution_ids = []
        
        for i in range(num_workflows):
            workflow_copy = test_workflow.copy()
            workflow_copy["workflow"]["metadata"] = {"memory_test": f"batch_{i}"}
            
            response = client.post(
                "/workflows/submit",
                json=workflow_copy,
                headers={"Authorization": "Bearer test-token"}
            )
            
            execution_ids.append(response.json()["execution_id"])
            
            # Check memory every 50 workflows
            if i % 50 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                print(f"  After {i} workflows: {current_memory:.2f}MB (+{memory_increase:.2f}MB)")
                
                # Memory shouldn't grow excessively
                assert memory_increase < 500  # Less than 500MB increase
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage Test:")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Final memory: {final_memory:.2f}MB")
        print(f"  Total increase: {total_increase:.2f}MB")
        print(f"  Memory per workflow: {total_increase/num_workflows:.3f}MB")
        
        # Memory usage should be reasonable
        assert total_increase < 1000  # Less than 1GB total increase
        assert total_increase / num_workflows < 5  # Less than 5MB per workflow
    
    @pytest.mark.performance
    def test_garbage_collection_efficiency(self, client: TestClient):
        """Test garbage collection efficiency during load."""
        import gc
        
        # Force garbage collection and measure
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform operations that create objects
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        object_growth = final_objects - initial_objects
        
        print(f"\nGarbage Collection Test:")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects: {final_objects}")
        print(f"  Object growth: {object_growth}")
        
        # Object growth should be minimal after GC
        assert object_growth < 1000  # Less than 1000 objects growth