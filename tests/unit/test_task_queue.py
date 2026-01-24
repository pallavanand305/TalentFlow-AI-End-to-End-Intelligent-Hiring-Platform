"""Unit tests for task queue functionality"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta, timezone

from backend.app.core.task_queue import TaskQueue
from backend.app.models.background_job import BackgroundJobStatus


class TestTaskQueue:
    """Test cases for TaskQueue class"""
    
    @pytest.fixture
    async def task_queue(self):
        """Create a task queue instance for testing"""
        queue = TaskQueue("redis://localhost:6379/1")  # Use test database
        
        # Mock Redis connection for unit tests
        mock_redis = AsyncMock()
        queue._redis = mock_redis
        
        yield queue
        
        # Cleanup
        if queue._redis:
            await queue.disconnect()
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful Redis connection"""
        queue = TaskQueue("redis://localhost:6379/1")
        
        with patch('redis.asyncio.from_url') as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            await queue.connect()
            
            assert queue._redis is not None
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test Redis connection failure"""
        queue = TaskQueue("redis://invalid:6379/1")
        
        with patch('redis.asyncio.from_url') as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_from_url.return_value = mock_redis
            
            with pytest.raises(Exception, match="Connection failed"):
                await queue.connect()
    
    @pytest.mark.asyncio
    async def test_enqueue_task_immediate(self, task_queue):
        """Test enqueueing an immediate task"""
        task_type = "resume_parsing"
        task_data = {"candidate_id": "123", "file_path": "/path/to/resume.pdf"}
        priority = 5
        
        # Mock Redis operations
        task_queue._redis.zadd = AsyncMock()
        task_queue._redis.setex = AsyncMock()
        
        job_id = await task_queue.enqueue_task(task_type, task_data, priority)
        
        # Verify job_id is returned
        assert job_id is not None
        assert isinstance(job_id, str)
        
        # Verify Redis operations were called
        task_queue._redis.zadd.assert_called_once()
        task_queue._redis.setex.assert_called_once()
        
        # Verify task payload structure
        zadd_call = task_queue._redis.zadd.call_args
        queue_name, task_mapping = zadd_call[0]
        assert queue_name == task_queue.queue_name
        
        # Parse the task payload
        task_json = list(task_mapping.keys())[0]
        task_payload = json.loads(task_json)
        
        assert task_payload["job_id"] == job_id
        assert task_payload["task_type"] == task_type
        assert task_payload["task_data"] == task_data
        assert task_payload["priority"] == priority
    
    @pytest.mark.asyncio
    async def test_enqueue_task_delayed(self, task_queue):
        """Test enqueueing a delayed task"""
        task_type = "batch_scoring"
        task_data = {"job_id": "456"}
        delay_seconds = 60
        
        # Mock Redis operations
        task_queue._redis.zadd = AsyncMock()
        task_queue._redis.setex = AsyncMock()
        
        job_id = await task_queue.enqueue_task(
            task_type, task_data, delay_seconds=delay_seconds
        )
        
        # Verify job_id is returned
        assert job_id is not None
        
        # Verify delayed queue was used
        zadd_call = task_queue._redis.zadd.call_args
        queue_name = zadd_call[0][0]
        assert queue_name == f"{task_queue.queue_name}:delayed"
    
    @pytest.mark.asyncio
    async def test_dequeue_task_success(self, task_queue):
        """Test successful task dequeue"""
        # Mock task payload
        job_id = "test-job-123"
        task_payload = {
            "job_id": job_id,
            "task_type": "resume_parsing",
            "task_data": {"candidate_id": "123"},
            "priority": 0,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Mock Redis operations
        task_queue._redis.zrangebyscore = AsyncMock(return_value=[])  # No delayed tasks
        task_queue._redis.zrem = AsyncMock()
        task_queue._redis.pipeline = AsyncMock()
        task_queue._redis.bzpopmax = AsyncMock(
            return_value=(task_queue.queue_name, json.dumps(task_payload), 0)
        )
        task_queue._redis.setex = AsyncMock()
        
        result = await task_queue.dequeue_task(timeout=1)
        
        # Verify result
        assert result is not None
        assert result["job_id"] == job_id
        assert result["task_type"] == "resume_parsing"
        
        # Verify status was updated
        task_queue._redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dequeue_task_timeout(self, task_queue):
        """Test task dequeue timeout"""
        # Mock Redis operations
        task_queue._redis.zrangebyscore = AsyncMock(return_value=[])
        task_queue._redis.bzpopmax = AsyncMock(return_value=None)
        
        result = await task_queue.dequeue_task(timeout=1)
        
        # Verify no task returned
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_task_status(self, task_queue):
        """Test setting task status"""
        job_id = "test-job-123"
        status = BackgroundJobStatus.COMPLETED
        result_data = {"result": "success"}
        
        # Mock Redis operations
        task_queue._redis.setex = AsyncMock()
        
        await task_queue.set_task_status(job_id, status, result_data)
        
        # Verify Redis operation
        task_queue._redis.setex.assert_called_once()
        
        # Verify status data structure
        setex_call = task_queue._redis.setex.call_args
        key, ttl, status_json = setex_call[0]
        
        assert key == f"{task_queue.status_prefix}:{job_id}"
        assert ttl == timedelta(days=7)
        
        status_data = json.loads(status_json)
        assert status_data["status"] == status.value
        assert status_data["result_data"] == result_data
        assert "updated_at" in status_data
    
    @pytest.mark.asyncio
    async def test_get_task_status_found(self, task_queue):
        """Test getting task status when found"""
        job_id = "test-job-123"
        status_data = {
            "status": "completed",
            "result_data": {"result": "success"},
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Mock Redis operations
        task_queue._redis.get = AsyncMock(return_value=json.dumps(status_data))
        
        result = await task_queue.get_task_status(job_id)
        
        # Verify result
        assert result is not None
        assert result["status"] == "completed"
        assert result["result_data"] == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, task_queue):
        """Test getting task status when not found"""
        job_id = "nonexistent-job"
        
        # Mock Redis operations
        task_queue._redis.get = AsyncMock(return_value=None)
        
        result = await task_queue.get_task_status(job_id)
        
        # Verify no result
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self, task_queue):
        """Test getting queue statistics"""
        # Mock Redis operations
        task_queue._redis.zcard = AsyncMock(side_effect=[5, 2])  # queued, delayed
        
        stats = await task_queue.get_queue_stats()
        
        # Verify stats structure
        assert "queued_tasks" in stats
        assert "delayed_tasks" in stats
        assert "total_tasks" in stats
        assert stats["queued_tasks"] == 5
        assert stats["delayed_tasks"] == 2
        assert stats["total_tasks"] == 7
    
    @pytest.mark.asyncio
    async def test_clear_queue(self, task_queue):
        """Test clearing the queue"""
        # Mock Redis operations
        task_queue._redis.delete = AsyncMock()
        
        await task_queue.clear_queue()
        
        # Verify Redis operations
        assert task_queue._redis.delete.call_count == 2
        
        # Verify correct queues were deleted
        delete_calls = [call[0][0] for call in task_queue._redis.delete.call_args_list]
        assert task_queue.queue_name in delete_calls
        assert f"{task_queue.queue_name}:delayed" in delete_calls
    
    @pytest.mark.asyncio
    async def test_move_ready_delayed_tasks(self, task_queue):
        """Test moving ready delayed tasks to main queue"""
        # Mock delayed tasks
        current_time = datetime.now(timezone.utc).timestamp()
        ready_task = {
            "job_id": "delayed-job-123",
            "task_type": "resume_parsing",
            "priority": 3
        }
        
        # Mock Redis operations
        task_queue._redis.zrangebyscore = AsyncMock(
            return_value=[(json.dumps(ready_task), current_time - 10)]
        )
        
        # Create a proper pipeline mock
        mock_pipeline = AsyncMock()
        mock_pipeline.zadd = AsyncMock(return_value=mock_pipeline)  # Return self for chaining
        mock_pipeline.zrem = AsyncMock(return_value=mock_pipeline)  # Return self for chaining
        mock_pipeline.execute = AsyncMock()
        
        # Mock the pipeline method to return our mock pipeline
        task_queue._redis.pipeline = MagicMock(return_value=mock_pipeline)
        
        await task_queue._move_ready_delayed_tasks()
        
        # Verify pipeline operations
        task_queue._redis.pipeline.assert_called_once()
        mock_pipeline.zadd.assert_called_once()
        mock_pipeline.zrem.assert_called_once()
        mock_pipeline.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retry_failed_task_within_limit(self, task_queue):
        """Test retrying a failed task within retry limit"""
        job_id = "failed-job-123"
        max_retries = 3
        
        # Mock Redis operations
        task_queue._redis.get = AsyncMock(return_value="1")  # Current retry count
        task_queue._redis.setex = AsyncMock()
        
        result = await task_queue.retry_failed_task(job_id, max_retries)
        
        # Verify retry was allowed
        assert result is True
        
        # Verify retry count was incremented
        task_queue._redis.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_retry_failed_task_exceeds_limit(self, task_queue):
        """Test retrying a failed task that exceeds retry limit"""
        job_id = "failed-job-123"
        max_retries = 3
        
        # Mock Redis operations
        task_queue._redis.get = AsyncMock(return_value="3")  # At max retries
        
        result = await task_queue.retry_failed_task(job_id, max_retries)
        
        # Verify retry was denied
        assert result is False