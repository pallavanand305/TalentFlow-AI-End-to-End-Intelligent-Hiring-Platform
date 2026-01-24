"""Property-based tests for background processing system"""

import pytest
import asyncio
import uuid
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta, timezone

from backend.app.core.task_queue import TaskQueue
from backend.app.services.background_processor import BackgroundProcessor
from backend.app.models.background_job import BackgroundJobStatus


class TestBackgroundProcessingProperties:
    """Property-based tests for background processing system"""
    
    @given(
        task_type=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        task_data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            min_size=1,
            max_size=10
        ),
        priority=st.integers(min_value=0, max_value=100),
        delay_seconds=st.integers(min_value=0, max_value=3600)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_25_async_operation_acknowledgment(
        self, task_type, task_data, priority, delay_seconds
    ):
        """
        **Feature: talentflow-ai, Property 25: Async operation acknowledgment**
        For any long-running operation (resume parsing, batch scoring), 
        the API should return an immediate response with a job ID rather than blocking until completion.
        **Validates: Requirements 4.5**
        """
        # Create mock background processor
        mock_background_processor = BackgroundProcessor()
        
        # Create mock task queue
        mock_task_queue = TaskQueue("redis://localhost:6379/1")
        mock_task_queue._redis = AsyncMock()
        mock_background_processor.task_queue = mock_task_queue
        mock_background_processor.job_repository = AsyncMock()
        
        # Mock the enqueue operation to return immediately
        expected_job_id = str(uuid.uuid4())
        mock_background_processor.task_queue.enqueue_task = AsyncMock(return_value=expected_job_id)
        mock_background_processor.job_repository.create_job = AsyncMock()
        
        # Measure response time
        start_time = datetime.now(timezone.utc)
        
        job_id = await mock_background_processor.enqueue_task(
            task_type=task_type,
            task_data=task_data,
            priority=priority,
            delay_seconds=delay_seconds
        )
        
        end_time = datetime.now(timezone.utc)
        response_time = (end_time - start_time).total_seconds()
        
        # Verify immediate response (should be very fast)
        assert response_time < 1.0, "Task enqueueing should be immediate"
        
        # Verify job ID is returned
        assert job_id is not None
        assert isinstance(job_id, str)
        assert len(job_id) > 0
        
        # Verify task was enqueued
        mock_background_processor.task_queue.enqueue_task.assert_called_once_with(
            task_type=task_type,
            task_data=task_data,
            priority=priority,
            delay_seconds=delay_seconds
        )
    
    @given(
        task_type=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        task_data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers()),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_32_async_task_execution(
        self, task_type, task_data
    ):
        """
        **Feature: talentflow-ai, Property 32: Async task execution**
        For any background task (resume parsing or batch scoring), 
        the task should execute asynchronously without blocking the API response.
        **Validates: Requirements 6.1, 6.2**
        """
        # Create mock background processor
        mock_background_processor = BackgroundProcessor()
        
        # Create mock task queue
        mock_task_queue = TaskQueue("redis://localhost:6379/1")
        mock_task_queue._redis = AsyncMock()
        mock_background_processor.task_queue = mock_task_queue
        mock_background_processor.job_repository = AsyncMock()
        
        # Register a test handler that simulates work
        async def test_handler(data):
            await asyncio.sleep(0.1)  # Simulate work
            return {"result": "processed", "input": data}
        
        mock_background_processor.register_task_handler(task_type, test_handler)
        
        # Mock dependencies
        job_id = str(uuid.uuid4())
        mock_background_processor.task_queue.enqueue_task = AsyncMock(return_value=job_id)
        mock_background_processor.job_repository.create_job = AsyncMock()
        mock_background_processor.job_repository.update_job_status = AsyncMock()
        mock_background_processor.task_queue.set_task_status = AsyncMock()
        
        # Enqueue task
        returned_job_id = await mock_background_processor.enqueue_task(task_type, task_data)
        
        # Verify task was enqueued immediately
        assert returned_job_id == job_id
        
        # Simulate task processing
        task_payload = {
            "job_id": job_id,
            "task_type": task_type,
            "task_data": task_data
        }
        
        # Process task asynchronously
        start_time = datetime.now(timezone.utc)
        await mock_background_processor._process_task(task_payload, worker_id=0)
        end_time = datetime.now(timezone.utc)
        
        # Verify task executed (took some time due to sleep)
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time >= 0.1, "Task should have executed with simulated work"
        
        # Verify status updates were called
        mock_background_processor.job_repository.update_job_status.assert_called()
    
    @given(
        num_tasks=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_33_job_id_generation(
        self, num_tasks
    ):
        """
        **Feature: talentflow-ai, Property 33: Job ID generation**
        For any queued background task, the system should return a unique job ID 
        that can be used for status tracking.
        **Validates: Requirements 6.3**
        """
        # Create mock background processor
        mock_background_processor = BackgroundProcessor()
        
        # Create mock task queue
        mock_task_queue = TaskQueue("redis://localhost:6379/1")
        mock_task_queue._redis = AsyncMock()
        mock_background_processor.task_queue = mock_task_queue
        mock_background_processor.job_repository = AsyncMock()
        
        # Mock dependencies
        mock_background_processor.task_queue.enqueue_task = AsyncMock()
        mock_background_processor.job_repository.create_job = AsyncMock()
        
        job_ids = set()
        
        # Generate multiple job IDs
        for i in range(num_tasks):
            job_id = str(uuid.uuid4())
            mock_background_processor.task_queue.enqueue_task.return_value = job_id
            
            returned_job_id = await mock_background_processor.enqueue_task(
                task_type=f"test_task_{i}",
                task_data={"index": i}
            )
            
            # Verify job ID is valid
            assert returned_job_id is not None
            assert isinstance(returned_job_id, str)
            assert len(returned_job_id) > 0
            
            # Verify uniqueness
            assert returned_job_id not in job_ids, f"Job ID {returned_job_id} is not unique"
            job_ids.add(returned_job_id)
        
        # Verify all job IDs are unique
        assert len(job_ids) == num_tasks
    
    @given(
        job_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        status=st.sampled_from(list(BackgroundJobStatus)),
        has_result=st.booleans(),
        has_error=st.booleans()
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_34_job_status_tracking(
        self, job_id, status, has_result, has_error
    ):
        """
        **Feature: talentflow-ai, Property 34: Job status tracking**
        For any background job ID, querying the status should return the current state 
        (queued, processing, completed, failed) and progress information.
        **Validates: Requirements 6.4**
        """
        # Create mock background processor
        mock_background_processor = BackgroundProcessor()
        mock_background_processor.job_repository = AsyncMock()
        
        # Mock repository response
        mock_job = AsyncMock()
        mock_job.id = job_id
        mock_job.job_type = "test_task"
        mock_job.status = status
        mock_job.input_data = {"test": "data"}
        mock_job.result_data = {"result": "success"} if has_result else None
        mock_job.error_message = "Test error" if has_error else None
        mock_job.created_at = datetime.now(timezone.utc)
        mock_job.started_at = datetime.now(timezone.utc) if status != BackgroundJobStatus.QUEUED else None
        mock_job.completed_at = datetime.now(timezone.utc) if status in [BackgroundJobStatus.COMPLETED, BackgroundJobStatus.FAILED] else None
        mock_job.updated_at = datetime.now(timezone.utc)
        
        mock_background_processor.job_repository.get_job_by_id = AsyncMock(return_value=mock_job)
        
        # Get status
        result = await mock_background_processor.get_task_status(job_id)
        
        # Verify status structure
        assert result is not None
        assert result["job_id"] == job_id
        assert result["status"] == status.value
        assert "job_type" in result
        assert "created_at" in result
        assert "updated_at" in result
        
        # Verify conditional fields
        if status != BackgroundJobStatus.QUEUED:
            assert "started_at" in result
        
        if status in [BackgroundJobStatus.COMPLETED, BackgroundJobStatus.FAILED]:
            assert "completed_at" in result
        
        if has_result:
            assert result["result_data"] is not None
        
        if has_error:
            assert result["error_message"] is not None
    
    @given(
        job_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        error_message=st.text(min_size=1, max_size=500)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_35_job_failure_handling(
        self, job_id, error_message
    ):
        """
        **Feature: talentflow-ai, Property 35: Job failure handling**
        For any background job that fails, the system should update the job status to 'failed', 
        log the error message, and make the error retrievable via the job status endpoint.
        **Validates: Requirements 6.5**
        """
        # Create mock background processor
        mock_background_processor = BackgroundProcessor()
        mock_background_processor.job_repository = AsyncMock()
        mock_background_processor.task_queue = AsyncMock()
        
        # Register a failing handler
        async def failing_handler(data):
            raise ValueError(error_message)
        
        mock_background_processor.register_task_handler("failing_task", failing_handler)
        
        # Mock dependencies
        mock_background_processor.job_repository.update_job_status = AsyncMock()
        mock_background_processor.task_queue.set_task_status = AsyncMock()
        
        task_payload = {
            "job_id": job_id,
            "task_type": "failing_task",
            "task_data": {"test": "data"}
        }
        
        # Process the failing task
        await mock_background_processor._process_task(task_payload, worker_id=0)
        
        # Verify failure handling
        mock_background_processor.job_repository.update_job_status.assert_called()
        mock_background_processor.task_queue.set_task_status.assert_called()
        
        # Check that status was updated to failed
        update_calls = mock_background_processor.job_repository.update_job_status.call_args_list
        failed_call = None
        for call in update_calls:
            # Check both positional and keyword arguments
            if len(call.args) > 1 and call.args[1] == BackgroundJobStatus.FAILED:
                failed_call = call
                break
            elif call.kwargs.get('status') == BackgroundJobStatus.FAILED:
                failed_call = call
                break
        
        assert failed_call is not None, "Job status should be updated to FAILED"
        
        # Check job ID - it could be in args or kwargs
        if len(failed_call.args) > 0:
            assert failed_call.args[0] == job_id, "Correct job ID should be updated"
        else:
            assert failed_call.kwargs.get('job_id') == job_id, "Correct job ID should be updated"
        
        # Check error message - it could be in args or kwargs
        error_msg = None
        if len(failed_call.args) > 3:
            error_msg = failed_call.args[3]
        else:
            error_msg = failed_call.kwargs.get('error_message')
        
        assert error_msg is not None, "Error message should be provided"
        assert error_message in error_msg, "Error message should be preserved"
    
    @given(
        max_retries=st.integers(min_value=1, max_value=10),
        current_retries=st.integers(min_value=0, max_value=15)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_36_retry_logic_for_transient_failures(
        self, max_retries, current_retries
    ):
        """
        **Feature: talentflow-ai, Property 36: Retry logic for transient failures**
        For any background job that fails with a transient error, the system should 
        automatically retry the operation before marking it as permanently failed.
        **Validates: Requirements 6.6**
        """
        # Create mock task queue
        mock_task_queue = TaskQueue("redis://localhost:6379/1")
        mock_task_queue._redis = AsyncMock()
        
        job_id = str(uuid.uuid4())
        
        # Mock Redis operations
        mock_task_queue._redis.get = AsyncMock(return_value=str(current_retries))
        mock_task_queue._redis.setex = AsyncMock()
        
        # Test retry logic
        result = await mock_task_queue.retry_failed_task(job_id, max_retries)
        
        if current_retries < max_retries:
            # Should allow retry
            assert result is True, f"Should allow retry when current_retries ({current_retries}) < max_retries ({max_retries})"
            mock_task_queue._redis.setex.assert_called()
        else:
            # Should deny retry
            assert result is False, f"Should deny retry when current_retries ({current_retries}) >= max_retries ({max_retries})"
    
    @given(
        task_types=st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            min_size=1,
            max_size=10,
            unique=True
        )
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_task_handler_registration_uniqueness(
        self, task_types
    ):
        """Test that task handlers can be registered and are unique per task type"""
        # Create mock background processor
        mock_background_processor = BackgroundProcessor()
        
        handlers = {}
        
        # Register handlers for each task type
        for task_type in task_types:
            async def handler(data, task_type=task_type):  # Capture task_type in closure
                return {"task_type": task_type, "data": data}
            
            mock_background_processor.register_task_handler(task_type, handler)
            handlers[task_type] = handler
        
        # Verify all handlers are registered
        for task_type in task_types:
            assert task_type in mock_background_processor.task_handlers
            assert mock_background_processor.task_handlers[task_type] == handlers[task_type]
        
        # Verify handler count matches
        assert len(mock_background_processor.task_handlers) >= len(task_types)