"""Integration tests for background job system"""

import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

from backend.app.main import app
from backend.app.services.background_processor import background_processor
from backend.app.models.background_job import BackgroundJobStatus
from tests.conftest import create_test_user, get_auth_headers


class TestBackgroundJobsAPI:
    """Integration tests for background jobs API"""
    
    @pytest.mark.asyncio
    async def test_get_job_status_success(self, async_client: AsyncClient, test_user):
        """Test getting job status successfully"""
        # Create a test job in the database
        job_id = "test-job-123"
        
        with patch.object(background_processor, 'get_task_status') as mock_get_status:
            mock_get_status.return_value = {
                "job_id": job_id,
                "job_type": "resume_parsing",
                "status": "completed",
                "input_data": {"candidate_id": "123"},
                "result_data": {"parsed_data": {"name": "John Doe"}},
                "error_message": None,
                "created_at": "2024-01-15T10:00:00Z",
                "started_at": "2024-01-15T10:00:01Z",
                "completed_at": "2024-01-15T10:00:05Z",
                "updated_at": "2024-01-15T10:00:05Z"
            }
            
            headers = get_auth_headers(test_user)
            response = await async_client.get(f"/api/v1/jobs/status/{job_id}", headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["job_id"] == job_id
            assert data["data"]["job_type"] == "resume_parsing"
            assert data["data"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, async_client: AsyncClient, test_user):
        """Test getting status for non-existent job"""
        job_id = "nonexistent-job"
        
        with patch.object(background_processor, 'get_task_status') as mock_get_status:
            mock_get_status.return_value = None
            
            headers = get_auth_headers(test_user)
            response = await async_client.get(f"/api/v1/jobs/status/{job_id}", headers=headers)
            
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_get_job_status_unauthorized(self, async_client: AsyncClient):
        """Test getting job status without authentication"""
        job_id = "test-job-123"
        
        response = await async_client.get(f"/api/v1/jobs/status/{job_id}")
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_get_queue_stats_admin(self, async_client: AsyncClient):
        """Test getting queue stats as admin user"""
        # Create admin user
        admin_user = await create_test_user(
            username="admin_user",
            email="admin@test.com",
            role="admin"
        )
        
        with patch.object(background_processor, 'get_queue_stats') as mock_get_stats:
            mock_get_stats.return_value = {
                "redis_queue": {
                    "queued_tasks": 5,
                    "delayed_tasks": 2,
                    "total_tasks": 7
                },
                "database_jobs": {
                    "queued": 3,
                    "processing": 1,
                    "completed": 10,
                    "failed": 2
                },
                "workers_running": 2,
                "is_processing": True
            }
            
            headers = get_auth_headers(admin_user)
            response = await async_client.get("/api/v1/jobs/stats", headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert "redis_queue" in data["data"]
            assert "database_jobs" in data["data"]
            assert data["data"]["workers_running"] == 2
    
    @pytest.mark.asyncio
    async def test_get_queue_stats_non_admin(self, async_client: AsyncClient, test_user):
        """Test getting queue stats as non-admin user"""
        headers = get_auth_headers(test_user)
        response = await async_client.get("/api/v1/jobs/stats", headers=headers)
        
        assert response.status_code == 403
        data = response.json()
        assert "admin access required" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_cleanup_old_jobs_admin(self, async_client: AsyncClient):
        """Test cleaning up old jobs as admin user"""
        # Create admin user
        admin_user = await create_test_user(
            username="admin_cleanup",
            email="admin_cleanup@test.com",
            role="admin"
        )
        
        with patch.object(background_processor, 'cleanup_old_jobs') as mock_cleanup:
            mock_cleanup.return_value = 5  # 5 jobs deleted
            
            headers = get_auth_headers(admin_user)
            response = await async_client.post(
                "/api/v1/jobs/cleanup?days_old=30",
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["deleted_jobs"] == 5
            assert data["data"]["days_old"] == 30
            assert "cleaned up" in data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_cleanup_old_jobs_non_admin(self, async_client: AsyncClient, test_user):
        """Test cleaning up old jobs as non-admin user"""
        headers = get_auth_headers(test_user)
        response = await async_client.post("/api/v1/jobs/cleanup", headers=headers)
        
        assert response.status_code == 403
        data = response.json()
        assert "admin access required" in data["detail"].lower()


class TestBackgroundProcessor:
    """Integration tests for BackgroundProcessor service"""
    
    @pytest.mark.asyncio
    async def test_enqueue_and_get_status(self):
        """Test enqueueing a task and getting its status"""
        task_type = "test_task"
        task_data = {"test": "data"}
        
        # Mock the task queue and repository
        with patch.object(background_processor.task_queue, 'enqueue_task') as mock_enqueue, \
             patch.object(background_processor.job_repository, 'create_job') as mock_create:
            
            mock_enqueue.return_value = "test-job-123"
            mock_create.return_value = AsyncMock()
            
            job_id = await background_processor.enqueue_task(task_type, task_data)
            
            assert job_id == "test-job-123"
            mock_enqueue.assert_called_once_with(
                task_type=task_type,
                task_data=task_data,
                priority=0,
                delay_seconds=0
            )
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_handler_registration(self):
        """Test registering task handlers"""
        from backend.app.services.background_processor import task_handler
        
        @task_handler("test_handler")
        async def test_task_handler(data):
            return {"result": "success"}
        
        # Verify handler was registered
        assert "test_handler" in background_processor.task_handlers
        assert background_processor.task_handlers["test_handler"] == test_task_handler
    
    @pytest.mark.asyncio
    async def test_worker_lifecycle(self):
        """Test starting and stopping workers"""
        # Mock the worker loop to prevent actual processing
        with patch.object(background_processor, '_worker_loop') as mock_worker_loop:
            mock_worker_loop.return_value = None
            
            # Start workers
            await background_processor.start_workers(num_workers=1)
            
            assert background_processor.is_running is True
            assert len(background_processor.worker_tasks) == 1
            
            # Stop workers
            await background_processor.stop_workers()
            
            assert background_processor.is_running is False
            assert len(background_processor.worker_tasks) == 0


class TestTaskProcessing:
    """Integration tests for actual task processing"""
    
    @pytest.mark.asyncio
    async def test_resume_parsing_task_flow(self):
        """Test complete resume parsing task flow"""
        # Register a test handler
        @background_processor.register_task_handler("resume_parsing")
        async def test_resume_handler(task_data):
            return {
                "candidate_id": task_data["candidate_id"],
                "parsed_data": {"name": "Test User"}
            }
        
        task_data = {"candidate_id": "123", "file_path": "/test/resume.pdf"}
        
        # Mock dependencies
        with patch.object(background_processor.task_queue, 'enqueue_task') as mock_enqueue, \
             patch.object(background_processor.job_repository, 'create_job') as mock_create, \
             patch.object(background_processor.task_queue, 'dequeue_task') as mock_dequeue, \
             patch.object(background_processor.job_repository, 'update_job_status') as mock_update:
            
            # Setup mocks
            job_id = "resume-job-123"
            mock_enqueue.return_value = job_id
            mock_create.return_value = AsyncMock()
            mock_dequeue.return_value = {
                "job_id": job_id,
                "task_type": "resume_parsing",
                "task_data": task_data
            }
            mock_update.return_value = AsyncMock()
            
            # Enqueue task
            returned_job_id = await background_processor.enqueue_task(
                "resume_parsing", task_data
            )
            
            assert returned_job_id == job_id
            
            # Simulate processing
            task_payload = await background_processor.task_queue.dequeue_task()
            assert task_payload is not None
            
            # Process the task (this would normally be done by a worker)
            await background_processor._process_task(task_payload, worker_id=0)
            
            # Verify status updates were called
            assert mock_update.call_count >= 2  # Processing and Completed
    
    @pytest.mark.asyncio
    async def test_task_failure_handling(self):
        """Test handling of task failures"""
        # Register a failing handler
        @background_processor.register_task_handler("failing_task")
        async def failing_handler(task_data):
            raise ValueError("Task failed intentionally")
        
        task_data = {"test": "data"}
        job_id = "failing-job-123"
        
        # Mock dependencies
        with patch.object(background_processor.job_repository, 'update_job_status') as mock_update, \
             patch.object(background_processor.task_queue, 'set_task_status') as mock_set_status:
            
            mock_update.return_value = AsyncMock()
            mock_set_status.return_value = None
            
            task_payload = {
                "job_id": job_id,
                "task_type": "failing_task",
                "task_data": task_data
            }
            
            # Process the failing task
            await background_processor._process_task(task_payload, worker_id=0)
            
            # Verify failure was handled
            mock_update.assert_called()
            mock_set_status.assert_called()
            
            # Check that status was set to failed
            update_calls = mock_update.call_args_list
            failed_call = None
            for call in update_calls:
                if call[1].get('status') == BackgroundJobStatus.FAILED:
                    failed_call = call
                    break
            
            assert failed_call is not None
            assert "Task failed" in failed_call[1]['error_message']