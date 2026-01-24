"""Background processor service for managing async tasks"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import traceback

from backend.app.core.task_queue import task_queue, TaskQueue
from backend.app.repositories.background_job_repository import BackgroundJobRepository
from backend.app.models.background_job import BackgroundJobStatus
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class BackgroundProcessor:
    """Service for managing background task processing"""
    
    def __init__(self):
        self.task_queue = task_queue
        self.job_repository = BackgroundJobRepository()
        self.task_handlers: Dict[str, Callable] = {}
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.max_retries = 3
        self.retry_delay = 60  # seconds
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """
        Register a handler function for a specific task type
        
        Args:
            task_type: Type of task (e.g., 'resume_parsing', 'batch_scoring')
            handler: Async function to handle the task
        """
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def enqueue_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        priority: int = 0,
        delay_seconds: int = 0
    ) -> str:
        """
        Enqueue a task for background processing
        
        Args:
            task_type: Type of task
            task_data: Task input data
            priority: Task priority (higher = more priority)
            delay_seconds: Delay before task becomes available
            
        Returns:
            job_id: Unique identifier for the task
        """
        try:
            # Enqueue in Redis
            job_id = await self.task_queue.enqueue_task(
                task_type=task_type,
                task_data=task_data,
                priority=priority,
                delay_seconds=delay_seconds
            )
            
            # Create database record
            await self.job_repository.create_job(
                job_type=task_type,
                input_data=task_data,
                job_id=job_id
            )
            
            logger.info(f"Enqueued task {task_type} with job_id {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task_type}: {e}")
            raise
    
    async def get_task_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status from database
        
        Args:
            job_id: Task job ID
            
        Returns:
            Task status information
        """
        try:
            job = await self.job_repository.get_job_by_id(job_id)
            
            if not job:
                return None
            
            return {
                "job_id": str(job.id),
                "job_type": job.job_type,
                "status": job.status.value,
                "input_data": job.input_data,
                "result_data": job.result_data,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "updated_at": job.updated_at.isoformat() if job.updated_at else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get task status for {job_id}: {e}")
            raise
    
    async def start_workers(self, num_workers: int = 2) -> None:
        """
        Start background worker processes
        
        Args:
            num_workers: Number of worker processes to start
        """
        if self.is_running:
            logger.warning("Workers are already running")
            return
        
        self.is_running = True
        logger.info(f"Starting {num_workers} background workers")
        
        # Start worker tasks
        for i in range(num_workers):
            worker_task = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"background_worker_{i}"
            )
            self.worker_tasks.append(worker_task)
        
        logger.info(f"Started {num_workers} background workers")
    
    async def stop_workers(self) -> None:
        """Stop all background workers"""
        if not self.is_running:
            logger.warning("Workers are not running")
            return
        
        logger.info("Stopping background workers")
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("Stopped all background workers")
    
    async def _worker_loop(self, worker_id: int) -> None:
        """
        Main worker loop for processing tasks
        
        Args:
            worker_id: Unique identifier for this worker
        """
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Dequeue task with timeout
                task_payload = await self.task_queue.dequeue_task(timeout=5)
                
                if not task_payload:
                    continue  # No tasks available, continue loop
                
                await self._process_task(task_payload, worker_id)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                # Continue processing other tasks
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task_payload: Dict[str, Any], worker_id: int) -> None:
        """
        Process a single task
        
        Args:
            task_payload: Task data from queue
            worker_id: ID of the worker processing the task
        """
        job_id = task_payload["job_id"]
        task_type = task_payload["task_type"]
        task_data = task_payload["task_data"]
        
        logger.info(f"Worker {worker_id} processing task {task_type} (job_id: {job_id})")
        
        try:
            # Update status to processing in database
            await self.job_repository.update_job_status(
                job_id=job_id,
                status=BackgroundJobStatus.PROCESSING
            )
            
            # Get task handler
            handler = self.task_handlers.get(task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task_type}")
            
            # Execute task
            start_time = datetime.utcnow()
            result = await handler(task_data)
            end_time = datetime.utcnow()
            
            # Update status to completed
            await self.job_repository.update_job_status(
                job_id=job_id,
                status=BackgroundJobStatus.COMPLETED,
                result_data={
                    "result": result,
                    "processing_time_seconds": (end_time - start_time).total_seconds()
                }
            )
            
            # Update Redis status
            await self.task_queue.set_task_status(
                job_id=job_id,
                status=BackgroundJobStatus.COMPLETED,
                result_data={"result": result}
            )
            
            logger.info(f"Worker {worker_id} completed task {task_type} (job_id: {job_id})")
            
        except Exception as e:
            error_message = f"Task failed: {str(e)}"
            error_traceback = traceback.format_exc()
            
            logger.error(f"Worker {worker_id} failed task {task_type} (job_id: {job_id}): {error_message}")
            logger.debug(f"Task failure traceback: {error_traceback}")
            
            # Update status to failed
            await self.job_repository.update_job_status(
                job_id=job_id,
                status=BackgroundJobStatus.FAILED,
                error_message=error_message
            )
            
            # Update Redis status
            await self.task_queue.set_task_status(
                job_id=job_id,
                status=BackgroundJobStatus.FAILED,
                error_message=error_message
            )
            
            # TODO: Implement retry logic here if needed
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive queue statistics
        
        Returns:
            Dictionary with queue and job statistics
        """
        try:
            # Get Redis queue stats
            redis_stats = await self.task_queue.get_queue_stats()
            
            # Get database job stats
            db_stats = await self.job_repository.get_job_statistics()
            
            return {
                "redis_queue": redis_stats,
                "database_jobs": db_stats,
                "workers_running": len(self.worker_tasks),
                "is_processing": self.is_running
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            raise
    
    async def cleanup_old_jobs(self, days_old: int = 30) -> int:
        """
        Clean up old completed/failed jobs
        
        Args:
            days_old: Delete jobs older than this many days
            
        Returns:
            Number of jobs cleaned up
        """
        try:
            return await self.job_repository.cleanup_old_jobs(days_old)
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
            raise


# Global background processor instance
background_processor = BackgroundProcessor()


# Task handler decorators for easy registration
def task_handler(task_type: str):
    """
    Decorator to register a function as a task handler
    
    Args:
        task_type: Type of task this handler processes
    """
    def decorator(func: Callable):
        background_processor.register_task_handler(task_type, func)
        return func
    return decorator