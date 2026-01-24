"""Task queue abstraction using Redis"""

import json
import uuid
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta, timezone
import redis.asyncio as redis
from redis.asyncio import Redis
import logging
from backend.app.core.config import settings
from backend.app.models.background_job import BackgroundJobStatus

logger = logging.getLogger(__name__)


class TaskQueue:
    """Redis-based task queue for background job processing"""
    
    def __init__(self, redis_url: str = None):
        """Initialize task queue with Redis connection"""
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis: Optional[Redis] = None
        self.queue_name = "talentflow:tasks"
        self.result_prefix = "talentflow:results"
        self.status_prefix = "talentflow:status"
    
    async def connect(self) -> None:
        """Establish Redis connection"""
        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            await self._redis.ping()
            logger.info("Connected to Redis task queue")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            logger.info("Disconnected from Redis task queue")
    
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
            task_type: Type of task (e.g., 'resume_parsing', 'batch_scoring')
            task_data: Task input data
            priority: Task priority (higher = more priority)
            delay_seconds: Delay before task becomes available
            
        Returns:
            job_id: Unique identifier for the task
        """
        if not self._redis:
            await self.connect()
        
        job_id = str(uuid.uuid4())
        
        # Create task payload
        task_payload = {
            "job_id": job_id,
            "task_type": task_type,
            "task_data": task_data,
            "priority": priority,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scheduled_at": (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()
        }
        
        try:
            # Set initial status
            await self.set_task_status(job_id, BackgroundJobStatus.QUEUED)
            
            if delay_seconds > 0:
                # Schedule task for later execution
                await self._redis.zadd(
                    f"{self.queue_name}:delayed",
                    {json.dumps(task_payload): (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).timestamp()}
                )
            else:
                # Add to immediate processing queue with priority
                await self._redis.zadd(
                    self.queue_name,
                    {json.dumps(task_payload): priority}
                )
            
            logger.info(f"Enqueued task {task_type} with job_id {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task_type}: {e}")
            raise
    
    async def dequeue_task(self, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        Dequeue the highest priority task
        
        Args:
            timeout: Timeout in seconds for blocking pop
            
        Returns:
            Task payload or None if no tasks available
        """
        if not self._redis:
            await self.connect()
        
        try:
            # First, move any ready delayed tasks to the main queue
            await self._move_ready_delayed_tasks()
            
            # Get highest priority task (ZREVRANGE gets highest scores first)
            result = await self._redis.bzpopmax(self.queue_name, timeout=timeout)
            
            if result:
                queue_name, task_json, priority = result
                task_payload = json.loads(task_json)
                
                # Update status to processing
                job_id = task_payload["job_id"]
                await self.set_task_status(job_id, BackgroundJobStatus.PROCESSING)
                
                logger.info(f"Dequeued task {task_payload['task_type']} with job_id {job_id}")
                return task_payload
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            raise
    
    async def _move_ready_delayed_tasks(self) -> None:
        """Move delayed tasks that are ready to the main queue"""
        try:
            current_time = datetime.now(timezone.utc).timestamp()
            
            # Get all delayed tasks that are ready
            ready_tasks = await self._redis.zrangebyscore(
                f"{self.queue_name}:delayed",
                0,
                current_time,
                withscores=True
            )
            
            if ready_tasks:
                # Move ready tasks to main queue
                pipe = self._redis.pipeline()
                for task_json, _ in ready_tasks:
                    task_payload = json.loads(task_json)
                    priority = task_payload.get("priority", 0)
                    
                    # Add to main queue
                    pipe.zadd(self.queue_name, {task_json: priority})
                    # Remove from delayed queue
                    pipe.zrem(f"{self.queue_name}:delayed", task_json)
                
                await pipe.execute()
                logger.info(f"Moved {len(ready_tasks)} delayed tasks to main queue")
                
        except Exception as e:
            logger.error(f"Failed to move delayed tasks: {e}")
    
    async def set_task_status(
        self,
        job_id: str,
        status: BackgroundJobStatus,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update task status in Redis
        
        Args:
            job_id: Task job ID
            status: New status
            result_data: Task result data (for completed tasks)
            error_message: Error message (for failed tasks)
        """
        if not self._redis:
            await self.connect()
        
        try:
            status_data = {
                "status": status.value,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            if result_data:
                status_data["result_data"] = result_data
            
            if error_message:
                status_data["error_message"] = error_message
            
            # Store status with expiration (7 days)
            await self._redis.setex(
                f"{self.status_prefix}:{job_id}",
                timedelta(days=7),
                json.dumps(status_data)
            )
            
            logger.debug(f"Updated task {job_id} status to {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to set task status for {job_id}: {e}")
            raise
    
    async def get_task_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status from Redis
        
        Args:
            job_id: Task job ID
            
        Returns:
            Status data or None if not found
        """
        if not self._redis:
            await self.connect()
        
        try:
            status_json = await self._redis.get(f"{self.status_prefix}:{job_id}")
            
            if status_json:
                return json.loads(status_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task status for {job_id}: {e}")
            raise
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """
        Get queue statistics
        
        Returns:
            Dictionary with queue statistics
        """
        if not self._redis:
            await self.connect()
        
        try:
            stats = {
                "queued_tasks": await self._redis.zcard(self.queue_name),
                "delayed_tasks": await self._redis.zcard(f"{self.queue_name}:delayed"),
                "total_tasks": 0
            }
            
            stats["total_tasks"] = stats["queued_tasks"] + stats["delayed_tasks"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            raise
    
    async def clear_queue(self) -> None:
        """Clear all tasks from the queue (for testing)"""
        if not self._redis:
            await self.connect()
        
        try:
            await self._redis.delete(self.queue_name)
            await self._redis.delete(f"{self.queue_name}:delayed")
            logger.info("Cleared task queue")
            
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
            raise
    
    async def retry_failed_task(self, job_id: str, max_retries: int = 3) -> bool:
        """
        Retry a failed task
        
        Args:
            job_id: Task job ID
            max_retries: Maximum number of retries
            
        Returns:
            True if task was requeued, False otherwise
        """
        if not self._redis:
            await self.connect()
        
        try:
            # Get current retry count
            retry_key = f"talentflow:retries:{job_id}"
            retry_count = await self._redis.get(retry_key)
            retry_count = int(retry_count) if retry_count else 0
            
            if retry_count >= max_retries:
                logger.warning(f"Task {job_id} exceeded max retries ({max_retries})")
                return False
            
            # Increment retry count
            await self._redis.setex(retry_key, timedelta(days=1), retry_count + 1)
            
            # Get original task data (this would need to be stored separately)
            # For now, we'll just update status to queued
            await self.set_task_status(job_id, BackgroundJobStatus.QUEUED)
            
            logger.info(f"Retrying task {job_id} (attempt {retry_count + 1})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retry task {job_id}: {e}")
            raise


# Global task queue instance
task_queue = TaskQueue()