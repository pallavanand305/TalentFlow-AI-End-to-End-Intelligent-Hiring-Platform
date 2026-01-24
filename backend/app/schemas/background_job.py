"""Background job schemas"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class BackgroundJobStatus(str, Enum):
    """Background job status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundJobResponse(BaseModel):
    """Response schema for background job status"""
    job_id: str = Field(..., description="Unique job identifier")
    job_type: str = Field(..., description="Type of background job")
    status: BackgroundJobStatus = Field(..., description="Current job status")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data for the job")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Result data (for completed jobs)")
    error_message: Optional[str] = Field(None, description="Error message (for failed jobs)")
    created_at: Optional[datetime] = Field(None, description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )


class QueueStatsResponse(BaseModel):
    """Response schema for queue statistics"""
    redis_queue: Dict[str, int] = Field(..., description="Redis queue statistics")
    database_jobs: Dict[str, int] = Field(..., description="Database job statistics")
    workers_running: int = Field(..., description="Number of active workers")
    is_processing: bool = Field(..., description="Whether processing is active")


class CleanupResponse(BaseModel):
    """Response schema for job cleanup operation"""
    deleted_jobs: int = Field(..., description="Number of jobs deleted")
    days_old: int = Field(..., description="Age threshold for deletion")


class TaskEnqueueRequest(BaseModel):
    """Request schema for enqueueing a task"""
    task_type: str = Field(..., description="Type of task to enqueue")
    task_data: Dict[str, Any] = Field(..., description="Task input data")
    priority: int = Field(0, description="Task priority (higher = more priority)")
    delay_seconds: int = Field(0, description="Delay before task becomes available")


class TaskEnqueueResponse(BaseModel):
    """Response schema for task enqueue operation"""
    job_id: str = Field(..., description="Unique job identifier")
    message: str = Field(..., description="Success message")