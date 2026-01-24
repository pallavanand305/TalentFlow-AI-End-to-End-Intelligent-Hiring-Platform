"""Background job status API endpoints"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer
import logging

from backend.app.services.background_processor import background_processor
from backend.app.core.security import get_current_user
from backend.app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["background-jobs"])
security = HTTPBearer()


@router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the status of a background job
    
    Args:
        job_id: Unique job identifier
        current_user: Authenticated user
        
    Returns:
        Job status information
        
    Raises:
        HTTPException: If job not found or access denied
    """
    try:
        status = await background_processor.get_task_status(job_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        return {
            "success": True,
            "data": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve job status"
        )


@router.get("/stats")
async def get_queue_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get queue and processing statistics
    
    Args:
        current_user: Authenticated user (admin role required)
        
    Returns:
        Queue statistics
        
    Raises:
        HTTPException: If access denied or error occurs
    """
    # Check if user has admin role
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    try:
        stats = await background_processor.get_queue_stats()
        
        return {
            "success": True,
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve queue statistics"
        )


@router.post("/cleanup")
async def cleanup_old_jobs(
    days_old: int = Query(30, ge=1, le=365, description="Delete jobs older than this many days"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Clean up old completed/failed jobs
    
    Args:
        days_old: Delete jobs older than this many days
        current_user: Authenticated user (admin role required)
        
    Returns:
        Cleanup results
        
    Raises:
        HTTPException: If access denied or error occurs
    """
    # Check if user has admin role
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    try:
        deleted_count = await background_processor.cleanup_old_jobs(days_old)
        
        return {
            "success": True,
            "data": {
                "deleted_jobs": deleted_count,
                "days_old": days_old
            },
            "message": f"Cleaned up {deleted_count} old jobs"
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup old jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to cleanup old jobs"
        )