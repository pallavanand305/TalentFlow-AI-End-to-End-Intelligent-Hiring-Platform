"""Background job repository for database operations"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, delete, func
from sqlalchemy.orm import selectinload
import uuid

from backend.app.models.background_job import BackgroundJob, BackgroundJobStatus
from backend.app.core.database import AsyncSessionLocal


class BackgroundJobRepository:
    """Repository for background job database operations"""
    
    def __init__(self, db_session: AsyncSession = None):
        self.db_session = db_session
    
    async def create_job(
        self,
        job_type: str,
        input_data: Dict[str, Any],
        job_id: str = None
    ) -> BackgroundJob:
        """
        Create a new background job record
        
        Args:
            job_type: Type of background job
            input_data: Input data for the job
            job_id: Optional job ID (will generate if not provided)
            
        Returns:
            Created BackgroundJob instance
        """
        async with AsyncSessionLocal() as session:
            job = BackgroundJob(
                id=uuid.UUID(job_id) if job_id else uuid.uuid4(),
                job_type=job_type,
                status=BackgroundJobStatus.QUEUED,
                input_data=input_data
            )
            
            session.add(job)
            await session.commit()
            await session.refresh(job)
            
            return job
    
    async def get_job_by_id(self, job_id: str) -> Optional[BackgroundJob]:
        """
        Get background job by ID
        
        Args:
            job_id: Job ID
            
        Returns:
            BackgroundJob instance or None if not found
        """
        async with AsyncSessionLocal() as session:
            stmt = select(BackgroundJob).where(BackgroundJob.id == uuid.UUID(job_id))
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def update_job_status(
        self,
        job_id: str,
        status: BackgroundJobStatus,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Optional[BackgroundJob]:
        """
        Update job status and related fields
        
        Args:
            job_id: Job ID
            status: New status
            result_data: Result data (for completed jobs)
            error_message: Error message (for failed jobs)
            
        Returns:
            Updated BackgroundJob instance or None if not found
        """
        async with AsyncSessionLocal() as session:
            # Prepare update data
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if status == BackgroundJobStatus.PROCESSING:
                update_data["started_at"] = datetime.utcnow()
            elif status in [BackgroundJobStatus.COMPLETED, BackgroundJobStatus.FAILED]:
                update_data["completed_at"] = datetime.utcnow()
            
            if result_data is not None:
                update_data["result_data"] = result_data
            
            if error_message is not None:
                update_data["error_message"] = error_message
            
            # Update job
            stmt = (
                update(BackgroundJob)
                .where(BackgroundJob.id == uuid.UUID(job_id))
                .values(**update_data)
                .returning(BackgroundJob)
            )
            
            result = await session.execute(stmt)
            await session.commit()
            
            return result.scalar_one_or_none()
    
    async def get_jobs_by_status(
        self,
        status: BackgroundJobStatus,
        limit: int = 100
    ) -> List[BackgroundJob]:
        """
        Get jobs by status
        
        Args:
            status: Job status to filter by
            limit: Maximum number of jobs to return
            
        Returns:
            List of BackgroundJob instances
        """
        async with AsyncSessionLocal() as session:
            stmt = (
                select(BackgroundJob)
                .where(BackgroundJob.status == status)
                .order_by(BackgroundJob.created_at.desc())
                .limit(limit)
            )
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_jobs_by_type(
        self,
        job_type: str,
        limit: int = 100
    ) -> List[BackgroundJob]:
        """
        Get jobs by type
        
        Args:
            job_type: Job type to filter by
            limit: Maximum number of jobs to return
            
        Returns:
            List of BackgroundJob instances
        """
        async with AsyncSessionLocal() as session:
            stmt = (
                select(BackgroundJob)
                .where(BackgroundJob.job_type == job_type)
                .order_by(BackgroundJob.created_at.desc())
                .limit(limit)
            )
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_recent_jobs(
        self,
        limit: int = 50,
        job_type: Optional[str] = None
    ) -> List[BackgroundJob]:
        """
        Get recent jobs
        
        Args:
            limit: Maximum number of jobs to return
            job_type: Optional job type filter
            
        Returns:
            List of BackgroundJob instances
        """
        async with AsyncSessionLocal() as session:
            stmt = select(BackgroundJob).order_by(BackgroundJob.created_at.desc())
            
            if job_type:
                stmt = stmt.where(BackgroundJob.job_type == job_type)
            
            stmt = stmt.limit(limit)
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def cleanup_old_jobs(self, days_old: int = 30) -> int:
        """
        Clean up old completed/failed jobs
        
        Args:
            days_old: Delete jobs older than this many days
            
        Returns:
            Number of jobs deleted
        """
        async with AsyncSessionLocal() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Delete old completed or failed jobs
            stmt = (
                select(BackgroundJob.id)
                .where(
                    and_(
                        BackgroundJob.status.in_([
                            BackgroundJobStatus.COMPLETED,
                            BackgroundJobStatus.FAILED
                        ]),
                        BackgroundJob.created_at < cutoff_date
                    )
                )
            )
            
            result = await session.execute(stmt)
            job_ids = result.scalars().all()
            
            if job_ids:
                delete_stmt = (
                    delete(BackgroundJob)
                    .where(BackgroundJob.id.in_(job_ids))
                )
                await session.execute(delete_stmt)
                await session.commit()
            
            return len(job_ids)
    
    async def get_job_statistics(self) -> Dict[str, int]:
        """
        Get job statistics by status
        
        Returns:
            Dictionary with job counts by status
        """
        async with AsyncSessionLocal() as session:
            stats = {}
            
            for status in BackgroundJobStatus:
                stmt = (
                    select(func.count(BackgroundJob.id))
                    .where(BackgroundJob.status == status)
                )
                result = await session.execute(stmt)
                stats[status.value] = result.scalar() or 0
            
            return stats