"""Job repository for database operations"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.orm import selectinload

from backend.app.models.job import Job, JobHistory, JobStatus, ExperienceLevel
from backend.app.models.user import User
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class JobRepository:
    """Repository for job-related database operations"""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize job repository
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def create(self, job_data: Dict[str, Any]) -> Job:
        """
        Create a new job
        
        Args:
            job_data: Job data dictionary
        
        Returns:
            Created job
        """
        job = Job(**job_data)
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        
        logger.info(f"Created job: {job.id}")
        return job
    
    async def get_by_id(self, job_id: UUID) -> Optional[Job]:
        """
        Get job by ID
        
        Args:
            job_id: Job UUID
        
        Returns:
            Job if found, None otherwise
        """
        stmt = select(Job).where(Job.id == job_id).options(
            selectinload(Job.creator),
            selectinload(Job.history)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[JobStatus] = None,
        created_by: Optional[UUID] = None
    ) -> List[Job]:
        """
        Get all jobs with optional filtering
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            status: Filter by job status
            created_by: Filter by creator ID
        
        Returns:
            List of jobs
        """
        stmt = select(Job).options(selectinload(Job.creator))
        
        # Apply filters
        if status:
            stmt = stmt.where(Job.status == status)
        
        if created_by:
            stmt = stmt.where(Job.created_by == created_by)
        
        # Apply pagination
        stmt = stmt.offset(skip).limit(limit).order_by(Job.created_at.desc())
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def search(
        self,
        query: Optional[str] = None,
        skills: Optional[List[str]] = None,
        experience_level: Optional[ExperienceLevel] = None,
        location: Optional[str] = None,
        salary_min: Optional[float] = None,
        salary_max: Optional[float] = None,
        status: Optional[JobStatus] = JobStatus.ACTIVE,
        skip: int = 0,
        limit: int = 100
    ) -> List[Job]:
        """
        Search jobs with various filters
        
        Args:
            query: Text search query (searches title and description)
            skills: Required skills
            experience_level: Required experience level
            location: Job location
            salary_min: Minimum salary
            salary_max: Maximum salary
            status: Job status filter
            skip: Pagination offset
            limit: Page size
        
        Returns:
            List of matching jobs
        """
        stmt = select(Job).options(selectinload(Job.creator))
        
        # Status filter (default to active jobs)
        if status:
            stmt = stmt.where(Job.status == status)
        
        # Text search in title and description
        if query:
            search_term = f"%{query}%"
            stmt = stmt.where(
                or_(
                    Job.title.ilike(search_term),
                    Job.description.ilike(search_term)
                )
            )
        
        # Skills filter (job must have all specified skills)
        if skills:
            for skill in skills:
                stmt = stmt.where(Job.required_skills.any(skill))
        
        # Experience level filter
        if experience_level:
            stmt = stmt.where(Job.experience_level == experience_level)
        
        # Location filter
        if location:
            location_term = f"%{location}%"
            stmt = stmt.where(Job.location.ilike(location_term))
        
        # Salary filters
        if salary_min is not None:
            stmt = stmt.where(
                or_(
                    Job.salary_min >= salary_min,
                    Job.salary_max >= salary_min
                )
            )
        
        if salary_max is not None:
            stmt = stmt.where(
                or_(
                    Job.salary_min <= salary_max,
                    Job.salary_max <= salary_max
                )
            )
        
        # Apply pagination and ordering
        stmt = stmt.offset(skip).limit(limit).order_by(Job.created_at.desc())
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def update(self, job_id: UUID, updates: Dict[str, Any], updated_by: UUID) -> Optional[Job]:
        """
        Update job and create history record
        
        Args:
            job_id: Job UUID
            updates: Fields to update
            updated_by: User ID making the update
        
        Returns:
            Updated job if found, None otherwise
        """
        # Get current job for history
        current_job = await self.get_by_id(job_id)
        if not current_job:
            return None
        
        # Create history record
        history = JobHistory(
            job_id=job_id,
            title=current_job.title,
            description=current_job.description,
            required_skills=current_job.required_skills,
            changed_by=updated_by
        )
        self.db.add(history)
        
        # Update job
        stmt = update(Job).where(Job.id == job_id).values(**updates)
        await self.db.execute(stmt)
        await self.db.commit()
        
        # Return updated job
        updated_job = await self.get_by_id(job_id)
        logger.info(f"Updated job: {job_id}")
        return updated_job
    
    async def delete(self, job_id: UUID) -> bool:
        """
        Soft delete job (mark as inactive)
        
        Args:
            job_id: Job UUID
        
        Returns:
            True if job was found and deleted, False otherwise
        """
        stmt = update(Job).where(Job.id == job_id).values(status=JobStatus.INACTIVE)
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        success = result.rowcount > 0
        if success:
            logger.info(f"Soft deleted job: {job_id}")
        
        return success
    
    async def get_history(self, job_id: UUID) -> List[JobHistory]:
        """
        Get job history
        
        Args:
            job_id: Job UUID
        
        Returns:
            List of job history records
        """
        stmt = select(JobHistory).where(JobHistory.job_id == job_id).options(
            selectinload(JobHistory.changer)
        ).order_by(JobHistory.changed_at.desc())
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def count(
        self,
        status: Optional[JobStatus] = None,
        created_by: Optional[UUID] = None
    ) -> int:
        """
        Count jobs with optional filtering
        
        Args:
            status: Filter by job status
            created_by: Filter by creator ID
        
        Returns:
            Number of matching jobs
        """
        stmt = select(func.count(Job.id))
        
        # Apply filters
        if status:
            stmt = stmt.where(Job.status == status)
        
        if created_by:
            stmt = stmt.where(Job.created_by == created_by)
        
        result = await self.db.execute(stmt)
        return result.scalar()
    
    async def get_jobs_by_creator(self, creator_id: UUID, limit: int = 10) -> List[Job]:
        """
        Get recent jobs created by a specific user
        
        Args:
            creator_id: Creator user ID
            limit: Maximum number of jobs to return
        
        Returns:
            List of jobs created by the user
        """
        stmt = select(Job).where(Job.created_by == creator_id).options(
            selectinload(Job.creator)
        ).order_by(Job.created_at.desc()).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_active_jobs_count(self) -> int:
        """
        Get count of active jobs
        
        Returns:
            Number of active jobs
        """
        return await self.count(status=JobStatus.ACTIVE)
    
    async def get_jobs_by_skills(self, skills: List[str], limit: int = 10) -> List[Job]:
        """
        Get jobs that require specific skills
        
        Args:
            skills: List of skills to match
            limit: Maximum number of jobs to return
        
        Returns:
            List of matching jobs
        """
        stmt = select(Job).where(Job.status == JobStatus.ACTIVE).options(
            selectinload(Job.creator)
        )
        
        # Job must have at least one of the specified skills
        skill_conditions = [Job.required_skills.any(skill) for skill in skills]
        stmt = stmt.where(or_(*skill_conditions))
        
        stmt = stmt.order_by(Job.created_at.desc()).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_jobs_needing_candidates(self, limit: int = 10) -> List[Job]:
        """
        Get active jobs that might need candidates
        (This is a placeholder - in a real system, this might check
        for jobs with few applications or low match scores)
        
        Args:
            limit: Maximum number of jobs to return
        
        Returns:
            List of jobs needing candidates
        """
        stmt = select(Job).where(Job.status == JobStatus.ACTIVE).options(
            selectinload(Job.creator)
        ).order_by(Job.created_at.desc()).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()