"""Job service for business logic operations"""

from typing import List, Optional, Dict, Any
from uuid import UUID

from backend.app.repositories.job_repository import JobRepository
from backend.app.models.job import Job, JobHistory, JobStatus, ExperienceLevel
from backend.app.core.logging import get_logger
from backend.app.core.exceptions import ValidationException, NotFoundException

logger = get_logger(__name__)


class JobService:
    """Service for job-related business logic"""
    
    def __init__(self, job_repository: JobRepository):
        """
        Initialize job service
        
        Args:
            job_repository: Job repository
        """
        self.job_repo = job_repository
    
    async def create_job(
        self,
        title: str,
        description: str,
        required_skills: List[str],
        experience_level: ExperienceLevel,
        created_by: UUID,
        location: Optional[str] = None,
        salary_min: Optional[float] = None,
        salary_max: Optional[float] = None
    ) -> Job:
        """
        Create a new job with validation
        
        Args:
            title: Job title
            description: Job description
            required_skills: List of required skills
            experience_level: Required experience level
            created_by: User ID creating the job
            location: Job location
            salary_min: Minimum salary
            salary_max: Maximum salary
        
        Returns:
            Created job
        
        Raises:
            ValidationException: If validation fails
        """
        logger.info(f"Creating job: {title}")
        
        # Validate required fields
        self._validate_job_data(
            title=title,
            description=description,
            required_skills=required_skills,
            experience_level=experience_level,
            salary_min=salary_min,
            salary_max=salary_max
        )
        
        # Create job data
        job_data = {
            'title': title.strip(),
            'description': description.strip(),
            'required_skills': [skill.strip() for skill in required_skills if skill.strip()],
            'experience_level': experience_level,
            'created_by': created_by,
            'location': location.strip() if location else None,
            'salary_min': salary_min,
            'salary_max': salary_max,
            'status': JobStatus.ACTIVE
        }
        
        job = await self.job_repo.create(job_data)
        logger.info(f"Successfully created job: {job.id}")
        return job
    
    async def get_job(self, job_id: UUID) -> Job:
        """
        Get job by ID
        
        Args:
            job_id: Job UUID
        
        Returns:
            Job
        
        Raises:
            NotFoundException: If job not found
        """
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        return job
    
    async def update_job(
        self,
        job_id: UUID,
        updated_by: UUID,
        title: Optional[str] = None,
        description: Optional[str] = None,
        required_skills: Optional[List[str]] = None,
        experience_level: Optional[ExperienceLevel] = None,
        location: Optional[str] = None,
        salary_min: Optional[float] = None,
        salary_max: Optional[float] = None,
        status: Optional[JobStatus] = None
    ) -> Job:
        """
        Update job with history preservation
        
        Args:
            job_id: Job UUID
            updated_by: User ID making the update
            title: New job title
            description: New job description
            required_skills: New required skills
            experience_level: New experience level
            location: New location
            salary_min: New minimum salary
            salary_max: New maximum salary
            status: New job status
        
        Returns:
            Updated job
        
        Raises:
            NotFoundException: If job not found
            ValidationException: If validation fails
        """
        logger.info(f"Updating job: {job_id}")
        
        # Check if job exists
        existing_job = await self.job_repo.get_by_id(job_id)
        if not existing_job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        # Build updates dictionary
        updates = {}
        
        if title is not None:
            if not title.strip():
                raise ValidationException("Job title cannot be empty")
            updates['title'] = title.strip()
        
        if description is not None:
            if not description.strip():
                raise ValidationException("Job description cannot be empty")
            updates['description'] = description.strip()
        
        if required_skills is not None:
            if not required_skills or not any(skill.strip() for skill in required_skills):
                raise ValidationException("At least one required skill must be specified")
            updates['required_skills'] = [skill.strip() for skill in required_skills if skill.strip()]
        
        if experience_level is not None:
            updates['experience_level'] = experience_level
        
        if location is not None:
            updates['location'] = location.strip() if location else None
        
        if salary_min is not None:
            if salary_min < 0:
                raise ValidationException("Minimum salary cannot be negative")
            updates['salary_min'] = salary_min
        
        if salary_max is not None:
            if salary_max < 0:
                raise ValidationException("Maximum salary cannot be negative")
            updates['salary_max'] = salary_max
        
        if status is not None:
            updates['status'] = status
        
        # Validate salary range if both are being updated
        final_min = updates.get('salary_min', existing_job.salary_min)
        final_max = updates.get('salary_max', existing_job.salary_max)
        
        if final_min is not None and final_max is not None and final_min > final_max:
            raise ValidationException("Minimum salary cannot be greater than maximum salary")
        
        # Perform update
        updated_job = await self.job_repo.update(job_id, updates, updated_by)
        logger.info(f"Successfully updated job: {job_id}")
        return updated_job
    
    async def delete_job(self, job_id: UUID) -> bool:
        """
        Soft delete job (mark as inactive)
        
        Args:
            job_id: Job UUID
        
        Returns:
            True if job was deleted successfully
        
        Raises:
            NotFoundException: If job not found
        """
        logger.info(f"Deleting job: {job_id}")
        
        # Check if job exists
        existing_job = await self.job_repo.get_by_id(job_id)
        if not existing_job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        success = await self.job_repo.delete(job_id)
        if success:
            logger.info(f"Successfully deleted job: {job_id}")
        
        return success
    
    async def search_jobs(
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
            query: Text search query
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
        # Validate pagination parameters
        if skip < 0:
            skip = 0
        if limit <= 0 or limit > 1000:
            limit = 100
        
        # Clean up skills list
        if skills:
            skills = [skill.strip() for skill in skills if skill.strip()]
            if not skills:
                skills = None
        
        jobs = await self.job_repo.search(
            query=query,
            skills=skills,
            experience_level=experience_level,
            location=location,
            salary_min=salary_min,
            salary_max=salary_max,
            status=status,
            skip=skip,
            limit=limit
        )
        
        return jobs
    
    async def get_job_history(self, job_id: UUID) -> List[JobHistory]:
        """
        Get job update history
        
        Args:
            job_id: Job UUID
        
        Returns:
            List of job history records
        
        Raises:
            NotFoundException: If job not found
        """
        # Check if job exists
        existing_job = await self.job_repo.get_by_id(job_id)
        if not existing_job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        history = await self.job_repo.get_history(job_id)
        return history
    
    async def get_jobs_by_creator(self, creator_id: UUID, limit: int = 10) -> List[Job]:
        """
        Get jobs created by a specific user
        
        Args:
            creator_id: Creator user ID
            limit: Maximum number of jobs to return
        
        Returns:
            List of jobs created by the user
        """
        if limit <= 0 or limit > 100:
            limit = 10
        
        jobs = await self.job_repo.get_jobs_by_creator(creator_id, limit)
        return jobs
    
    async def get_active_jobs_count(self) -> int:
        """
        Get count of active jobs
        
        Returns:
            Number of active jobs
        """
        return await self.job_repo.get_active_jobs_count()
    
    async def get_recommended_jobs_for_skills(self, skills: List[str], limit: int = 10) -> List[Job]:
        """
        Get jobs that match given skills
        
        Args:
            skills: List of skills to match
            limit: Maximum number of jobs to return
        
        Returns:
            List of matching jobs
        """
        if not skills:
            return []
        
        if limit <= 0 or limit > 50:
            limit = 10
        
        # Clean up skills
        clean_skills = [skill.strip() for skill in skills if skill.strip()]
        if not clean_skills:
            return []
        
        jobs = await self.job_repo.get_jobs_by_skills(clean_skills, limit)
        return jobs
    
    def _validate_job_data(
        self,
        title: str,
        description: str,
        required_skills: List[str],
        experience_level: ExperienceLevel,
        salary_min: Optional[float] = None,
        salary_max: Optional[float] = None
    ) -> None:
        """
        Validate job data
        
        Args:
            title: Job title
            description: Job description
            required_skills: List of required skills
            experience_level: Required experience level
            salary_min: Minimum salary
            salary_max: Maximum salary
        
        Raises:
            ValidationException: If validation fails
        """
        # Validate title
        if not title or not title.strip():
            raise ValidationException("Job title is required")
        
        if len(title.strip()) < 3:
            raise ValidationException("Job title must be at least 3 characters long")
        
        if len(title.strip()) > 255:
            raise ValidationException("Job title cannot exceed 255 characters")
        
        # Validate description
        if not description or not description.strip():
            raise ValidationException("Job description is required")
        
        if len(description.strip()) < 10:
            raise ValidationException("Job description must be at least 10 characters long")
        
        # Validate required skills
        if not required_skills:
            raise ValidationException("At least one required skill must be specified")
        
        clean_skills = [skill.strip() for skill in required_skills if skill.strip()]
        if not clean_skills:
            raise ValidationException("At least one valid required skill must be specified")
        
        if len(clean_skills) > 50:
            raise ValidationException("Cannot specify more than 50 required skills")
        
        # Validate experience level
        if not isinstance(experience_level, ExperienceLevel):
            raise ValidationException("Invalid experience level")
        
        # Validate salary range
        if salary_min is not None and salary_min < 0:
            raise ValidationException("Minimum salary cannot be negative")
        
        if salary_max is not None and salary_max < 0:
            raise ValidationException("Maximum salary cannot be negative")
        
        if (salary_min is not None and salary_max is not None and 
            salary_min > salary_max):
            raise ValidationException("Minimum salary cannot be greater than maximum salary")
    
    async def close_job(self, job_id: UUID, closed_by: UUID) -> Job:
        """
        Close a job (mark as closed)
        
        Args:
            job_id: Job UUID
            closed_by: User ID closing the job
        
        Returns:
            Updated job
        
        Raises:
            NotFoundException: If job not found
        """
        logger.info(f"Closing job: {job_id}")
        
        updated_job = await self.update_job(
            job_id=job_id,
            updated_by=closed_by,
            status=JobStatus.CLOSED
        )
        
        logger.info(f"Successfully closed job: {job_id}")
        return updated_job
    
    async def reopen_job(self, job_id: UUID, reopened_by: UUID) -> Job:
        """
        Reopen a closed job (mark as active)
        
        Args:
            job_id: Job UUID
            reopened_by: User ID reopening the job
        
        Returns:
            Updated job
        
        Raises:
            NotFoundException: If job not found
        """
        logger.info(f"Reopening job: {job_id}")
        
        updated_job = await self.update_job(
            job_id=job_id,
            updated_by=reopened_by,
            status=JobStatus.ACTIVE
        )
        
        logger.info(f"Successfully reopened job: {job_id}")
        return updated_job