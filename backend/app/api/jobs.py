"""Job management API endpoints"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user, require_hiring_manager, require_any_role
from backend.app.models.user import User
from backend.app.models.job import Job, JobStatus, ExperienceLevel
from backend.app.repositories.job_repository import JobRepository
from backend.app.repositories.score_repository import ScoreRepository
from backend.app.repositories.candidate_repository import CandidateRepository
from backend.app.services.job_service import JobService
from backend.app.services.scoring_service import ScoringService
from backend.app.schemas.job import (
    JobCreateRequest, JobUpdateRequest, JobResponse, JobListResponse,
    JobHistoryResponse
)
from backend.app.schemas.score import CandidateRankingResponse, ScoringResponse
from backend.app.core.exceptions import ValidationException, NotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


async def get_job_service(db: AsyncSession = Depends(get_db)) -> JobService:
    """Dependency to get job service"""
    job_repo = JobRepository(db)
    return JobService(job_repo)


async def get_scoring_service(db: AsyncSession = Depends(get_db)) -> ScoringService:
    """Dependency to get scoring service"""
    score_repo = ScoreRepository(db)
    candidate_repo = CandidateRepository(db)
    job_repo = JobRepository(db)
    return ScoringService(score_repo, candidate_repo, job_repo)


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    job_data: JobCreateRequest,
    current_user: User = Depends(require_hiring_manager),
    job_service: JobService = Depends(get_job_service)
):
    """
    Create a new job posting
    
    **Requirements:**
    - User must have hiring_manager or admin role
    - All required fields must be provided
    - Skills list cannot be empty
    - Salary range must be valid (min <= max)
    
    **Returns:**
    - Complete job details with unique ID
    - Creation timestamp and metadata
    - Creator information
    
    **Validation:**
    - Title: 3-255 characters, required
    - Description: minimum 10 characters, required
    - Required skills: at least one skill, maximum 50 skills
    - Experience level: must be valid enum value
    - Salary range: min cannot exceed max, both must be non-negative
    """
    logger.info(f"Job creation request from user {current_user.id}: {job_data.title}")
    
    try:
        job = await job_service.create_job(
            title=job_data.title,
            description=job_data.description,
            required_skills=job_data.required_skills,
            experience_level=job_data.experience_level,
            created_by=current_user.id,
            location=job_data.location,
            salary_min=job_data.salary_min,
            salary_max=job_data.salary_max
        )
        
        return JobResponse.from_job(job)
        
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Job creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create job"
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_details(
    job_id: UUID,
    current_user: User = Depends(require_any_role),
    job_service: JobService = Depends(get_job_service)
):
    """
    Get job details by ID
    
    **Returns:**
    - Complete job information
    - Creator details
    - Creation and update timestamps
    - Current status
    
    **Access Control:**
    - Any authenticated user can view job details
    - Inactive jobs are visible to creators and admins only
    """
    logger.info(f"Job details request from user {current_user.id} for job {job_id}")
    
    try:
        job = await job_service.get_job(job_id)
        
        # Check access permissions for inactive jobs
        if (job.status == JobStatus.INACTIVE and 
            job.created_by != current_user.id and 
            current_user.role.value != "admin"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        return JobResponse.from_job(job)
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Failed to get job details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job details"
        )


@router.put("/{job_id}", response_model=JobResponse)
async def update_job(
    job_id: UUID,
    job_updates: JobUpdateRequest,
    current_user: User = Depends(require_hiring_manager),
    job_service: JobService = Depends(get_job_service)
):
    """
    Update job details with history tracking
    
    **Requirements:**
    - User must have hiring_manager or admin role
    - User must be the job creator or admin
    - Only provided fields will be updated
    - History record is created for all changes
    
    **Returns:**
    - Updated job details
    - New update timestamp
    
    **History Tracking:**
    - Previous values are stored in job_history table
    - Change timestamp and user are recorded
    - Complete audit trail is maintained
    
    **Validation:**
    - Same validation rules as job creation
    - Salary range consistency is enforced
    - Status transitions are validated
    """
    logger.info(f"Job update request from user {current_user.id} for job {job_id}")
    
    try:
        # Check if job exists and user has permission
        existing_job = await job_service.get_job(job_id)
        
        if (existing_job.created_by != current_user.id and 
            current_user.role.value != "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only update jobs you created"
            )
        
        # Prepare update data (only include non-None fields)
        update_data = {}
        if job_updates.title is not None:
            update_data['title'] = job_updates.title
        if job_updates.description is not None:
            update_data['description'] = job_updates.description
        if job_updates.required_skills is not None:
            update_data['required_skills'] = job_updates.required_skills
        if job_updates.experience_level is not None:
            update_data['experience_level'] = job_updates.experience_level
        if job_updates.location is not None:
            update_data['location'] = job_updates.location
        if job_updates.salary_min is not None:
            update_data['salary_min'] = job_updates.salary_min
        if job_updates.salary_max is not None:
            update_data['salary_max'] = job_updates.salary_max
        if job_updates.status is not None:
            update_data['status'] = job_updates.status
        
        if not update_data:
            # No updates provided, return current job
            return JobResponse.from_job(existing_job)
        
        updated_job = await job_service.update_job(
            job_id=job_id,
            updated_by=current_user.id,
            **update_data
        )
        
        return JobResponse.from_job(updated_job)
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Job update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update job"
        )


@router.delete("/{job_id}")
async def delete_job(
    job_id: UUID,
    current_user: User = Depends(require_hiring_manager),
    job_service: JobService = Depends(get_job_service)
):
    """
    Soft delete a job (mark as inactive)
    
    **Requirements:**
    - User must have hiring_manager or admin role
    - User must be the job creator or admin
    
    **Behavior:**
    - Job is marked as inactive, not physically deleted
    - Job remains in database for audit purposes
    - Associated data (scores, history) is preserved
    - Job will not appear in active job searches
    
    **Returns:**
    - Success confirmation message
    - Deletion timestamp
    """
    logger.info(f"Job deletion request from user {current_user.id} for job {job_id}")
    
    try:
        # Check if job exists and user has permission
        existing_job = await job_service.get_job(job_id)
        
        if (existing_job.created_by != current_user.id and 
            current_user.role.value != "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete jobs you created"
            )
        
        success = await job_service.delete_job(job_id)
        
        if success:
            return {
                "success": True,
                "message": "Job deleted successfully",
                "job_id": str(job_id)
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete job"
            )
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Job deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete job"
        )


@router.get("", response_model=JobListResponse)
async def list_jobs(
    skip: int = Query(0, description="Number of records to skip", ge=0),
    limit: int = Query(100, description="Maximum number of records to return", ge=1, le=1000),
    query: Optional[str] = Query(None, description="Text search query"),
    skills: Optional[str] = Query(None, description="Comma-separated list of required skills"),
    experience_level: Optional[ExperienceLevel] = Query(None, description="Required experience level"),
    location: Optional[str] = Query(None, description="Job location filter"),
    salary_min: Optional[float] = Query(None, description="Minimum salary filter", ge=0),
    salary_max: Optional[float] = Query(None, description="Maximum salary filter", ge=0),
    status: Optional[JobStatus] = Query(JobStatus.ACTIVE, description="Job status filter"),
    current_user: User = Depends(require_any_role),
    job_service: JobService = Depends(get_job_service)
):
    """
    Search and list jobs with filtering
    
    **Query Parameters:**
    - skip: Pagination offset (default 0)
    - limit: Page size (1-1000, default 100)
    - query: Text search across job title and description
    - skills: Comma-separated skills (e.g., "Python,Java,SQL")
    - experience_level: Filter by required experience level
    - location: Filter by job location (partial match)
    - salary_min: Filter by minimum salary
    - salary_max: Filter by maximum salary
    - status: Filter by job status (default: active)
    
    **Returns:**
    - List of job summaries
    - Pagination metadata (total count, has_more)
    - Search/filter parameters used
    
    **Search Features:**
    - Full-text search across title and description
    - Skill-based filtering with AND logic
    - Salary range filtering
    - Location partial matching
    - Sorted by creation date (newest first)
    
    **Access Control:**
    - Active jobs: visible to all authenticated users
    - Inactive jobs: visible to creators and admins only
    """
    logger.info(f"Job search request from user {current_user.id}")
    
    try:
        # Parse skills parameter
        skills_list = None
        if skills:
            skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
        
        # Restrict inactive job access
        if status == JobStatus.INACTIVE and current_user.role.value != "admin":
            # Non-admin users can only see their own inactive jobs
            # This would require additional filtering in the service
            # For now, we'll just return active jobs for non-admins
            status = JobStatus.ACTIVE
        
        jobs = await job_service.search_jobs(
            query=query,
            skills=skills_list,
            experience_level=experience_level,
            location=location,
            salary_min=salary_min,
            salary_max=salary_max,
            status=status,
            skip=skip,
            limit=limit
        )
        
        # Get total count for pagination
        # In a real implementation, we'd optimize this with a separate count query
        total_count = len(jobs) + skip  # Approximation
        has_more = len(jobs) == limit
        
        job_responses = [JobResponse.from_job(job) for job in jobs]
        
        return JobListResponse(
            jobs=job_responses,
            total=total_count,
            skip=skip,
            limit=limit,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Failed to search jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search jobs"
        )


@router.get("/{job_id}/history", response_model=List[JobHistoryResponse])
async def get_job_history(
    job_id: UUID,
    current_user: User = Depends(require_any_role),
    job_service: JobService = Depends(get_job_service)
):
    """
    Get job update history
    
    **Requirements:**
    - User must be job creator or admin to view history
    
    **Returns:**
    - List of all changes made to the job
    - Previous values for each change
    - Change timestamps and user information
    - Complete audit trail
    
    **History Information:**
    - What fields were changed
    - Previous values before change
    - User who made the change
    - Timestamp of change
    - Ordered by most recent first
    """
    logger.info(f"Job history request from user {current_user.id} for job {job_id}")
    
    try:
        # Check if job exists and user has permission
        job = await job_service.get_job(job_id)
        
        if (job.created_by != current_user.id and 
            current_user.role.value != "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only view history for jobs you created"
            )
        
        history = await job_service.get_job_history(job_id)
        
        return [JobHistoryResponse.from_history(h) for h in history]
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Failed to get job history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job history"
        )


@router.post("/{job_id}/close")
async def close_job(
    job_id: UUID,
    current_user: User = Depends(require_hiring_manager),
    job_service: JobService = Depends(get_job_service)
):
    """
    Close a job posting
    
    **Requirements:**
    - User must be job creator or admin
    - Job must be currently active
    
    **Behavior:**
    - Job status is changed to 'closed'
    - Job will not appear in active searches
    - History record is created
    - Associated data is preserved
    
    **Returns:**
    - Updated job details
    - Closure timestamp
    """
    logger.info(f"Job close request from user {current_user.id} for job {job_id}")
    
    try:
        # Check if job exists and user has permission
        existing_job = await job_service.get_job(job_id)
        
        if (existing_job.created_by != current_user.id and 
            current_user.role.value != "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only close jobs you created"
            )
        
        if existing_job.status != JobStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only active jobs can be closed"
            )
        
        closed_job = await job_service.close_job(job_id, current_user.id)
        
        return {
            "success": True,
            "message": "Job closed successfully",
            "job": JobResponse.from_job(closed_job)
        }
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Job close failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to close job"
        )


@router.post("/{job_id}/reopen")
async def reopen_job(
    job_id: UUID,
    current_user: User = Depends(require_hiring_manager),
    job_service: JobService = Depends(get_job_service)
):
    """
    Reopen a closed job posting
    
    **Requirements:**
    - User must be job creator or admin
    - Job must be currently closed
    
    **Behavior:**
    - Job status is changed to 'active'
    - Job will appear in active searches again
    - History record is created
    
    **Returns:**
    - Updated job details
    - Reopen timestamp
    """
    logger.info(f"Job reopen request from user {current_user.id} for job {job_id}")
    
    try:
        # Check if job exists and user has permission
        existing_job = await job_service.get_job(job_id)
        
        if (existing_job.created_by != current_user.id and 
            current_user.role.value != "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only reopen jobs you created"
            )
        
        if existing_job.status != JobStatus.CLOSED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only closed jobs can be reopened"
            )
        
        reopened_job = await job_service.reopen_job(job_id, current_user.id)
        
        return {
            "success": True,
            "message": "Job reopened successfully",
            "job": JobResponse.from_job(reopened_job)
        }
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Job reopen failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reopen job"
        )


@router.get("/{job_id}/candidates", response_model=List[CandidateRankingResponse])
async def get_ranked_candidates_for_job(
    job_id: UUID,
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum score threshold"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of candidates to return"),
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Get ranked candidates for a specific job
    
    **Returns candidates ordered by score (highest first)**
    
    **Query Parameters:**
    - min_score: Filter candidates with score >= threshold
    - limit: Maximum number of candidates to return
    
    **Use Cases:**
    - Reviewing top candidates for a position
    - Filtering candidates by minimum qualification threshold
    - Generating candidate shortlists
    
    **Scoring Context:**
    - Scores computed using latest ML model version
    - Rankings updated when job description changes
    - Includes candidate metadata for quick assessment
    """
    logger.info(f"Ranked candidates request from user {current_user.id} for job {job_id}")
    
    try:
        rankings = await scoring_service.get_ranked_candidates_for_job(
            job_id=job_id,
            min_score=min_score,
            limit=limit
        )
        
        # Convert to response format
        response = []
        for i, ranking in enumerate(rankings, 1):
            response.append(CandidateRankingResponse(
                candidate_id=ranking['candidate_id'],
                candidate_name=ranking['candidate_name'],
                score=ranking['score'],
                rank=i,
                explanation=ranking.get('explanation'),
                created_at=ranking['created_at'],
                skills=ranking.get('skills', []),
                experience_years=ranking.get('experience_years'),
                education_level=ranking.get('education_level')
            ))
        
        return response
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Failed to get ranked candidates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve ranked candidates"
        )


@router.get("/{job_id}/top-candidates", response_model=List[ScoringResponse])
async def get_top_candidates_for_job(
    job_id: UUID,
    limit: int = Query(10, ge=1, le=50, description="Number of top candidates to return"),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum score threshold"),
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Get top N candidates for a job
    
    **Returns the highest-scoring candidates for quick review**
    
    **Query Parameters:**
    - limit: Number of top candidates (default: 10, max: 50)
    - min_score: Only return candidates above this threshold
    
    **Use Cases:**
    - Quick candidate review for hiring managers
    - Generating interview shortlists
    - Identifying high-potential candidates
    
    **Performance Notes:**
    - Optimized query for fast retrieval
    - Results cached for frequently accessed jobs
    - Automatically refreshed when new candidates are scored
    """
    logger.info(f"Top candidates request from user {current_user.id} for job {job_id}")
    
    try:
        scores = await scoring_service.get_top_candidates_for_job(
            job_id=job_id,
            limit=limit,
            min_score=min_score
        )
        
        return [ScoringResponse.from_score(score) for score in scores]
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Failed to get top candidates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve top candidates"
        )


@router.get("/stats/summary")
async def get_job_stats(
    current_user: User = Depends(require_any_role),
    job_service: JobService = Depends(get_job_service)
):
    """
    Get job statistics summary
    
    **Returns:**
    - Total active jobs count
    - User's created jobs count (if applicable)
    - Recent job creation trends
    
    **Access Control:**
    - All users can see total active jobs
    - Users see their own job creation stats
    - Admins see system-wide statistics
    """
    logger.info(f"Job stats request from user {current_user.id}")
    
    try:
        active_jobs_count = await job_service.get_active_jobs_count()
        user_jobs = await job_service.get_jobs_by_creator(current_user.id, limit=5)
        
        stats = {
            "active_jobs_total": active_jobs_count,
            "user_jobs_count": len(user_jobs),
            "user_recent_jobs": [
                {
                    "id": str(job.id),
                    "title": job.title,
                    "status": job.status.value,
                    "created_at": job.created_at
                }
                for job in user_jobs
            ]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get job stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job statistics"
        )