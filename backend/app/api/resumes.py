"""Resume API endpoints"""

from typing import List, Optional
from uuid import UUID
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user, require_recruiter, require_any_role
from backend.app.models.user import User
from backend.app.models.background_job import BackgroundJob, BackgroundJobStatus
from backend.app.repositories.candidate_repository import CandidateRepository
from backend.app.services.resume_service import ResumeService
from backend.app.services.s3_service import S3Service
from backend.app.schemas.resume import ResumeUploadResponse, ResumeDetailResponse
from backend.app.core.exceptions import ValidationException, NotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


async def get_resume_service(db: AsyncSession = Depends(get_db)) -> ResumeService:
    """Dependency to get resume service"""
    candidate_repo = CandidateRepository(db)
    s3_service = S3Service()
    return ResumeService(candidate_repo, s3_service)


@router.post("/upload", response_model=ResumeUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_resume(
    file: UploadFile = File(...),
    candidate_name: Optional[str] = None,
    candidate_email: Optional[str] = None,
    current_user: User = Depends(require_recruiter),
    resume_service: ResumeService = Depends(get_resume_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and parse a resume file
    
    **Requirements:**
    - File must be PDF or DOCX format
    - File size must be under 10MB
    - User must have recruiter or admin role
    
    **Process:**
    1. Validates file format and size
    2. Creates background job for async processing
    3. Returns job ID for status tracking
    4. Processing includes: text extraction, parsing, S3 upload, database storage
    
    **Returns:**
    - job_id: Use this to track processing status via `/jobs/status/{job_id}`
    - message: Human-readable status message
    - status: Always "processing" for async operations
    """
    logger.info(f"Resume upload request from user {current_user.id}: {file.filename}")
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    # Check file size (10MB limit)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size must be under 10MB"
        )
    
    # Validate file format
    file_ext = file.filename.lower().split('.')[-1]
    if file_ext not in ['pdf', 'docx']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and DOCX files are supported"
        )
    
    try:
        # Create background job record
        job_id = uuid.uuid4()
        background_job = BackgroundJob(
            id=job_id,
            job_type="resume_parsing",
            status=BackgroundJobStatus.QUEUED,
            input_data={
                "filename": file.filename,
                "content_type": file.content_type,
                "candidate_name": candidate_name,
                "candidate_email": candidate_email,
                "uploaded_by": str(current_user.id)
            }
        )
        
        db.add(background_job)
        await db.commit()
        
        # For now, process synchronously (will be made async in task 18)
        # This is a temporary implementation to get the API working
        try:
            background_job.status = BackgroundJobStatus.PROCESSING
            background_job.started_at = background_job.updated_at
            await db.commit()
            
            # Process the resume
            candidate_id, parsed_resume = await resume_service.upload_and_parse_resume(
                file.file,
                file.filename,
                file.content_type or "application/octet-stream",
                candidate_name,
                candidate_email
            )
            
            # Update job as completed
            background_job.status = BackgroundJobStatus.COMPLETED
            background_job.completed_at = background_job.updated_at
            background_job.result_data = {
                "candidate_id": str(candidate_id),
                "parsing_success": True,
                "sections_found": len(parsed_resume.sections),
                "skills_extracted": len(parsed_resume.skills),
                "experience_entries": len(parsed_resume.work_experience),
                "education_entries": len(parsed_resume.education)
            }
            await db.commit()
            
            logger.info(f"Resume processing completed for job {job_id}, candidate {candidate_id}")
            
        except Exception as e:
            # Update job as failed
            background_job.status = BackgroundJobStatus.FAILED
            background_job.error_message = str(e)
            background_job.completed_at = background_job.updated_at
            await db.commit()
            
            logger.error(f"Resume processing failed for job {job_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Resume processing failed: {str(e)}"
            )
        
        return ResumeUploadResponse(
            job_id=str(job_id),
            message="Resume upload successful. Processing in background.",
            status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resume upload failed"
        )


@router.get("/{resume_id}", response_model=ResumeDetailResponse)
async def get_resume_details(
    resume_id: UUID,
    current_user: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db)
):
    """
    Get resume details and parsed data
    
    **Returns:**
    - candidate_id: Unique identifier for the candidate
    - parsed_resume: Structured data extracted from the resume
    - created_at: When the resume was uploaded
    - updated_at: When the resume was last modified
    
    **Parsed Resume includes:**
    - Raw text content
    - Identified sections (experience, education, skills)
    - Structured work experience entries
    - Structured education entries
    - Extracted skills with confidence scores
    - Certifications
    - Low confidence fields flagged for review
    """
    logger.info(f"Resume details request from user {current_user.id} for resume {resume_id}")
    
    try:
        # Get candidate from repository
        candidate_repo = CandidateRepository(db)
        candidate = await candidate_repo.get_by_id(resume_id)
        
        if not candidate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        return ResumeDetailResponse(
            candidate_id=str(candidate.id),
            parsed_resume=candidate.parsed_data,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get resume details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve resume details"
        )


@router.get("/{resume_id}/download")
async def download_resume(
    resume_id: UUID,
    current_user: User = Depends(require_any_role),
    resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Download the original resume file
    
    **Returns:**
    - Original resume file (PDF or DOCX) as binary stream
    - Appropriate content-type header
    - Content-disposition header with original filename
    """
    logger.info(f"Resume download request from user {current_user.id} for resume {resume_id}")
    
    try:
        file_content, filename = await resume_service.get_resume(resume_id)
        
        # Determine content type from filename
        content_type = "application/pdf" if filename.endswith('.pdf') else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        return StreamingResponse(
            iter([file_content]),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to download resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download resume"
        )


@router.get("/{resume_id}/url")
async def get_resume_url(
    resume_id: UUID,
    expiration: int = Query(3600, description="URL expiration in seconds", ge=60, le=86400),
    current_user: User = Depends(require_any_role),
    resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Get a presigned URL for resume download
    
    **Parameters:**
    - expiration: URL expiration time in seconds (60 to 86400, default 3600)
    
    **Returns:**
    - presigned_url: Temporary URL for direct S3 access
    - expires_in: Expiration time in seconds
    
    **Use case:**
    - Frontend applications that need direct file access
    - Sharing resume links with external systems
    - Avoiding server bandwidth for file downloads
    """
    logger.info(f"Resume URL request from user {current_user.id} for resume {resume_id}")
    
    try:
        url = await resume_service.get_resume_url(resume_id, expiration)
        
        return {
            "presigned_url": url,
            "expires_in": expiration
        }
        
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to generate resume URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate resume URL"
        )


@router.get("", response_model=List[dict])
async def list_resumes(
    skip: int = Query(0, description="Number of records to skip", ge=0),
    limit: int = Query(100, description="Maximum number of records to return", ge=1, le=1000),
    query: Optional[str] = Query(None, description="Text search query"),
    skills: Optional[str] = Query(None, description="Comma-separated list of required skills"),
    min_experience_years: Optional[int] = Query(None, description="Minimum years of experience", ge=0),
    education_level: Optional[str] = Query(None, description="Required education level"),
    current_user: User = Depends(require_any_role),
    resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Search and list resumes with filtering
    
    **Query Parameters:**
    - skip: Pagination offset (default 0)
    - limit: Page size (1-1000, default 100)
    - query: Text search across resume content
    - skills: Comma-separated skills (e.g., "Python,Java,SQL")
    - min_experience_years: Filter by minimum experience
    - education_level: Filter by education level
    
    **Returns:**
    - List of candidate summaries with basic info
    - Pagination metadata
    - Total count of matching records
    
    **Search Features:**
    - Full-text search across resume content
    - Skill-based filtering with AND logic
    - Experience and education filters
    - Sorted by relevance or upload date
    """
    logger.info(f"Resume search request from user {current_user.id}")
    
    try:
        # Parse skills parameter
        skills_list = None
        if skills:
            skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
        
        candidates = await resume_service.search_candidates(
            query=query,
            skills=skills_list,
            min_experience_years=min_experience_years,
            education_level=education_level,
            skip=skip,
            limit=limit
        )
        
        # Convert to response format
        results = []
        for candidate in candidates:
            results.append({
                "candidate_id": str(candidate.id),
                "name": candidate.name,
                "email": candidate.email,
                "skills": candidate.skills,
                "experience_years": candidate.experience_years,
                "education_level": candidate.education_level,
                "created_at": candidate.created_at,
                "updated_at": candidate.updated_at
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to search resumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search resumes"
        )


@router.delete("/{resume_id}")
async def delete_resume(
    resume_id: UUID,
    current_user: User = Depends(require_recruiter),
    resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Delete a resume (soft delete)
    
    **Behavior:**
    - Removes candidate record from database
    - Keeps original file in S3 for audit purposes
    - Cannot be undone through API
    
    **Returns:**
    - success: Boolean indicating deletion status
    - message: Confirmation message
    """
    logger.info(f"Resume deletion request from user {current_user.id} for resume {resume_id}")
    
    try:
        success = await resume_service.delete_resume(resume_id)
        
        if success:
            return {
                "success": True,
                "message": "Resume deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete resume"
            )
        
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to delete resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete resume"
        )