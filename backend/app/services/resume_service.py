"""Resume service for orchestrating resume operations"""

from typing import Optional, List, BinaryIO
from uuid import UUID
import tempfile
import os

from backend.app.repositories.candidate_repository import CandidateRepository
from backend.app.services.s3_service import S3Service
from ml.parsing.resume_parser import ResumeParser
from backend.app.schemas.resume import ParsedResume
from backend.app.core.logging import get_logger
from backend.app.core.exceptions import ValidationException, NotFoundException

logger = get_logger(__name__)


class ResumeService:
    """Service for resume upload, parsing, and retrieval"""
    
    def __init__(
        self,
        candidate_repository: CandidateRepository,
        s3_service: S3Service
    ):
        """
        Initialize resume service
        
        Args:
            candidate_repository: Candidate repository
            s3_service: S3 service
        """
        self.candidate_repo = candidate_repository
        self.s3_service = s3_service
        self.resume_parser = ResumeParser()
    
    async def upload_and_parse_resume(
        self,
        file_content: BinaryIO,
        filename: str,
        content_type: str,
        candidate_name: Optional[str] = None,
        candidate_email: Optional[str] = None
    ) -> tuple[UUID, ParsedResume]:
        """
        Upload resume to S3 and parse it
        
        Workflow:
        1. Validate file format
        2. Create temporary file for parsing
        3. Parse resume
        4. Upload to S3
        5. Create candidate record in database
        
        Args:
            file_content: File content as binary stream
            filename: Original filename
            content_type: MIME type
            candidate_name: Optional candidate name (extracted if not provided)
            candidate_email: Optional candidate email
        
        Returns:
            Tuple of (candidate_id, parsed_resume)
        
        Raises:
            ValidationException: If validation or processing fails
        """
        logger.info(f"Starting resume upload and parse for: {filename}")
        
        # Validate file format
        self._validate_file_format(filename)
        
        # Create temporary file for parsing
        temp_file = None
        try:
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(filename)[1]
            )
            temp_file.write(file_content.read())
            temp_file.close()
            
            # Parse resume
            parsed_resume = self.resume_parser.parse_resume(temp_file.name)
            
            # Validate minimum fields
            if not self.resume_parser.validate_minimum_fields(parsed_resume):
                raise ValidationException(
                    "Resume does not contain minimum required fields (experience or education)"
                )
            
            # Extract candidate info from parsed data if not provided
            if not candidate_name and parsed_resume.work_experience:
                # Try to extract name from first line of raw text
                first_line = parsed_resume.raw_text.split('\n')[0].strip()
                if len(first_line) < 50:  # Likely a name
                    candidate_name = first_line
            
            if not candidate_name:
                candidate_name = "Unknown Candidate"
            
            # Create candidate record first to get ID
            candidate_data = {
                'name': candidate_name,
                'email': candidate_email,
                'resume_file_path': '',  # Will update after S3 upload
                'parsed_data': parsed_resume.model_dump(mode='json'),
                'skills': [skill.skill for skill in parsed_resume.skills],
                'experience_years': self._calculate_experience_years(parsed_resume),
                'education_level': self._extract_education_level(parsed_resume)
            }
            
            candidate = await self.candidate_repo.create(candidate_data)
            
            # Reset file pointer for S3 upload
            file_content.seek(0)
            
            # Upload to S3
            s3_key = await self.s3_service.upload_resume(
                file_content,
                str(candidate.id),
                filename,
                content_type
            )
            
            # Update candidate with S3 path
            await self.candidate_repo.update(
                candidate.id,
                {'resume_file_path': s3_key}
            )
            
            logger.info(f"Successfully processed resume for candidate: {candidate.id}")
            
            return candidate.id, parsed_resume
        
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Resume processing failed: {str(e)}")
            raise ValidationException(f"Failed to process resume: {str(e)}")
        
        finally:
            # Cleanup temporary file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    async def get_resume(self, candidate_id: UUID) -> tuple[bytes, str]:
        """
        Retrieve resume file from S3
        
        Args:
            candidate_id: Candidate UUID
        
        Returns:
            Tuple of (file_content, filename)
        
        Raises:
            NotFoundException: If candidate or resume not found
        """
        # Get candidate
        candidate = await self.candidate_repo.get_by_id(candidate_id)
        
        if not candidate:
            raise NotFoundException(f"Candidate not found: {candidate_id}")
        
        if not candidate.resume_file_path:
            raise NotFoundException(f"No resume file for candidate: {candidate_id}")
        
        # Download from S3
        file_content = await self.s3_service.download_resume(candidate.resume_file_path)
        
        # Extract filename from S3 key
        filename = os.path.basename(candidate.resume_file_path)
        
        return file_content, filename
    
    async def get_resume_url(self, candidate_id: UUID, expiration: int = 3600) -> str:
        """
        Get presigned URL for resume download
        
        Args:
            candidate_id: Candidate UUID
            expiration: URL expiration in seconds
        
        Returns:
            Presigned URL
        
        Raises:
            NotFoundException: If candidate or resume not found
        """
        candidate = await self.candidate_repo.get_by_id(candidate_id)
        
        if not candidate:
            raise NotFoundException(f"Candidate not found: {candidate_id}")
        
        if not candidate.resume_file_path:
            raise NotFoundException(f"No resume file for candidate: {candidate_id}")
        
        url = await self.s3_service.generate_presigned_url(
            candidate.resume_file_path,
            expiration
        )
        
        return url
    
    async def delete_resume(self, candidate_id: UUID) -> bool:
        """
        Soft delete resume (mark candidate as inactive)
        
        Args:
            candidate_id: Candidate UUID
        
        Returns:
            True if deleted successfully
        
        Raises:
            NotFoundException: If candidate not found
        """
        candidate = await self.candidate_repo.get_by_id(candidate_id)
        
        if not candidate:
            raise NotFoundException(f"Candidate not found: {candidate_id}")
        
        # Soft delete: just delete from database
        # S3 file can be kept for audit purposes
        success = await self.candidate_repo.delete(candidate_id)
        
        logger.info(f"Soft deleted candidate: {candidate_id}")
        
        return success
    
    async def search_candidates(
        self,
        query: Optional[str] = None,
        skills: Optional[List[str]] = None,
        min_experience_years: Optional[int] = None,
        education_level: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List:
        """
        Search candidates with filters
        
        Args:
            query: Text search query
            skills: Required skills
            min_experience_years: Minimum experience
            education_level: Required education level
            skip: Pagination offset
            limit: Page size
        
        Returns:
            List of matching candidates
        """
        candidates = await self.candidate_repo.search(
            query=query,
            skills=skills,
            min_experience_years=min_experience_years,
            education_level=education_level,
            skip=skip,
            limit=limit
        )
        
        return candidates
    
    def _validate_file_format(self, filename: str) -> None:
        """
        Validate file format
        
        Args:
            filename: Filename to validate
        
        Raises:
            ValidationException: If format is invalid
        """
        ext = os.path.splitext(filename)[1].lower()
        
        if ext not in ['.pdf', '.docx']:
            raise ValidationException(
                f"Unsupported file format: {ext}. Supported formats: .pdf, .docx"
            )
    
    def _calculate_experience_years(self, parsed_resume: ParsedResume) -> Optional[int]:
        """
        Calculate total years of experience from parsed resume
        
        Args:
            parsed_resume: Parsed resume data
        
        Returns:
            Total years of experience or None
        """
        if not parsed_resume.work_experience:
            return None
        
        # Simple heuristic: count number of jobs * 2 years average
        # In production, would parse dates properly
        return len(parsed_resume.work_experience) * 2
    
    def _extract_education_level(self, parsed_resume: ParsedResume) -> Optional[str]:
        """
        Extract highest education level from parsed resume
        
        Args:
            parsed_resume: Parsed resume data
        
        Returns:
            Education level or None
        """
        if not parsed_resume.education:
            return None
        
        # Simple heuristic: look for degree keywords
        education_text = ' '.join([edu.degree or '' for edu in parsed_resume.education]).lower()
        
        if 'phd' in education_text or 'doctorate' in education_text:
            return 'PhD'
        elif 'master' in education_text:
            return 'Master'
        elif 'bachelor' in education_text:
            return 'Bachelor'
        else:
            return 'Other'
