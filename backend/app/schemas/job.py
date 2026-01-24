"""Job schemas for API requests and responses"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from uuid import UUID

from backend.app.models.job import ExperienceLevel, JobStatus


class JobCreateRequest(BaseModel):
    """Request schema for creating a job"""
    title: str = Field(..., min_length=3, max_length=255, description="Job title")
    description: str = Field(..., min_length=10, description="Job description")
    required_skills: List[str] = Field(..., min_length=1, max_length=50, description="Required skills")
    experience_level: ExperienceLevel = Field(..., description="Required experience level")
    location: Optional[str] = Field(None, max_length=255, description="Job location")
    salary_min: Optional[float] = Field(None, ge=0, description="Minimum salary")
    salary_max: Optional[float] = Field(None, ge=0, description="Maximum salary")
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('Job title cannot be empty')
        return v.strip()
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError('Job description cannot be empty')
        return v.strip()
    
    @field_validator('required_skills')
    @classmethod
    def validate_skills(cls, v):
        if not v:
            raise ValueError('At least one required skill must be specified')
        
        clean_skills = [skill.strip() for skill in v if skill.strip()]
        if not clean_skills:
            raise ValueError('At least one valid required skill must be specified')
        
        return clean_skills
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return v
    
    @field_validator('salary_max')
    @classmethod
    def validate_salary_range(cls, v, info):
        if v is not None and 'salary_min' in info.data and info.data['salary_min'] is not None:
            if v < info.data['salary_min']:
                raise ValueError('Maximum salary cannot be less than minimum salary')
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Senior Python Developer",
                "description": "We are looking for an experienced Python developer to join our team...",
                "required_skills": ["Python", "Django", "PostgreSQL", "Docker"],
                "experience_level": "senior",
                "location": "San Francisco, CA",
                "salary_min": 120000,
                "salary_max": 180000
            }
        }
    )


class JobUpdateRequest(BaseModel):
    """Request schema for updating a job"""
    title: Optional[str] = Field(None, min_length=3, max_length=255, description="Job title")
    description: Optional[str] = Field(None, min_length=10, description="Job description")
    required_skills: Optional[List[str]] = Field(None, min_length=1, max_length=50, description="Required skills")
    experience_level: Optional[ExperienceLevel] = Field(None, description="Required experience level")
    location: Optional[str] = Field(None, max_length=255, description="Job location")
    salary_min: Optional[float] = Field(None, ge=0, description="Minimum salary")
    salary_max: Optional[float] = Field(None, ge=0, description="Maximum salary")
    status: Optional[JobStatus] = Field(None, description="Job status")
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('Job title cannot be empty')
            return v.strip()
        return v
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('Job description cannot be empty')
            return v.strip()
        return v
    
    @field_validator('required_skills')
    @classmethod
    def validate_skills(cls, v):
        if v is not None:
            if not v:
                raise ValueError('At least one required skill must be specified')
            
            clean_skills = [skill.strip() for skill in v if skill.strip()]
            if not clean_skills:
                raise ValueError('At least one valid required skill must be specified')
            
            return clean_skills
        return v
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Senior Python Developer",
                "description": "Updated job description...",
                "required_skills": ["Python", "Django", "PostgreSQL", "Docker", "Kubernetes"],
                "salary_max": 200000
            }
        }
    )


class JobResponse(BaseModel):
    """Response schema for job details"""
    id: UUID
    title: str
    description: str
    required_skills: List[str]
    experience_level: ExperienceLevel
    location: Optional[str]
    salary_min: Optional[float]
    salary_max: Optional[float]
    status: JobStatus
    created_by: UUID
    created_at: datetime
    updated_at: datetime
    
    @classmethod
    def from_job(cls, job):
        """Create JobResponse from Job model"""
        return cls(
            id=job.id,
            title=job.title,
            description=job.description,
            required_skills=job.required_skills,
            experience_level=job.experience_level,
            location=job.location,
            salary_min=job.salary_min,
            salary_max=job.salary_max,
            status=job.status,
            created_by=job.created_by,
            created_at=job.created_at,
            updated_at=job.updated_at
        )
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Senior Python Developer",
                "description": "We are looking for an experienced Python developer...",
                "required_skills": ["Python", "Django", "PostgreSQL", "Docker"],
                "experience_level": "senior",
                "location": "San Francisco, CA",
                "salary_min": 120000,
                "salary_max": 180000,
                "status": "active",
                "created_by": "456e7890-e89b-12d3-a456-426614174001",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }
    )


class JobListResponse(BaseModel):
    """Response schema for job listing with pagination"""
    jobs: List[JobResponse]
    total: int
    skip: int
    limit: int
    has_more: bool
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "jobs": [],
                "total": 150,
                "skip": 0,
                "limit": 100,
                "has_more": True
            }
        }
    )


class JobSearchRequest(BaseModel):
    """Request schema for job search"""
    query: Optional[str] = Field(None, description="Text search query")
    skills: Optional[List[str]] = Field(None, description="Required skills")
    experience_level: Optional[ExperienceLevel] = Field(None, description="Experience level")
    location: Optional[str] = Field(None, description="Job location")
    salary_min: Optional[float] = Field(None, ge=0, description="Minimum salary")
    salary_max: Optional[float] = Field(None, ge=0, description="Maximum salary")
    status: Optional[JobStatus] = Field(JobStatus.ACTIVE, description="Job status")
    skip: int = Field(0, ge=0, description="Number of records to skip")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of records to return")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "python developer",
                "skills": ["Python", "Django"],
                "experience_level": "senior",
                "location": "San Francisco",
                "salary_min": 100000,
                "status": "active",
                "skip": 0,
                "limit": 20
            }
        }
    )


class JobHistoryResponse(BaseModel):
    """Response schema for job history"""
    id: UUID
    job_id: UUID
    title: Optional[str]
    description: Optional[str]
    required_skills: Optional[List[str]]
    changed_by: Optional[UUID]
    changed_at: datetime
    
    @classmethod
    def from_history(cls, history):
        """Create JobHistoryResponse from JobHistory model"""
        return cls(
            id=history.id,
            job_id=history.job_id,
            title=history.title,
            description=history.description,
            required_skills=history.required_skills,
            changed_by=history.changed_by,
            changed_at=history.changed_at
        )
    
    model_config = ConfigDict(from_attributes=True)


class JobStatsResponse(BaseModel):
    """Response schema for job statistics"""
    total_jobs: int
    active_jobs: int
    inactive_jobs: int
    closed_jobs: int
    jobs_by_experience_level: Dict[str, int]
    recent_jobs: List[JobListResponse]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_jobs": 150,
                "active_jobs": 120,
                "inactive_jobs": 20,
                "closed_jobs": 10,
                "jobs_by_experience_level": {
                    "entry": 30,
                    "mid": 60,
                    "senior": 45,
                    "lead": 15
                },
                "recent_jobs": []
            }
        }
    )