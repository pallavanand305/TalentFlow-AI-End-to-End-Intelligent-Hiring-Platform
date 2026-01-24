"""Resume schemas"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class WorkExperience(BaseModel):
    """Work experience entry"""
    company: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class Education(BaseModel):
    """Education entry"""
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class Skill(BaseModel):
    """Skill entry"""
    skill: str
    confidence: float = Field(ge=0.0, le=1.0)


class Certification(BaseModel):
    """Certification entry"""
    certification: str
    date: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class ParsedResume(BaseModel):
    """Parsed resume data"""
    
    # Raw text and sections
    raw_text: str
    sections: Dict[str, str] = Field(default_factory=dict)
    
    # Extracted entities
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    
    # Low confidence fields
    low_confidence_fields: List[str] = Field(default_factory=list)
    
    # Metadata
    parsing_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_format: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "raw_text": "John Doe\nSoftware Engineer...",
                "sections": {
                    "experience": "Software Engineer at TechCorp...",
                    "education": "BS Computer Science...",
                    "skills": "Python, Java, SQL"
                },
                "work_experience": [
                    {
                        "company": "TechCorp",
                        "title": "Software Engineer",
                        "start_date": "Jan 2020",
                        "end_date": "Present",
                        "description": "Developed web applications...",
                        "confidence": 0.85
                    }
                ],
                "education": [
                    {
                        "institution": "State University",
                        "degree": "Bachelor of Science",
                        "field_of_study": "Computer Science",
                        "start_date": "2016",
                        "end_date": "2020",
                        "description": "BS Computer Science...",
                        "confidence": 0.90
                    }
                ],
                "skills": [
                    {"skill": "Python", "confidence": 0.7},
                    {"skill": "Java", "confidence": 0.7}
                ],
                "certifications": [
                    {
                        "certification": "AWS Certified Developer",
                        "date": "2021",
                        "confidence": 0.8
                    }
                ],
                "low_confidence_fields": ["work_experience[0].start_date"],
                "file_format": "pdf"
            }
        }
    )


class ResumeUploadResponse(BaseModel):
    """Response for resume upload"""
    job_id: str
    message: str
    status: str = "processing"


class ResumeDetailResponse(BaseModel):
    """Response for resume details"""
    candidate_id: str
    parsed_resume: ParsedResume
    created_at: datetime
    updated_at: datetime
