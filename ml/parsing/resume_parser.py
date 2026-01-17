"""Resume parser orchestration"""

from pathlib import Path
from typing import Dict, List, Any
from backend.app.core.logging import get_logger
from backend.app.core.exceptions import ValidationException
from backend.app.schemas.resume import (
    ParsedResume,
    WorkExperience,
    Education,
    Skill,
    Certification
)
from ml.parsing.text_extractor import TextExtractor
from ml.parsing.section_identifier import SectionIdentifier
from ml.parsing.entity_extractor import EntityExtractor

logger = get_logger(__name__)


class ResumeParser:
    """Orchestrate resume parsing pipeline"""
    
    # Confidence threshold for flagging low confidence fields
    LOW_CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(self):
        """Initialize resume parser"""
        self.text_extractor = TextExtractor()
        self.section_identifier = SectionIdentifier()
        self.entity_extractor = EntityExtractor()
    
    def parse_resume(self, file_path: str) -> ParsedResume:
        """
        Parse resume file through complete pipeline
        
        Args:
            file_path: Path to resume file (PDF or DOCX)
        
        Returns:
            ParsedResume object with all extracted data
        
        Raises:
            ValidationException: If parsing fails
        """
        logger.info(f"Starting resume parsing for: {file_path}")
        
        try:
            # Step 1: Extract text
            raw_text = self.text_extractor.extract_text(file_path)
            logger.info(f"Extracted {len(raw_text)} characters of text")
            
            # Step 2: Identify sections
            sections = self.section_identifier.identify_sections(raw_text)
            logger.info(f"Identified {len(sections)} sections")
            
            # Step 3: Extract entities from sections
            work_experience = []
            education = []
            skills = []
            certifications = []
            
            # Extract work experience
            if 'experience' in sections:
                work_exp_data = self.entity_extractor.extract_work_experience(sections['experience'])
                work_experience = [WorkExperience(**exp) for exp in work_exp_data]
                logger.info(f"Extracted {len(work_experience)} work experience entries")
            
            # Extract education
            if 'education' in sections:
                education_data = self.entity_extractor.extract_education(sections['education'])
                education = [Education(**edu) for edu in education_data]
                logger.info(f"Extracted {len(education)} education entries")
            
            # Extract skills
            if 'skills' in sections:
                skills_data = self.entity_extractor.extract_skills(sections['skills'])
                skills = [Skill(**skill) for skill in skills_data]
                logger.info(f"Extracted {len(skills)} skills")
            
            # Extract certifications
            if 'certifications' in sections:
                cert_data = self.entity_extractor.extract_certifications(sections['certifications'])
                certifications = [Certification(**cert) for cert in cert_data]
                logger.info(f"Extracted {len(certifications)} certifications")
            
            # Step 4: Identify low confidence fields
            low_confidence_fields = self._identify_low_confidence_fields(
                work_experience, education, skills, certifications
            )
            
            # Step 5: Determine file format
            file_format = Path(file_path).suffix.lower().replace('.', '')
            
            # Create ParsedResume object
            parsed_resume = ParsedResume(
                raw_text=raw_text,
                sections=sections,
                work_experience=work_experience,
                education=education,
                skills=skills,
                certifications=certifications,
                low_confidence_fields=low_confidence_fields,
                file_format=file_format
            )
            
            logger.info(f"Successfully parsed resume with {len(low_confidence_fields)} low confidence fields")
            
            return parsed_resume
        
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Resume parsing failed: {str(e)}")
            raise ValidationException(f"Failed to parse resume: {str(e)}")
    
    def _identify_low_confidence_fields(
        self,
        work_experience: List[WorkExperience],
        education: List[Education],
        skills: List[Skill],
        certifications: List[Certification]
    ) -> List[str]:
        """
        Identify fields with confidence below threshold
        
        Args:
            work_experience: List of work experience entries
            education: List of education entries
            skills: List of skills
            certifications: List of certifications
        
        Returns:
            List of field paths with low confidence
        """
        low_confidence = []
        
        # Check work experience
        for i, exp in enumerate(work_experience):
            if exp.confidence < self.LOW_CONFIDENCE_THRESHOLD:
                low_confidence.append(f"work_experience[{i}]")
            
            # Check individual fields
            if exp.company is None:
                low_confidence.append(f"work_experience[{i}].company")
            if exp.title is None:
                low_confidence.append(f"work_experience[{i}].title")
            if exp.start_date is None:
                low_confidence.append(f"work_experience[{i}].start_date")
        
        # Check education
        for i, edu in enumerate(education):
            if edu.confidence < self.LOW_CONFIDENCE_THRESHOLD:
                low_confidence.append(f"education[{i}]")
            
            # Check individual fields
            if edu.institution is None:
                low_confidence.append(f"education[{i}].institution")
            if edu.degree is None:
                low_confidence.append(f"education[{i}].degree")
        
        # Check skills
        for i, skill in enumerate(skills):
            if skill.confidence < self.LOW_CONFIDENCE_THRESHOLD:
                low_confidence.append(f"skills[{i}]")
        
        # Check certifications
        for i, cert in enumerate(certifications):
            if cert.confidence < self.LOW_CONFIDENCE_THRESHOLD:
                low_confidence.append(f"certifications[{i}]")
        
        return low_confidence
    
    def validate_minimum_fields(self, parsed_resume: ParsedResume) -> bool:
        """
        Validate that minimum required fields are present
        
        Args:
            parsed_resume: Parsed resume object
        
        Returns:
            True if minimum fields are present, False otherwise
        """
        # Minimum requirements: at least one of experience or education
        has_experience = len(parsed_resume.work_experience) > 0
        has_education = len(parsed_resume.education) > 0
        
        return has_experience or has_education
