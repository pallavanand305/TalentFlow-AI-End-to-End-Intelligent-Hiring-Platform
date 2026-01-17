"""Entity extraction from resume sections"""

import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """Extract structured entities from resume sections"""
    
    # Common date patterns
    DATE_PATTERNS = [
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b',
        r'\b\d{1,2}/\d{4}\b',
        r'\b\d{4}\b',
        r'\b(present|current|now)\b',
    ]
    
    # Degree patterns
    DEGREE_PATTERNS = [
        r'\b(ph\.?d\.?|doctorate|doctoral)\b',
        r'\b(master|m\.?s\.?|m\.?a\.?|mba|m\.?eng\.?)\b',
        r'\b(bachelor|b\.?s\.?|b\.?a\.?|b\.?eng\.?|b\.?tech\.?)\b',
        r'\b(associate|a\.?s\.?|a\.?a\.?)\b',
        r'\b(diploma|certificate)\b',
    ]
    
    def __init__(self):
        """Initialize entity extractor"""
        self.nlp = None
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy NER model"""
        try:
            import spacy
            # Try to load the model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Entity extraction will use rule-based approach only.")
                self.nlp = None
        except ImportError:
            logger.warning("spaCy not installed. Entity extraction will use rule-based approach only.")
            self.nlp = None
    
    def extract_work_experience(self, experience_text: str) -> List[Dict[str, Any]]:
        """
        Extract work experience entities
        
        Args:
            experience_text: Text from experience section
        
        Returns:
            List of work experience entries with company, title, dates, confidence
        """
        experiences = []
        
        # Split into individual job entries (separated by blank lines or date patterns)
        entries = self._split_experience_entries(experience_text)
        
        for entry in entries:
            experience_data = {
                'company': None,
                'title': None,
                'start_date': None,
                'end_date': None,
                'description': entry,
                'confidence': 0.0
            }
            
            # Extract dates
            dates = self._extract_dates(entry)
            if len(dates) >= 2:
                experience_data['start_date'] = dates[0]
                experience_data['end_date'] = dates[1]
                experience_data['confidence'] += 0.3
            elif len(dates) == 1:
                experience_data['start_date'] = dates[0]
                experience_data['confidence'] += 0.15
            
            # Extract company and title using NER if available
            if self.nlp:
                doc = self.nlp(entry[:500])  # Limit to first 500 chars for performance
                
                # Extract organizations (likely companies)
                orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
                if orgs:
                    experience_data['company'] = orgs[0]
                    experience_data['confidence'] += 0.3
                
                # Extract job title (first line often contains title)
                lines = entry.split('\n')
                if lines:
                    first_line = lines[0].strip()
                    # If first line doesn't look like a company, it's likely a title
                    if first_line and not any(org.lower() in first_line.lower() for org in orgs):
                        experience_data['title'] = first_line
                        experience_data['confidence'] += 0.2
            else:
                # Rule-based extraction
                lines = entry.split('\n')
                if len(lines) >= 2:
                    experience_data['title'] = lines[0].strip()
                    experience_data['company'] = lines[1].strip()
                    experience_data['confidence'] += 0.4
            
            # Minimum confidence threshold
            if experience_data['confidence'] >= 0.3:
                experiences.append(experience_data)
        
        logger.info(f"Extracted {len(experiences)} work experience entries")
        return experiences
    
    def extract_education(self, education_text: str) -> List[Dict[str, Any]]:
        """
        Extract education entities
        
        Args:
            education_text: Text from education section
        
        Returns:
            List of education entries with institution, degree, dates, confidence
        """
        educations = []
        
        # Split into individual education entries
        entries = self._split_education_entries(education_text)
        
        for entry in entries:
            education_data = {
                'institution': None,
                'degree': None,
                'field_of_study': None,
                'start_date': None,
                'end_date': None,
                'description': entry,
                'confidence': 0.0
            }
            
            # Extract dates
            dates = self._extract_dates(entry)
            if len(dates) >= 2:
                education_data['start_date'] = dates[0]
                education_data['end_date'] = dates[1]
                education_data['confidence'] += 0.25
            elif len(dates) == 1:
                education_data['end_date'] = dates[0]
                education_data['confidence'] += 0.15
            
            # Extract degree
            degree = self._extract_degree(entry)
            if degree:
                education_data['degree'] = degree
                education_data['confidence'] += 0.3
            
            # Extract institution using NER if available
            if self.nlp:
                doc = self.nlp(entry[:500])
                orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
                if orgs:
                    education_data['institution'] = orgs[0]
                    education_data['confidence'] += 0.3
            else:
                # Rule-based: institution is often on first or second line
                lines = entry.split('\n')
                for line in lines[:3]:
                    if line.strip() and not self._extract_degree(line):
                        education_data['institution'] = line.strip()
                        education_data['confidence'] += 0.2
                        break
            
            # Minimum confidence threshold
            if education_data['confidence'] >= 0.3:
                educations.append(education_data)
        
        logger.info(f"Extracted {len(educations)} education entries")
        return educations
    
    def extract_skills(self, skills_text: str) -> List[Dict[str, Any]]:
        """
        Extract skills
        
        Args:
            skills_text: Text from skills section
        
        Returns:
            List of skills with confidence scores
        """
        skills = []
        
        # Common skill separators
        separators = [',', '•', '·', '|', ';', '\n']
        
        # Split by separators
        skill_candidates = [skills_text]
        for sep in separators:
            new_candidates = []
            for candidate in skill_candidates:
                new_candidates.extend(candidate.split(sep))
            skill_candidates = new_candidates
        
        # Clean and filter skills
        for skill in skill_candidates:
            skill_clean = skill.strip()
            
            # Filter out empty, too long, or too short entries
            if not skill_clean or len(skill_clean) < 2 or len(skill_clean) > 50:
                continue
            
            # Filter out entries that look like sentences
            if skill_clean.count(' ') > 4:
                continue
            
            skills.append({
                'skill': skill_clean,
                'confidence': 0.7  # Base confidence for skills
            })
        
        logger.info(f"Extracted {len(skills)} skills")
        return skills
    
    def extract_certifications(self, certifications_text: str) -> List[Dict[str, Any]]:
        """
        Extract certifications
        
        Args:
            certifications_text: Text from certifications section
        
        Returns:
            List of certifications with dates and confidence
        """
        certifications = []
        
        # Split into individual certification entries
        lines = certifications_text.split('\n')
        
        for line in lines:
            line_clean = line.strip()
            
            if not line_clean or len(line_clean) < 5:
                continue
            
            cert_data = {
                'certification': line_clean,
                'date': None,
                'confidence': 0.6
            }
            
            # Extract date if present
            dates = self._extract_dates(line_clean)
            if dates:
                cert_data['date'] = dates[0]
                cert_data['confidence'] += 0.2
                # Remove date from certification name
                for date in dates:
                    cert_data['certification'] = cert_data['certification'].replace(date, '').strip()
            
            certifications.append(cert_data)
        
        logger.info(f"Extracted {len(certifications)} certifications")
        return certifications
    
    def _split_experience_entries(self, text: str) -> List[str]:
        """Split experience text into individual job entries"""
        # Split by double newlines or date patterns followed by newlines
        entries = []
        current_entry = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if not line.strip():
                if current_entry:
                    entries.append('\n'.join(current_entry))
                    current_entry = []
            else:
                # Check if this line starts a new entry (has dates)
                if i > 0 and self._extract_dates(line) and current_entry:
                    entries.append('\n'.join(current_entry))
                    current_entry = [line]
                else:
                    current_entry.append(line)
        
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        return [e for e in entries if e.strip()]
    
    def _split_education_entries(self, text: str) -> List[str]:
        """Split education text into individual entries"""
        # Similar to experience splitting
        entries = []
        current_entry = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if not line.strip():
                if current_entry:
                    entries.append('\n'.join(current_entry))
                    current_entry = []
            else:
                # Check if this line starts a new entry (has degree or dates)
                if i > 0 and (self._extract_degree(line) or self._extract_dates(line)) and current_entry:
                    entries.append('\n'.join(current_entry))
                    current_entry = [line]
                else:
                    current_entry.append(line)
        
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        return [e for e in entries if e.strip()]
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        dates = []
        
        for pattern in self.DATE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0)
                if date_str.lower() not in ['present', 'current', 'now']:
                    dates.append(date_str)
                else:
                    dates.append('Present')
        
        return dates
    
    def _extract_degree(self, text: str) -> Optional[str]:
        """Extract degree from text"""
        for pattern in self.DEGREE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract surrounding context for full degree name
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                
                # Clean up the degree string
                degree_parts = context.split()
                if len(degree_parts) <= 6:
                    return context
                else:
                    return match.group(0)
        
        return None
