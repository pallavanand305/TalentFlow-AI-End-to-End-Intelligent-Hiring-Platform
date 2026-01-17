"""Resume section identification"""

import re
from typing import Dict, List, Tuple
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class SectionIdentifier:
    """Identify sections in resume text"""
    
    # Section headers patterns (case-insensitive)
    SECTION_PATTERNS = {
        'experience': [
            r'\b(work\s+experience|professional\s+experience|employment\s+history|experience|work\s+history)\b',
            r'\b(career\s+history|professional\s+background)\b',
        ],
        'education': [
            r'\b(education|academic\s+background|qualifications|academic\s+qualifications)\b',
            r'\b(educational\s+background|degrees)\b',
        ],
        'skills': [
            r'\b(skills|technical\s+skills|core\s+competencies|competencies)\b',
            r'\b(expertise|proficiencies|capabilities)\b',
        ],
        'certifications': [
            r'\b(certifications|certificates|professional\s+certifications)\b',
            r'\b(licenses|credentials)\b',
        ],
        'summary': [
            r'\b(summary|professional\s+summary|profile|career\s+summary)\b',
            r'\b(objective|career\s+objective|about\s+me)\b',
        ],
        'projects': [
            r'\b(projects|key\s+projects|notable\s+projects)\b',
        ],
        'awards': [
            r'\b(awards|honors|achievements|recognition)\b',
        ],
    }
    
    @staticmethod
    def identify_sections(text: str) -> Dict[str, str]:
        """
        Identify and extract sections from resume text
        
        Args:
            text: Resume text
        
        Returns:
            Dictionary mapping section names to their content
        """
        lines = text.split('\n')
        sections = {}
        current_section = 'header'
        current_content = []
        
        # Track section boundaries
        section_boundaries: List[Tuple[int, str, float]] = []
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Check if line is a section header
            detected_section = None
            max_confidence = 0.0
            
            for section_name, patterns in SectionIdentifier.SECTION_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        # Calculate confidence based on line characteristics
                        confidence = SectionIdentifier._calculate_header_confidence(line_stripped)
                        
                        if confidence > max_confidence:
                            max_confidence = confidence
                            detected_section = section_name
            
            # If we detected a section header with good confidence
            if detected_section and max_confidence > 0.5:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = detected_section
                current_content = []
                section_boundaries.append((line_num, detected_section, max_confidence))
                
                logger.debug(f"Detected section '{detected_section}' at line {line_num} with confidence {max_confidence:.2f}")
            else:
                # Add line to current section
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        logger.info(f"Identified {len(sections)} sections: {list(sections.keys())}")
        
        return sections
    
    @staticmethod
    def _calculate_header_confidence(line: str) -> float:
        """
        Calculate confidence that a line is a section header
        
        Args:
            line: Line of text
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.5  # Base confidence for pattern match
        
        # Boost confidence for short lines (likely headers)
        if len(line) < 50:
            confidence += 0.2
        
        # Boost confidence for all caps
        if line.isupper():
            confidence += 0.15
        
        # Boost confidence for lines ending with colon
        if line.endswith(':'):
            confidence += 0.1
        
        # Reduce confidence for very long lines
        if len(line) > 100:
            confidence -= 0.2
        
        # Reduce confidence if line contains many numbers (likely content)
        num_digits = sum(c.isdigit() for c in line)
        if num_digits > 5:
            confidence -= 0.15
        
        return min(1.0, max(0.0, confidence))
    
    @staticmethod
    def get_section_confidence_scores(text: str) -> Dict[str, float]:
        """
        Get confidence scores for detected sections
        
        Args:
            text: Resume text
        
        Returns:
            Dictionary mapping section names to confidence scores
        """
        lines = text.split('\n')
        confidence_scores = {}
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            for section_name, patterns in SectionIdentifier.SECTION_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        confidence = SectionIdentifier._calculate_header_confidence(line_stripped)
                        
                        if section_name not in confidence_scores or confidence > confidence_scores[section_name]:
                            confidence_scores[section_name] = confidence
        
        return confidence_scores
