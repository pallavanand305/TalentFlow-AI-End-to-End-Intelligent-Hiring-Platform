"""Scoring engine for candidate-job matching"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging

from backend.app.schemas.resume import ParsedResume
from backend.app.models.job import Job

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Scoring engine for computing candidate-job similarity scores
    
    Implements both TF-IDF baseline and semantic similarity models
    for matching candidates to job descriptions.
    """
    
    def __init__(self, model_type: str = "tfidf"):
        """
        Initialize scoring engine
        
        Args:
            model_type: Type of model to use ("tfidf" or "semantic")
        """
        self.model_type = model_type
        self.tfidf_vectorizer = None
        self.semantic_model = None
        
        # Section weights for scoring
        self.section_weights = {
            'skills': 0.4,
            'experience': 0.4,
            'education': 0.2
        }
        
        if model_type == "tfidf":
            self._initialize_tfidf()
        elif model_type == "semantic":
            self._initialize_semantic()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9+#.]*\b'  # Include tech terms like C++, C#
        )
        logger.info("Initialized TF-IDF scoring engine")
    
    def _initialize_semantic(self):
        """Initialize semantic similarity model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized semantic scoring engine")
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to TF-IDF")
            self.model_type = "tfidf"
            self._initialize_tfidf()
    
    def score_candidate(
        self, 
        resume: ParsedResume, 
        job: Job
    ) -> Dict[str, float]:
        """
        Compute similarity score between resume and job
        
        Args:
            resume: Parsed resume data
            job: Job description
        
        Returns:
            Dictionary containing overall score and section scores
        """
        logger.info(f"Scoring candidate for job: {job.title}")
        
        # Extract text from resume sections
        resume_sections = self._extract_resume_sections(resume)
        job_sections = self._extract_job_sections(job)
        
        # Compute section-wise similarities
        section_scores = {}
        
        if self.model_type == "tfidf":
            section_scores = self._compute_tfidf_scores(resume_sections, job_sections)
        elif self.model_type == "semantic":
            section_scores = self._compute_semantic_scores(resume_sections, job_sections)
        
        # Compute weighted overall score
        overall_score = self._compute_weighted_score(section_scores)
        
        # Normalize to [0, 1] range
        overall_score = max(0.0, min(1.0, overall_score))
        
        result = {
            'overall_score': overall_score,
            'skills_score': section_scores.get('skills', 0.0),
            'experience_score': section_scores.get('experience', 0.0),
            'education_score': section_scores.get('education', 0.0)
        }
        
        logger.info(f"Computed score: {overall_score:.3f}")
        return result
    
    def _extract_resume_sections(self, resume: ParsedResume) -> Dict[str, str]:
        """
        Extract text from resume sections
        
        Args:
            resume: Parsed resume data
        
        Returns:
            Dictionary of section texts
        """
        sections = {}
        
        # Skills section
        skills_text = ' '.join([skill.skill for skill in resume.skills])
        sections['skills'] = skills_text
        
        # Experience section
        experience_texts = []
        for exp in resume.work_experience:
            exp_text = f"{exp.title or ''} {exp.company or ''} {exp.description}"
            experience_texts.append(exp_text)
        sections['experience'] = ' '.join(experience_texts)
        
        # Education section
        education_texts = []
        for edu in resume.education:
            edu_text = f"{edu.degree or ''} {edu.field_of_study or ''} {edu.institution or ''} {edu.description}"
            education_texts.append(edu_text)
        sections['education'] = ' '.join(education_texts)
        
        return sections
    
    def _extract_job_sections(self, job: Job) -> Dict[str, str]:
        """
        Extract text from job sections
        
        Args:
            job: Job description
        
        Returns:
            Dictionary of section texts
        """
        sections = {}
        
        # Skills section (required skills)
        sections['skills'] = ' '.join(job.required_skills)
        
        # Experience section (from job description)
        # Extract experience-related keywords from description
        experience_keywords = self._extract_experience_keywords(job.description)
        sections['experience'] = f"{job.experience_level.value} {experience_keywords}"
        
        # Education section (from job description)
        # Extract education-related keywords from description
        education_keywords = self._extract_education_keywords(job.description)
        sections['education'] = education_keywords
        
        return sections
    
    def _extract_experience_keywords(self, description: str) -> str:
        """Extract experience-related keywords from job description"""
        experience_patterns = [
            r'\b\d+\+?\s*years?\s*(?:of\s*)?(?:experience|exp)\b',
            r'\bexperience\s+(?:with|in|using)\s+[\w\s,]+',
            r'\b(?:senior|junior|lead|principal|staff)\b',
            r'\b(?:develop|build|design|implement|manage|lead)\b',
            r'\b(?:project|team|product|system)\s+(?:management|leadership|development)\b'
        ]
        
        keywords = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            keywords.extend(matches)
        
        return ' '.join(keywords)
    
    def _extract_education_keywords(self, description: str) -> str:
        """Extract education-related keywords from job description"""
        education_patterns = [
            r'\b(?:bachelor|master|phd|doctorate|degree)\b',
            r'\b(?:bs|ba|ms|ma|mba|phd)\b',
            r'\b(?:computer\s+science|engineering|mathematics|physics)\b',
            r'\b(?:university|college|institute)\b',
            r'\b(?:certification|certified)\b'
        ]
        
        keywords = []
        for pattern in education_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            keywords.extend(matches)
        
        return ' '.join(keywords)
    
    def _compute_tfidf_scores(
        self, 
        resume_sections: Dict[str, str], 
        job_sections: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Compute TF-IDF based similarity scores
        
        Args:
            resume_sections: Resume section texts
            job_sections: Job section texts
        
        Returns:
            Dictionary of section similarity scores
        """
        scores = {}
        
        for section in ['skills', 'experience', 'education']:
            resume_text = resume_sections.get(section, '')
            job_text = job_sections.get(section, '')
            
            if not resume_text.strip() or not job_text.strip():
                scores[section] = 0.0
                continue
            
            try:
                # Fit TF-IDF on both texts
                texts = [resume_text, job_text]
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                
                # Compute cosine similarity
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                scores[section] = float(similarity)
                
            except Exception as e:
                logger.warning(f"Error computing TF-IDF score for {section}: {e}")
                scores[section] = 0.0
        
        return scores
    
    def _compute_semantic_scores(
        self, 
        resume_sections: Dict[str, str], 
        job_sections: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Compute semantic similarity scores using sentence transformers
        
        Args:
            resume_sections: Resume section texts
            job_sections: Job section texts
        
        Returns:
            Dictionary of section similarity scores
        """
        scores = {}
        
        for section in ['skills', 'experience', 'education']:
            resume_text = resume_sections.get(section, '')
            job_text = job_sections.get(section, '')
            
            if not resume_text.strip() or not job_text.strip():
                scores[section] = 0.0
                continue
            
            try:
                # Generate embeddings
                resume_embedding = self.semantic_model.encode([resume_text])
                job_embedding = self.semantic_model.encode([job_text])
                
                # Compute cosine similarity
                similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
                scores[section] = float(similarity)
                
            except Exception as e:
                logger.warning(f"Error computing semantic score for {section}: {e}")
                scores[section] = 0.0
        
        return scores
    
    def _compute_weighted_score(self, section_scores: Dict[str, float]) -> float:
        """
        Compute weighted overall score from section scores
        
        Args:
            section_scores: Dictionary of section scores
        
        Returns:
            Weighted overall score
        """
        weighted_score = 0.0
        total_weight = 0.0
        
        for section, weight in self.section_weights.items():
            if section in section_scores:
                weighted_score += section_scores[section] * weight
                total_weight += weight
        
        # Normalize by total weight to handle missing sections
        if total_weight > 0:
            weighted_score /= total_weight
        
        return weighted_score
    
    def batch_score_candidates(
        self, 
        resumes: List[ParsedResume], 
        job: Job
    ) -> List[Dict[str, float]]:
        """
        Score multiple candidates against a job
        
        Args:
            resumes: List of parsed resumes
            job: Job description
        
        Returns:
            List of score dictionaries
        """
        logger.info(f"Batch scoring {len(resumes)} candidates for job: {job.title}")
        
        scores = []
        for resume in resumes:
            try:
                score = self.score_candidate(resume, job)
                scores.append(score)
            except Exception as e:
                logger.error(f"Error scoring candidate: {e}")
                # Return zero score for failed candidates
                scores.append({
                    'overall_score': 0.0,
                    'skills_score': 0.0,
                    'experience_score': 0.0,
                    'education_score': 0.0
                })
        
        return scores
    
    def rank_candidates(
        self, 
        candidate_scores: List[Tuple[str, Dict[str, float]]]
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Rank candidates by their overall scores
        
        Args:
            candidate_scores: List of (candidate_id, scores) tuples
        
        Returns:
            Sorted list of (candidate_id, scores) tuples
        """
        return sorted(
            candidate_scores, 
            key=lambda x: x[1]['overall_score'], 
            reverse=True
        )
    
    def explain_score(
        self, 
        resume: ParsedResume, 
        job: Job, 
        scores: Dict[str, float],
        detailed: bool = False,
        use_advanced_generator: bool = True
    ) -> str:
        """
        Generate natural language explanation for the score
        
        Args:
            resume: Parsed resume data
            job: Job description
            scores: Computed scores
            detailed: Whether to include detailed section analysis
            use_advanced_generator: Whether to use the advanced explanation generator
        
        Returns:
            Natural language explanation
        """
        if use_advanced_generator:
            try:
                from ml.inference.explanation_generator import generate_enhanced_explanation
                return generate_enhanced_explanation(
                    resume=resume,
                    job=job,
                    scores=scores,
                    section_weights=self.section_weights,
                    detailed=detailed
                )
            except ImportError:
                logger.warning("Advanced explanation generator not available, using basic explanation")
        
        # Fallback to basic explanation (original implementation)
        return self._generate_basic_explanation(resume, job, scores)
    
    def _generate_basic_explanation(
        self, 
        resume: ParsedResume, 
        job: Job, 
        scores: Dict[str, float]
    ) -> str:
        """
        Generate basic explanation (original implementation)
        
        Args:
            resume: Parsed resume data
            job: Job description
            scores: Computed scores
        
        Returns:
            Basic natural language explanation
        """
        overall_score = scores['overall_score']
        skills_score = scores['skills_score']
        experience_score = scores['experience_score']
        education_score = scores['education_score']
        
        # Determine overall match level
        if overall_score >= 0.8:
            match_level = "excellent"
        elif overall_score >= 0.6:
            match_level = "good"
        elif overall_score >= 0.4:
            match_level = "moderate"
        else:
            match_level = "poor"
        
        explanation = f"This candidate shows a {match_level} match (score: {overall_score:.2f}) for the {job.title} position. "
        
        # Skills analysis
        if skills_score >= 0.7:
            explanation += f"The candidate's skills are well-aligned with the job requirements (skills score: {skills_score:.2f}). "
        elif skills_score >= 0.4:
            explanation += f"The candidate has some relevant skills but may need additional training (skills score: {skills_score:.2f}). "
        else:
            explanation += f"The candidate's skills show limited alignment with job requirements (skills score: {skills_score:.2f}). "
        
        # Experience analysis
        if experience_score >= 0.7:
            explanation += f"Their experience background is highly relevant (experience score: {experience_score:.2f}). "
        elif experience_score >= 0.4:
            explanation += f"They have some relevant experience (experience score: {experience_score:.2f}). "
        else:
            explanation += f"Their experience may not be directly relevant (experience score: {experience_score:.2f}). "
        
        # Education analysis
        if education_score >= 0.7:
            explanation += f"Their educational background is well-suited for this role (education score: {education_score:.2f})."
        elif education_score >= 0.4:
            explanation += f"Their education provides a reasonable foundation (education score: {education_score:.2f})."
        else:
            explanation += f"Their educational background may not be directly relevant (education score: {education_score:.2f})."
        
        return explanation
    
    def generate_detailed_explanation(
        self, 
        resume: ParsedResume, 
        job: Job, 
        scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation with section-wise analysis
        
        Args:
            resume: Parsed resume data
            job: Job description
            scores: Computed scores
        
        Returns:
            Dictionary containing detailed explanation and analysis
        """
        try:
            from ml.inference.explanation_generator import AdvancedExplanationGenerator
            
            generator = AdvancedExplanationGenerator()
            return generator.generate_explanation(
                resume=resume,
                job=job,
                scores=scores,
                section_weights=self.section_weights,
                detailed=True
            )
        except ImportError:
            logger.warning("Advanced explanation generator not available")
            return {
                'explanation': self._generate_basic_explanation(resume, job, scores),
                'match_level': 'unknown',
                'overall_score': scores.get('overall_score', 0.0),
                'section_analysis': []
            }
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_type,
            'version': '1.0.0',
            'description': f'{"TF-IDF baseline" if self.model_type == "tfidf" else "Semantic similarity"} scoring engine',
            'section_weights': str(self.section_weights)
        }


class ScoringResult:
    """Container for scoring results"""
    
    def __init__(
        self,
        candidate_id: str,
        job_id: str,
        overall_score: float,
        section_scores: Dict[str, float],
        explanation: Optional[str] = None,
        model_version: str = "1.0.0"
    ):
        self.candidate_id = candidate_id
        self.job_id = job_id
        self.overall_score = overall_score
        self.section_scores = section_scores
        self.explanation = explanation
        self.model_version = model_version
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'candidate_id': self.candidate_id,
            'job_id': self.job_id,
            'overall_score': self.overall_score,
            'section_scores': self.section_scores,
            'explanation': self.explanation,
            'model_version': self.model_version
        }