"""
Advanced explanation generator for candidate-job scoring

This module provides template-based explanation generation with section-wise
contribution analysis and optional LLM integration for natural language explanations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import re

if TYPE_CHECKING:
    from backend.app.schemas.resume import ParsedResume
    from backend.app.models.job import Job

logger = logging.getLogger(__name__)


class MatchLevel(Enum):
    """Match level categories for scoring"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    MODERATE = "moderate"
    POOR = "poor"


@dataclass
class SectionContribution:
    """Represents the contribution of a resume section to the overall score"""
    section_name: str
    score: float
    weight: float
    contribution: float  # score * weight
    key_matches: List[str]
    missing_elements: List[str]
    confidence: float = 1.0


@dataclass
class ExplanationContext:
    """Context information for generating explanations"""
    resume: 'ParsedResume'
    job: 'Job'
    overall_score: float
    section_scores: Dict[str, float]
    section_weights: Dict[str, float]
    model_type: str = "tfidf"
    model_version: str = "1.0.0"


class ExplanationTemplate:
    """Template-based explanation generator"""
    
    def __init__(self):
        """Initialize explanation templates"""
        self.templates = {
            MatchLevel.EXCELLENT: {
                "opening": "This candidate shows an {match_level} match (score: {score:.2f}) for the {job_title} position.",
                "skills_high": "The candidate's skills are exceptionally well-aligned with the job requirements (skills score: {skills_score:.2f}). Key matching skills include: {key_skills}.",
                "skills_medium": "The candidate has strong relevant skills (skills score: {skills_score:.2f}) with good alignment in: {key_skills}.",
                "experience_high": "Their experience background is highly relevant (experience score: {experience_score:.2f}), demonstrating {experience_highlights}.",
                "experience_medium": "They bring valuable relevant experience (experience score: {experience_score:.2f}) in {experience_areas}.",
                "education_high": "Their educational background is perfectly suited for this role (education score: {education_score:.2f}).",
                "education_medium": "Their education provides a solid foundation (education score: {education_score:.2f}).",
                "closing": "This candidate would be an excellent fit for the position."
            },
            MatchLevel.GOOD: {
                "opening": "This candidate shows a {match_level} match (score: {score:.2f}) for the {job_title} position.",
                "skills_high": "The candidate's skills are well-aligned with the job requirements (skills score: {skills_score:.2f}). Strong matches include: {key_skills}.",
                "skills_medium": "The candidate has relevant skills (skills score: {skills_score:.2f}) with good coverage in: {key_skills}.",
                "skills_low": "The candidate has some relevant skills (skills score: {skills_score:.2f}) but may benefit from additional training in: {missing_skills}.",
                "experience_high": "Their experience background is highly relevant (experience score: {experience_score:.2f}).",
                "experience_medium": "They have solid relevant experience (experience score: {experience_score:.2f}).",
                "experience_low": "Their experience provides some relevant background (experience score: {experience_score:.2f}).",
                "education_high": "Their educational background is well-suited for this role (education score: {education_score:.2f}).",
                "education_medium": "Their education provides a good foundation (education score: {education_score:.2f}).",
                "education_low": "Their educational background provides basic preparation (education score: {education_score:.2f}).",
                "closing": "This candidate would be a good fit for the position with potential for growth."
            },
            MatchLevel.MODERATE: {
                "opening": "This candidate shows a {match_level} match (score: {score:.2f}) for the {job_title} position.",
                "skills_medium": "The candidate has some relevant skills (skills score: {skills_score:.2f}) including: {key_skills}.",
                "skills_low": "The candidate's skills show partial alignment (skills score: {skills_score:.2f}) but would benefit from development in: {missing_skills}.",
                "experience_medium": "They have some relevant experience (experience score: {experience_score:.2f}).",
                "experience_low": "Their experience provides limited direct relevance (experience score: {experience_score:.2f}).",
                "education_medium": "Their education provides a reasonable foundation (education score: {education_score:.2f}).",
                "education_low": "Their educational background may not be directly relevant (education score: {education_score:.2f}).",
                "closing": "This candidate could be considered with additional training and development."
            },
            MatchLevel.POOR: {
                "opening": "This candidate shows a {match_level} match (score: {score:.2f}) for the {job_title} position.",
                "skills_low": "The candidate's skills show limited alignment with job requirements (skills score: {skills_score:.2f}). Significant gaps exist in: {missing_skills}.",
                "experience_low": "Their experience may not be directly relevant (experience score: {experience_score:.2f}).",
                "education_low": "Their educational background may not be directly relevant (education score: {education_score:.2f}).",
                "closing": "This candidate would require substantial additional training to be suitable for this position."
            }
        }
    
    def generate_explanation(
        self, 
        context: ExplanationContext,
        detailed: bool = False
    ) -> str:
        """
        Generate template-based explanation
        
        Args:
            context: Explanation context with all necessary data
            detailed: Whether to include detailed section analysis
        
        Returns:
            Generated explanation string
        """
        match_level = self._determine_match_level(context.overall_score)
        template = self.templates[match_level]
        
        # Build explanation components
        components = []
        
        # Opening statement
        components.append(template["opening"].format(
            match_level=match_level.value,
            score=context.overall_score,
            job_title=context.job.title
        ))
        
        # Section-wise analysis
        section_contributions = self._analyze_section_contributions(context)
        
        for contribution in section_contributions:
            section_text = self._generate_section_explanation(
                contribution, template, detailed
            )
            if section_text:
                components.append(section_text)
        
        # Closing statement
        if "closing" in template:
            components.append(template["closing"])
        
        return " ".join(components)
    
    def _determine_match_level(self, score: float) -> MatchLevel:
        """Determine match level from overall score"""
        if score >= 0.8:
            return MatchLevel.EXCELLENT
        elif score >= 0.6:
            return MatchLevel.GOOD
        elif score >= 0.4:
            return MatchLevel.MODERATE
        else:
            return MatchLevel.POOR
    
    def _analyze_section_contributions(
        self, 
        context: ExplanationContext
    ) -> List[SectionContribution]:
        """Analyze how each section contributes to the overall score"""
        contributions = []
        
        for section in ['skills', 'experience', 'education']:
            score = context.section_scores.get(section, 0.0)
            weight = context.section_weights.get(section, 0.0)
            contribution = score * weight
            
            # Analyze key matches and missing elements
            key_matches, missing_elements = self._analyze_section_details(
                section, context
            )
            
            contributions.append(SectionContribution(
                section_name=section,
                score=score,
                weight=weight,
                contribution=contribution,
                key_matches=key_matches,
                missing_elements=missing_elements
            ))
        
        # Sort by contribution (highest first)
        contributions.sort(key=lambda x: x.contribution, reverse=True)
        return contributions
    
    def _analyze_section_details(
        self, 
        section: str, 
        context: ExplanationContext
    ) -> Tuple[List[str], List[str]]:
        """Analyze specific matches and gaps for a section"""
        key_matches = []
        missing_elements = []
        
        if section == 'skills':
            # Analyze skill matches
            resume_skills = [skill.skill.lower() for skill in context.resume.skills]
            job_skills = [skill.lower() for skill in context.job.required_skills]
            
            # Find matches
            for job_skill in job_skills:
                for resume_skill in resume_skills:
                    if (job_skill in resume_skill or 
                        resume_skill in job_skill or
                        self._skills_similar(job_skill, resume_skill)):
                        key_matches.append(job_skill.title())
                        break
            
            # Find missing skills
            matched_lower = [m.lower() for m in key_matches]
            for job_skill in job_skills:
                if job_skill not in matched_lower:
                    missing_elements.append(job_skill.title())
        
        elif section == 'experience':
            # Analyze experience relevance
            if context.resume.work_experience:
                # Extract key experience indicators
                for exp in context.resume.work_experience:
                    if exp.title:
                        key_matches.append(exp.title)
                    if exp.company:
                        key_matches.append(f"experience at {exp.company}")
            
            # Check for experience level match
            job_exp_level = context.job.experience_level.value
            if job_exp_level not in ['entry']:
                # For non-entry level positions, check if candidate has sufficient experience
                total_exp_years = sum(
                    self._estimate_experience_years(exp) 
                    for exp in context.resume.work_experience
                )
                if total_exp_years < 2:
                    missing_elements.append(f"{job_exp_level} level experience")
        
        elif section == 'education':
            # Analyze education relevance
            if context.resume.education:
                for edu in context.resume.education:
                    if edu.degree:
                        key_matches.append(edu.degree)
                    if edu.field_of_study:
                        key_matches.append(edu.field_of_study)
            
            # Check for education requirements in job description
            job_desc_lower = context.job.description.lower()
            education_keywords = [
                'bachelor', 'master', 'phd', 'degree', 'university', 'college'
            ]
            
            required_education = []
            for keyword in education_keywords:
                if keyword in job_desc_lower:
                    required_education.append(keyword)
            
            if required_education and not context.resume.education:
                missing_elements.extend(required_education)
        
        # Limit to most relevant items
        key_matches = key_matches[:5]
        missing_elements = missing_elements[:3]
        
        return key_matches, missing_elements
    
    def _skills_similar(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are similar (basic similarity check)"""
        # Simple similarity check - can be enhanced with more sophisticated matching
        skill1_words = set(skill1.split())
        skill2_words = set(skill2.split())
        
        # Check for common words
        common_words = skill1_words.intersection(skill2_words)
        return len(common_words) > 0
    
    def _estimate_experience_years(self, experience) -> int:
        """Estimate years of experience from work experience entry"""
        # Simple estimation - can be enhanced with date parsing
        if hasattr(experience, 'start_date') and hasattr(experience, 'end_date'):
            # If we had proper date parsing, we'd calculate the difference
            # For now, assume each job is ~2 years on average
            return 2
        return 1
    
    def _generate_section_explanation(
        self, 
        contribution: SectionContribution,
        template: Dict[str, str],
        detailed: bool
    ) -> Optional[str]:
        """Generate explanation text for a specific section"""
        section = contribution.section_name
        score = contribution.score
        
        # Determine score level for template selection
        if score >= 0.7:
            level = "high"
        elif score >= 0.4:
            level = "medium"
        else:
            level = "low"
        
        # Get template key
        template_key = f"{section}_{level}"
        
        if template_key not in template:
            return None
        
        # Prepare template variables
        template_vars = {
            f"{section}_score": score
        }
        
        # Add section-specific variables
        if section == 'skills':
            template_vars['key_skills'] = ', '.join(contribution.key_matches[:3]) or 'various technical skills'
            template_vars['missing_skills'] = ', '.join(contribution.missing_elements[:3]) or 'additional technical areas'
        elif section == 'experience':
            template_vars['experience_highlights'] = ', '.join(contribution.key_matches[:2]) or 'relevant professional background'
            template_vars['experience_areas'] = ', '.join(contribution.key_matches[:3]) or 'related fields'
        
        try:
            return template[template_key].format(**template_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return None


class AdvancedExplanationGenerator:
    """
    Advanced explanation generator with multiple explanation strategies
    """
    
    def __init__(self, enable_llm: bool = False, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize advanced explanation generator
        
        Args:
            enable_llm: Whether to enable LLM-based explanations
            llm_config: Configuration for LLM integration
        """
        self.template_generator = ExplanationTemplate()
        self.enable_llm = enable_llm
        self.llm_config = llm_config or {}
        
        # Initialize LLM client if enabled
        self.llm_client = None
        if enable_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client for natural language generation"""
        try:
            # Placeholder for LLM integration (e.g., OpenAI, Anthropic, local models)
            # This would be implemented based on the chosen LLM provider
            logger.info("LLM integration not implemented yet - using template-based generation")
            self.enable_llm = False
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            self.enable_llm = False
    
    def generate_explanation(
        self,
        resume: 'ParsedResume',
        job: 'Job',
        scores: Dict[str, float],
        section_weights: Dict[str, float],
        detailed: bool = False,
        use_llm: bool = False
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a candidate-job score
        
        Args:
            resume: Parsed resume data
            job: Job description
            scores: Dictionary with overall_score and section scores
            section_weights: Weights used for each section
            detailed: Whether to include detailed analysis
            use_llm: Whether to use LLM for generation (if available)
        
        Returns:
            Dictionary containing explanation and analysis
        """
        logger.info(f"Generating explanation for score: {scores.get('overall_score', 0):.3f}")
        
        # Create explanation context
        context = ExplanationContext(
            resume=resume,
            job=job,
            overall_score=scores.get('overall_score', 0.0),
            section_scores={
                'skills': scores.get('skills_score', 0.0),
                'experience': scores.get('experience_score', 0.0),
                'education': scores.get('education_score', 0.0)
            },
            section_weights=section_weights
        )
        
        # Generate template-based explanation
        template_explanation = self.template_generator.generate_explanation(
            context, detailed=detailed
        )
        
        # Analyze section contributions
        section_contributions = self.template_generator._analyze_section_contributions(context)
        
        # Prepare result
        result = {
            'explanation': template_explanation,
            'match_level': self.template_generator._determine_match_level(
                context.overall_score
            ).value,
            'overall_score': context.overall_score,
            'section_analysis': []
        }
        
        # Add detailed section analysis if requested
        if detailed:
            for contrib in section_contributions:
                section_info = {
                    'section': contrib.section_name,
                    'score': contrib.score,
                    'weight': contrib.weight,
                    'contribution': contrib.contribution,
                    'key_matches': contrib.key_matches,
                    'missing_elements': contrib.missing_elements
                }
                result['section_analysis'].append(section_info)
            
            # Add improvement suggestions
            result['improvement_suggestions'] = self._generate_improvement_suggestions(
                section_contributions
            )
        
        # Generate LLM explanation if requested and available
        if use_llm and self.enable_llm:
            try:
                llm_explanation = self._generate_llm_explanation(context)
                result['llm_explanation'] = llm_explanation
            except Exception as e:
                logger.warning(f"LLM explanation generation failed: {e}")
        
        return result
    
    def _generate_improvement_suggestions(
        self, 
        contributions: List[SectionContribution]
    ) -> List[str]:
        """Generate suggestions for improving candidate-job match"""
        suggestions = []
        
        for contrib in contributions:
            if contrib.score < 0.6 and contrib.missing_elements:
                section_name = contrib.section_name.title()
                missing = ', '.join(contrib.missing_elements[:2])
                suggestions.append(
                    f"Consider developing {section_name.lower()} in: {missing}"
                )
        
        # Add general suggestions based on overall score
        overall_score = contributions[0].contribution if contributions else 0
        if overall_score < 0.5:
            suggestions.append(
                "Consider additional training or certification to better align with job requirements"
            )
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _generate_llm_explanation(self, context: ExplanationContext) -> str:
        """Generate LLM-based natural language explanation"""
        # Placeholder for LLM integration
        # This would make an API call to the configured LLM service
        # with a carefully crafted prompt including the context information
        
        prompt = self._build_llm_prompt(context)
        
        # For now, return a placeholder
        return "LLM-based explanation not yet implemented"
    
    def _build_llm_prompt(self, context: ExplanationContext) -> str:
        """Build prompt for LLM explanation generation"""
        prompt = f"""
        Generate a professional explanation for why a candidate received a score of {context.overall_score:.2f} for the position "{context.job.title}".
        
        Job Requirements:
        - Required Skills: {', '.join(context.job.required_skills)}
        - Experience Level: {context.job.experience_level.value}
        - Description: {context.job.description[:200]}...
        
        Candidate Profile:
        - Skills: {', '.join([skill.skill for skill in context.resume.skills[:10]])}
        - Experience: {len(context.resume.work_experience)} positions
        - Education: {len(context.resume.education)} degrees/certifications
        
        Section Scores:
        - Skills: {context.section_scores['skills']:.2f}
        - Experience: {context.section_scores['experience']:.2f}
        - Education: {context.section_scores['education']:.2f}
        
        Please provide a clear, professional explanation that highlights strengths, identifies gaps, and suggests areas for improvement.
        """
        
        return prompt
    
    def get_explanation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the explanation generator"""
        return {
            'template_based': True,
            'llm_enabled': self.enable_llm,
            'detailed_analysis': True,
            'section_contribution_analysis': True,
            'improvement_suggestions': True,
            'version': '1.0.0'
        }


# Convenience function for backward compatibility
def generate_enhanced_explanation(
    resume: 'ParsedResume',
    job: 'Job',
    scores: Dict[str, float],
    section_weights: Dict[str, float],
    detailed: bool = False
) -> str:
    """
    Generate enhanced explanation using the advanced generator
    
    Args:
        resume: Parsed resume data
        job: Job description
        scores: Score dictionary
        section_weights: Section weights
        detailed: Whether to include detailed analysis
    
    Returns:
        Generated explanation string
    """
    generator = AdvancedExplanationGenerator()
    result = generator.generate_explanation(
        resume=resume,
        job=job,
        scores=scores,
        section_weights=section_weights,
        detailed=detailed
    )
    return result['explanation']