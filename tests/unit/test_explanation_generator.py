"""Unit tests for explanation generator"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, List

from ml.inference.explanation_generator import (
    ExplanationTemplate, AdvancedExplanationGenerator, 
    MatchLevel, SectionContribution, ExplanationContext,
    generate_enhanced_explanation
)
from backend.app.schemas.resume import ParsedResume, WorkExperience, Education, Skill
from backend.app.models.job import Job, ExperienceLevel


class TestExplanationTemplate:
    """Test the template-based explanation generator"""
    
    @pytest.fixture
    def template_generator(self):
        """Create template generator instance"""
        return ExplanationTemplate()
    
    @pytest.fixture
    def sample_resume(self):
        """Create sample parsed resume"""
        return ParsedResume(
            candidate_name="John Doe",
            email="john@example.com",
            skills=[
                Skill(skill="Python", confidence=0.9),
                Skill(skill="Machine Learning", confidence=0.8),
                Skill(skill="SQL", confidence=0.7)
            ],
            work_experience=[
                WorkExperience(
                    company="Tech Corp",
                    title="Software Engineer",
                    description="Developed ML models",
                    confidence=0.9
                )
            ],
            education=[
                Education(
                    institution="University",
                    degree="Computer Science",
                    confidence=0.8
                )
            ]
        )
    
    @pytest.fixture
    def sample_job(self):
        """Create sample job"""
        job = Mock(spec=Job)
        job.title = "Senior Python Developer"
        job.description = "Looking for experienced Python developer with ML background"
        job.required_skills = ["Python", "Machine Learning", "Django"]
        job.experience_level = ExperienceLevel.SENIOR
        return job
    
    @pytest.fixture
    def sample_context(self, sample_resume, sample_job):
        """Create sample explanation context"""
        return ExplanationContext(
            resume=sample_resume,
            job=sample_job,
            overall_score=0.75,
            section_scores={
                'skills': 0.8,
                'experience': 0.7,
                'education': 0.6
            },
            section_weights={
                'skills': 0.4,
                'experience': 0.4,
                'education': 0.2
            }
        )
    
    def test_determine_match_level(self, template_generator):
        """Test match level determination from scores"""
        assert template_generator._determine_match_level(0.9) == MatchLevel.EXCELLENT
        assert template_generator._determine_match_level(0.7) == MatchLevel.GOOD
        assert template_generator._determine_match_level(0.5) == MatchLevel.MODERATE
        assert template_generator._determine_match_level(0.3) == MatchLevel.POOR
        
        # Test boundary conditions
        assert template_generator._determine_match_level(0.8) == MatchLevel.EXCELLENT
        assert template_generator._determine_match_level(0.6) == MatchLevel.GOOD
        assert template_generator._determine_match_level(0.4) == MatchLevel.MODERATE
    
    def test_analyze_section_contributions(self, template_generator, sample_context):
        """Test section contribution analysis"""
        contributions = template_generator._analyze_section_contributions(sample_context)
        
        # Should have 3 sections
        assert len(contributions) == 3
        
        # Should be sorted by contribution (score * weight)
        assert contributions[0].contribution >= contributions[1].contribution
        assert contributions[1].contribution >= contributions[2].contribution
        
        # Check contribution calculation
        for contrib in contributions:
            expected_contribution = contrib.score * contrib.weight
            assert abs(contrib.contribution - expected_contribution) < 0.001
        
        # Check section names
        section_names = {contrib.section_name for contrib in contributions}
        assert section_names == {'skills', 'experience', 'education'}
    
    def test_analyze_skills_section(self, template_generator, sample_context):
        """Test skills section analysis"""
        key_matches, missing_elements = template_generator._analyze_section_details(
            'skills', sample_context
        )
        
        # Should find matches for Python and Machine Learning
        assert len(key_matches) >= 2
        assert any('python' in match.lower() for match in key_matches)
        assert any('machine learning' in match.lower() for match in key_matches)
        
        # Should identify Django as missing
        assert any('django' in missing.lower() for missing in missing_elements)
    
    def test_analyze_experience_section(self, template_generator, sample_context):
        """Test experience section analysis"""
        key_matches, missing_elements = template_generator._analyze_section_details(
            'experience', sample_context
        )
        
        # Should extract job title and company
        assert len(key_matches) >= 1
        assert any('software engineer' in match.lower() for match in key_matches)
    
    def test_analyze_education_section(self, template_generator, sample_context):
        """Test education section analysis"""
        key_matches, missing_elements = template_generator._analyze_section_details(
            'education', sample_context
        )
        
        # Should extract degree information
        assert len(key_matches) >= 1
        assert any('computer science' in match.lower() for match in key_matches)
    
    def test_skills_similarity(self, template_generator):
        """Test skill similarity matching"""
        # Exact match
        assert template_generator._skills_similar("python", "python")
        
        # Partial match
        assert template_generator._skills_similar("machine learning", "machine")
        assert template_generator._skills_similar("data science", "data analysis")
        
        # No match
        assert not template_generator._skills_similar("python", "java")
    
    def test_generate_explanation_excellent_match(self, template_generator, sample_context):
        """Test explanation generation for excellent match"""
        sample_context.overall_score = 0.85
        explanation = template_generator.generate_explanation(sample_context)
        
        assert "excellent match" in explanation.lower()
        assert "0.85" in explanation
        assert "senior python developer" in explanation.lower()
        assert len(explanation) > 50  # Should be substantial
    
    def test_generate_explanation_poor_match(self, template_generator, sample_context):
        """Test explanation generation for poor match"""
        sample_context.overall_score = 0.25
        sample_context.section_scores = {
            'skills': 0.3,
            'experience': 0.2,
            'education': 0.2
        }
        explanation = template_generator.generate_explanation(sample_context)
        
        assert "poor match" in explanation.lower()
        assert "0.25" in explanation
        assert "limited" in explanation.lower() or "gaps" in explanation.lower()
    
    def test_generate_detailed_explanation(self, template_generator, sample_context):
        """Test detailed explanation generation"""
        explanation = template_generator.generate_explanation(sample_context, detailed=True)
        
        # Should include section scores
        assert "skills score" in explanation.lower()
        assert "experience score" in explanation.lower()
        assert "education score" in explanation.lower()
        
        # Should be longer than basic explanation
        basic_explanation = template_generator.generate_explanation(sample_context, detailed=False)
        assert len(explanation) >= len(basic_explanation)


class TestAdvancedExplanationGenerator:
    """Test the advanced explanation generator"""
    
    @pytest.fixture
    def advanced_generator(self):
        """Create advanced generator instance"""
        return AdvancedExplanationGenerator(enable_llm=False)
    
    @pytest.fixture
    def sample_resume(self):
        """Create sample parsed resume"""
        return ParsedResume(
            candidate_name="Jane Smith",
            email="jane@example.com",
            skills=[
                Skill(skill="React", confidence=0.9),
                Skill(skill="JavaScript", confidence=0.8),
                Skill(skill="Node.js", confidence=0.7)
            ],
            work_experience=[
                WorkExperience(
                    company="Web Solutions",
                    title="Frontend Developer",
                    description="Built React applications",
                    confidence=0.9
                )
            ],
            education=[
                Education(
                    institution="Tech University",
                    degree="Software Engineering",
                    confidence=0.8
                )
            ]
        )
    
    @pytest.fixture
    def sample_job(self):
        """Create sample job"""
        job = Mock(spec=Job)
        job.title = "Full Stack Developer"
        job.description = "Looking for full stack developer with React and Node.js experience"
        job.required_skills = ["React", "Node.js", "MongoDB"]
        job.experience_level = ExperienceLevel.MID
        return job
    
    def test_generate_explanation_basic(self, advanced_generator, sample_resume, sample_job):
        """Test basic explanation generation"""
        scores = {
            'overall_score': 0.7,
            'skills_score': 0.8,
            'experience_score': 0.6,
            'education_score': 0.7
        }
        section_weights = {'skills': 0.4, 'experience': 0.4, 'education': 0.2}
        
        result = advanced_generator.generate_explanation(
            resume=sample_resume,
            job=sample_job,
            scores=scores,
            section_weights=section_weights,
            detailed=False
        )
        
        assert 'explanation' in result
        assert 'match_level' in result
        assert 'overall_score' in result
        assert result['overall_score'] == 0.7
        assert result['match_level'] in ['excellent', 'good', 'moderate', 'poor']
        assert len(result['explanation']) > 0
    
    def test_generate_explanation_detailed(self, advanced_generator, sample_resume, sample_job):
        """Test detailed explanation generation"""
        scores = {
            'overall_score': 0.75,
            'skills_score': 0.8,
            'experience_score': 0.7,
            'education_score': 0.6
        }
        section_weights = {'skills': 0.4, 'experience': 0.4, 'education': 0.2}
        
        result = advanced_generator.generate_explanation(
            resume=sample_resume,
            job=sample_job,
            scores=scores,
            section_weights=section_weights,
            detailed=True
        )
        
        assert 'explanation' in result
        assert 'section_analysis' in result
        assert 'improvement_suggestions' in result
        
        # Check section analysis
        assert len(result['section_analysis']) == 3
        for section in result['section_analysis']:
            assert 'section' in section
            assert 'score' in section
            assert 'weight' in section
            assert 'contribution' in section
            assert 'key_matches' in section
            assert 'missing_elements' in section
        
        # Check improvement suggestions
        assert isinstance(result['improvement_suggestions'], list)
    
    def test_generate_improvement_suggestions(self, advanced_generator):
        """Test improvement suggestion generation"""
        contributions = [
            SectionContribution(
                section_name='skills',
                score=0.4,  # Low score
                weight=0.4,
                contribution=0.16,
                key_matches=['React'],
                missing_elements=['MongoDB', 'Docker']
            ),
            SectionContribution(
                section_name='experience',
                score=0.8,  # High score
                weight=0.4,
                contribution=0.32,
                key_matches=['Frontend Development'],
                missing_elements=[]
            )
        ]
        
        suggestions = advanced_generator._generate_improvement_suggestions(contributions)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest improvement in skills (low score with missing elements)
        skills_suggestion = next((s for s in suggestions if 'skills' in s.lower()), None)
        assert skills_suggestion is not None
        assert 'mongodb' in skills_suggestion.lower() or 'docker' in skills_suggestion.lower()
    
    def test_get_explanation_metadata(self, advanced_generator):
        """Test explanation metadata"""
        metadata = advanced_generator.get_explanation_metadata()
        
        assert 'template_based' in metadata
        assert 'llm_enabled' in metadata
        assert 'detailed_analysis' in metadata
        assert 'section_contribution_analysis' in metadata
        assert 'improvement_suggestions' in metadata
        assert 'version' in metadata
        
        assert metadata['template_based'] is True
        assert metadata['llm_enabled'] is False  # Disabled in test
    
    def test_llm_initialization_disabled(self):
        """Test LLM initialization when disabled"""
        generator = AdvancedExplanationGenerator(enable_llm=False)
        assert generator.enable_llm is False
        assert generator.llm_client is None
    
    def test_llm_initialization_enabled_but_unavailable(self):
        """Test LLM initialization when enabled but unavailable"""
        # This should gracefully fall back to template-based generation
        generator = AdvancedExplanationGenerator(enable_llm=True)
        assert generator.enable_llm is False  # Should be disabled due to unavailability


class TestConvenienceFunction:
    """Test the convenience function for backward compatibility"""
    
    @pytest.fixture
    def sample_resume(self):
        """Create sample parsed resume"""
        return ParsedResume(
            candidate_name="Test User",
            skills=[Skill(skill="Python", confidence=0.9)],
            work_experience=[],
            education=[]
        )
    
    @pytest.fixture
    def sample_job(self):
        """Create sample job"""
        job = Mock(spec=Job)
        job.title = "Python Developer"
        job.description = "Python development role"
        job.required_skills = ["Python"]
        job.experience_level = ExperienceLevel.ENTRY
        return job
    
    def test_generate_enhanced_explanation(self, sample_resume, sample_job):
        """Test the convenience function"""
        scores = {
            'overall_score': 0.6,
            'skills_score': 0.7,
            'experience_score': 0.5,
            'education_score': 0.6
        }
        section_weights = {'skills': 0.4, 'experience': 0.4, 'education': 0.2}
        
        explanation = generate_enhanced_explanation(
            resume=sample_resume,
            job=sample_job,
            scores=scores,
            section_weights=section_weights,
            detailed=False
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "python developer" in explanation.lower()
        assert "0.60" in explanation or "0.6" in explanation


class TestSectionContribution:
    """Test the SectionContribution dataclass"""
    
    def test_section_contribution_creation(self):
        """Test creating section contribution"""
        contrib = SectionContribution(
            section_name='skills',
            score=0.8,
            weight=0.4,
            contribution=0.32,
            key_matches=['Python', 'SQL'],
            missing_elements=['Docker'],
            confidence=0.9
        )
        
        assert contrib.section_name == 'skills'
        assert contrib.score == 0.8
        assert contrib.weight == 0.4
        assert contrib.contribution == 0.32
        assert contrib.key_matches == ['Python', 'SQL']
        assert contrib.missing_elements == ['Docker']
        assert contrib.confidence == 0.9
    
    def test_section_contribution_default_confidence(self):
        """Test default confidence value"""
        contrib = SectionContribution(
            section_name='experience',
            score=0.7,
            weight=0.4,
            contribution=0.28,
            key_matches=[],
            missing_elements=[]
        )
        
        assert contrib.confidence == 1.0  # Default value


class TestExplanationContext:
    """Test the ExplanationContext dataclass"""
    
    def test_explanation_context_creation(self):
        """Test creating explanation context"""
        resume = Mock(spec=ParsedResume)
        job = Mock(spec=Job)
        
        context = ExplanationContext(
            resume=resume,
            job=job,
            overall_score=0.75,
            section_scores={'skills': 0.8},
            section_weights={'skills': 0.4},
            model_type='semantic',
            model_version='2.0.0'
        )
        
        assert context.resume == resume
        assert context.job == job
        assert context.overall_score == 0.75
        assert context.section_scores == {'skills': 0.8}
        assert context.section_weights == {'skills': 0.4}
        assert context.model_type == 'semantic'
        assert context.model_version == '2.0.0'
    
    def test_explanation_context_defaults(self):
        """Test default values in explanation context"""
        resume = Mock(spec=ParsedResume)
        job = Mock(spec=Job)
        
        context = ExplanationContext(
            resume=resume,
            job=job,
            overall_score=0.5,
            section_scores={},
            section_weights={}
        )
        
        assert context.model_type == 'tfidf'  # Default
        assert context.model_version == '1.0.0'  # Default


class TestMatchLevel:
    """Test the MatchLevel enum"""
    
    def test_match_level_values(self):
        """Test match level enum values"""
        assert MatchLevel.EXCELLENT.value == "excellent"
        assert MatchLevel.GOOD.value == "good"
        assert MatchLevel.MODERATE.value == "moderate"
        assert MatchLevel.POOR.value == "poor"
    
    def test_match_level_comparison(self):
        """Test match level enum usage"""
        level = MatchLevel.GOOD
        assert level == MatchLevel.GOOD
        assert level != MatchLevel.EXCELLENT
        assert level.value == "good"