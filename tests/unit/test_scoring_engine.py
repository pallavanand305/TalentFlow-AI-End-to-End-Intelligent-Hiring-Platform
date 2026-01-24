"""Unit tests for scoring engine"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from ml.inference.scoring_engine import ScoringEngine, ScoringResult
from backend.app.schemas.resume import ParsedResume, WorkExperience, Education, Skill
from backend.app.models.job import Job, ExperienceLevel, JobStatus
from uuid import uuid4


class TestScoringEngine:
    """Unit tests for ScoringEngine"""
    
    @pytest.fixture
    def sample_resume(self):
        """Create sample parsed resume"""
        return ParsedResume(
            raw_text="John Doe Software Engineer",
            sections={
                "experience": "Software Engineer at TechCorp",
                "education": "BS Computer Science",
                "skills": "Python, Java, SQL"
            },
            work_experience=[
                WorkExperience(
                    company="TechCorp",
                    title="Software Engineer",
                    start_date="2020-01",
                    end_date="2023-12",
                    description="Developed web applications using Python and Django",
                    confidence=0.9
                )
            ],
            education=[
                Education(
                    institution="State University",
                    degree="Bachelor of Science",
                    field_of_study="Computer Science",
                    start_date="2016",
                    end_date="2020",
                    description="BS Computer Science",
                    confidence=0.9
                )
            ],
            skills=[
                Skill(skill="Python", confidence=0.8),
                Skill(skill="Java", confidence=0.7),
                Skill(skill="SQL", confidence=0.8),
                Skill(skill="Django", confidence=0.7)
            ],
            certifications=[],
            low_confidence_fields=[],
            file_format="pdf"
        )
    
    @pytest.fixture
    def sample_job(self):
        """Create sample job"""
        return Job(
            id=uuid4(),
            title="Senior Python Developer",
            description="We are looking for a senior Python developer with experience in web development and databases.",
            required_skills=["Python", "Django", "PostgreSQL", "REST APIs"],
            experience_level=ExperienceLevel.SENIOR,
            location="San Francisco, CA",
            salary_min=120000,
            salary_max=180000,
            status=JobStatus.ACTIVE,
            created_by=uuid4()
        )
    
    def test_tfidf_engine_initialization(self):
        """Test TF-IDF engine initialization"""
        engine = ScoringEngine(model_type="tfidf")
        
        assert engine.model_type == "tfidf"
        assert engine.tfidf_vectorizer is not None
        assert engine.semantic_model is None
        assert engine.section_weights == {
            'skills': 0.4,
            'experience': 0.4,
            'education': 0.2
        }
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_engine_initialization(self, mock_transformer):
        """Test semantic engine initialization"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        engine = ScoringEngine(model_type="semantic")
        
        assert engine.model_type == "semantic"
        assert engine.semantic_model == mock_model
        mock_transformer.assert_called_once_with('all-MiniLM-L6-v2')
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type"""
        with pytest.raises(ValueError, match="Unsupported model type"):
            ScoringEngine(model_type="invalid")
    
    def test_extract_resume_sections(self, sample_resume):
        """Test resume section extraction"""
        engine = ScoringEngine(model_type="tfidf")
        sections = engine._extract_resume_sections(sample_resume)
        
        assert 'skills' in sections
        assert 'experience' in sections
        assert 'education' in sections
        
        assert "Python" in sections['skills']
        assert "Java" in sections['skills']
        assert "TechCorp" in sections['experience']
        assert "Software Engineer" in sections['experience']
        assert "Computer Science" in sections['education']
    
    def test_extract_job_sections(self, sample_job):
        """Test job section extraction"""
        engine = ScoringEngine(model_type="tfidf")
        sections = engine._extract_job_sections(sample_job)
        
        assert 'skills' in sections
        assert 'experience' in sections
        assert 'education' in sections
        
        assert "Python" in sections['skills']
        assert "Django" in sections['skills']
        assert "senior" in sections['experience']
    
    def test_extract_experience_keywords(self):
        """Test experience keyword extraction"""
        engine = ScoringEngine(model_type="tfidf")
        description = "We need 5+ years of experience with Python development and team leadership."
        
        keywords = engine._extract_experience_keywords(description)
        
        assert "5+ years" in keywords or "experience" in keywords
    
    def test_extract_education_keywords(self):
        """Test education keyword extraction"""
        engine = ScoringEngine(model_type="tfidf")
        description = "Bachelor's degree in Computer Science or related field required."
        
        keywords = engine._extract_education_keywords(description)
        
        assert "bachelor" in keywords.lower() or "computer science" in keywords.lower()
    
    def test_compute_weighted_score(self):
        """Test weighted score computation"""
        engine = ScoringEngine(model_type="tfidf")
        section_scores = {
            'skills': 0.8,
            'experience': 0.6,
            'education': 0.4
        }
        
        weighted_score = engine._compute_weighted_score(section_scores)
        
        # Expected: (0.8 * 0.4 + 0.6 * 0.4 + 0.4 * 0.2) / 1.0 = 0.64
        expected = 0.64
        assert abs(weighted_score - expected) < 0.01
    
    def test_compute_weighted_score_missing_sections(self):
        """Test weighted score computation with missing sections"""
        engine = ScoringEngine(model_type="tfidf")
        section_scores = {
            'skills': 0.8,
            'experience': 0.6
            # education missing
        }
        
        weighted_score = engine._compute_weighted_score(section_scores)
        
        # Should normalize by available weights: (0.8 * 0.4 + 0.6 * 0.4) / 0.8 = 0.7
        expected = 0.7
        assert abs(weighted_score - expected) < 0.01
    
    @patch('ml.inference.scoring_engine.cosine_similarity')
    def test_compute_tfidf_scores(self, mock_cosine, sample_resume, sample_job):
        """Test TF-IDF score computation"""
        mock_cosine.return_value = [[0.75]]
        
        engine = ScoringEngine(model_type="tfidf")
        resume_sections = engine._extract_resume_sections(sample_resume)
        job_sections = engine._extract_job_sections(sample_job)
        
        scores = engine._compute_tfidf_scores(resume_sections, job_sections)
        
        assert 'skills' in scores
        assert 'experience' in scores
        assert 'education' in scores
        assert all(0 <= score <= 1 for score in scores.values())
    
    def test_score_candidate_tfidf(self, sample_resume, sample_job):
        """Test candidate scoring with TF-IDF"""
        engine = ScoringEngine(model_type="tfidf")
        
        result = engine.score_candidate(sample_resume, sample_job)
        
        assert 'overall_score' in result
        assert 'skills_score' in result
        assert 'experience_score' in result
        assert 'education_score' in result
        
        assert 0 <= result['overall_score'] <= 1
        assert all(0 <= score <= 1 for score in result.values())
    
    def test_batch_score_candidates(self, sample_resume, sample_job):
        """Test batch candidate scoring"""
        engine = ScoringEngine(model_type="tfidf")
        resumes = [sample_resume, sample_resume]  # Duplicate for testing
        
        scores = engine.batch_score_candidates(resumes, sample_job)
        
        assert len(scores) == 2
        assert all('overall_score' in score for score in scores)
    
    def test_rank_candidates(self):
        """Test candidate ranking"""
        engine = ScoringEngine(model_type="tfidf")
        
        candidate_scores = [
            ("candidate1", {"overall_score": 0.6}),
            ("candidate2", {"overall_score": 0.8}),
            ("candidate3", {"overall_score": 0.4})
        ]
        
        ranked = engine.rank_candidates(candidate_scores)
        
        assert len(ranked) == 3
        assert ranked[0][0] == "candidate2"  # Highest score first
        assert ranked[1][0] == "candidate1"
        assert ranked[2][0] == "candidate3"
    
    def test_explain_score_excellent_match(self, sample_resume, sample_job):
        """Test score explanation for excellent match"""
        engine = ScoringEngine(model_type="tfidf")
        scores = {
            'overall_score': 0.85,
            'skills_score': 0.9,
            'experience_score': 0.8,
            'education_score': 0.8
        }
        
        explanation = engine.explain_score(sample_resume, sample_job, scores)
        
        assert "excellent match" in explanation
        assert "well-aligned" in explanation
        assert str(scores['overall_score']) in explanation
    
    def test_explain_score_poor_match(self, sample_resume, sample_job):
        """Test score explanation for poor match"""
        engine = ScoringEngine(model_type="tfidf")
        scores = {
            'overall_score': 0.2,
            'skills_score': 0.1,
            'experience_score': 0.3,
            'education_score': 0.2
        }
        
        explanation = engine.explain_score(sample_resume, sample_job, scores)
        
        assert "poor match" in explanation
        assert "limited alignment" in explanation
    
    def test_get_model_info(self):
        """Test model info retrieval"""
        engine = ScoringEngine(model_type="tfidf")
        info = engine.get_model_info()
        
        assert info['model_type'] == 'tfidf'
        assert info['version'] == '1.0.0'
        assert 'TF-IDF baseline' in info['description']
        assert 'section_weights' in info
    
    def test_score_bounds(self, sample_resume, sample_job):
        """Test that scores are always in [0, 1] range"""
        engine = ScoringEngine(model_type="tfidf")
        
        result = engine.score_candidate(sample_resume, sample_job)
        
        # Test overall score bounds
        assert 0.0 <= result['overall_score'] <= 1.0
        
        # Test section score bounds
        for score_key in ['skills_score', 'experience_score', 'education_score']:
            assert 0.0 <= result[score_key] <= 1.0
    
    def test_empty_resume_sections(self, sample_job):
        """Test scoring with empty resume sections"""
        empty_resume = ParsedResume(
            raw_text="",
            sections={},
            work_experience=[],
            education=[],
            skills=[],
            certifications=[],
            low_confidence_fields=[],
            file_format="pdf"
        )
        
        engine = ScoringEngine(model_type="tfidf")
        result = engine.score_candidate(empty_resume, sample_job)
        
        # Should handle empty sections gracefully
        assert result['overall_score'] == 0.0
        assert result['skills_score'] == 0.0
        assert result['experience_score'] == 0.0
        assert result['education_score'] == 0.0
    
    def test_error_handling_in_batch_scoring(self, sample_job):
        """Test error handling in batch scoring"""
        engine = ScoringEngine(model_type="tfidf")
        
        # Create a resume that might cause errors
        problematic_resume = ParsedResume(
            raw_text="",
            sections={},
            work_experience=[],
            education=[],
            skills=[],
            certifications=[],
            low_confidence_fields=[],
            file_format="pdf"
        )
        
        # Mock the score_candidate method to raise an exception
        original_method = engine.score_candidate
        def mock_score_candidate(resume, job):
            if resume == problematic_resume:
                raise Exception("Test error")
            return original_method(resume, job)
        
        engine.score_candidate = mock_score_candidate
        
        resumes = [problematic_resume]
        scores = engine.batch_score_candidates(resumes, sample_job)
        
        # Should return zero scores for failed candidates
        assert len(scores) == 1
        assert scores[0]['overall_score'] == 0.0


class TestScoringResult:
    """Unit tests for ScoringResult"""
    
    def test_scoring_result_creation(self):
        """Test ScoringResult creation"""
        candidate_id = str(uuid4())
        job_id = str(uuid4())
        
        result = ScoringResult(
            candidate_id=candidate_id,
            job_id=job_id,
            overall_score=0.75,
            section_scores={'skills': 0.8, 'experience': 0.7},
            explanation="Good match",
            model_version="1.0.0"
        )
        
        assert result.candidate_id == candidate_id
        assert result.job_id == job_id
        assert result.overall_score == 0.75
        assert result.section_scores == {'skills': 0.8, 'experience': 0.7}
        assert result.explanation == "Good match"
        assert result.model_version == "1.0.0"
    
    def test_scoring_result_to_dict(self):
        """Test ScoringResult to_dict conversion"""
        candidate_id = str(uuid4())
        job_id = str(uuid4())
        
        result = ScoringResult(
            candidate_id=candidate_id,
            job_id=job_id,
            overall_score=0.75,
            section_scores={'skills': 0.8},
            explanation="Test explanation"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['candidate_id'] == candidate_id
        assert result_dict['job_id'] == job_id
        assert result_dict['overall_score'] == 0.75
        assert result_dict['section_scores'] == {'skills': 0.8}
        assert result_dict['explanation'] == "Test explanation"
        assert result_dict['model_version'] == "1.0.0"


class TestScoringEngineEdgeCases:
    """Test edge cases for scoring engine"""
    
    def test_identical_resume_and_job(self):
        """Test scoring when resume and job have identical content"""
        engine = ScoringEngine(model_type="tfidf")
        
        # Create resume and job with identical skills
        resume = ParsedResume(
            raw_text="Python Django PostgreSQL",
            sections={"skills": "Python Django PostgreSQL"},
            work_experience=[],
            education=[],
            skills=[
                Skill(skill="Python", confidence=1.0),
                Skill(skill="Django", confidence=1.0),
                Skill(skill="PostgreSQL", confidence=1.0)
            ],
            certifications=[],
            low_confidence_fields=[],
            file_format="pdf"
        )
        
        job = Job(
            id=uuid4(),
            title="Python Developer",
            description="Python Django PostgreSQL development",
            required_skills=["Python", "Django", "PostgreSQL"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.ACTIVE,
            created_by=uuid4()
        )
        
        result = engine.score_candidate(resume, job)
        
        # Should have high skills score due to identical skills
        assert result['skills_score'] > 0.5
    
    def test_completely_different_resume_and_job(self):
        """Test scoring when resume and job have no overlap"""
        engine = ScoringEngine(model_type="tfidf")
        
        resume = ParsedResume(
            raw_text="Marketing Sales Customer Service",
            sections={"skills": "Marketing Sales Customer Service"},
            work_experience=[],
            education=[],
            skills=[
                Skill(skill="Marketing", confidence=1.0),
                Skill(skill="Sales", confidence=1.0)
            ],
            certifications=[],
            low_confidence_fields=[],
            file_format="pdf"
        )
        
        job = Job(
            id=uuid4(),
            title="Software Engineer",
            description="Python programming and software development",
            required_skills=["Python", "Java", "SQL"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.ACTIVE,
            created_by=uuid4()
        )
        
        result = engine.score_candidate(resume, job)
        
        # Should have low overall score due to no overlap
        assert result['overall_score'] < 0.3
    
    def test_special_characters_in_skills(self):
        """Test handling of special characters in skills"""
        engine = ScoringEngine(model_type="tfidf")
        
        resume = ParsedResume(
            raw_text="C++ C# .NET",
            sections={"skills": "C++ C# .NET"},
            work_experience=[],
            education=[],
            skills=[
                Skill(skill="C++", confidence=1.0),
                Skill(skill="C#", confidence=1.0),
                Skill(skill=".NET", confidence=1.0)
            ],
            certifications=[],
            low_confidence_fields=[],
            file_format="pdf"
        )
        
        job = Job(
            id=uuid4(),
            title="C++ Developer",
            description="C++ and .NET development",
            required_skills=["C++", "C#", ".NET"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.ACTIVE,
            created_by=uuid4()
        )
        
        result = engine.score_candidate(resume, job)
        
        # Should handle special characters correctly
        assert result['skills_score'] > 0.0
        assert 0 <= result['overall_score'] <= 1