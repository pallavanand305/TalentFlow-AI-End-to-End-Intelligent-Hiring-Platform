"""Unit tests for semantic scoring functionality"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from ml.inference.scoring_engine import ScoringEngine
from backend.app.schemas.resume import ParsedResume, WorkExperience, Education, Skill
from backend.app.models.job import Job, ExperienceLevel, JobStatus
from uuid import uuid4


class TestSemanticScoring:
    """Unit tests for semantic scoring functionality"""
    
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
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_engine_initialization_success(self, mock_transformer_class):
        """Test successful semantic engine initialization"""
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
        engine = ScoringEngine(model_type="semantic")
        
        assert engine.model_type == "semantic"
        assert engine.semantic_model == mock_model
        mock_transformer_class.assert_called_once_with('all-MiniLM-L6-v2')
    
    def test_semantic_engine_fallback_to_tfidf(self):
        """Test fallback to TF-IDF when sentence-transformers is not available"""
        # This will naturally fail to import sentence_transformers in test environment
        # and should fall back to TF-IDF
        engine = ScoringEngine(model_type="semantic")
        
        # Should fall back to TF-IDF
        assert engine.model_type == "tfidf"
        assert engine.tfidf_vectorizer is not None
        assert engine.semantic_model is None
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_score_computation(self, mock_transformer_class, sample_resume, sample_job):
        """Test semantic score computation"""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
        # Mock embeddings - simulate high similarity
        mock_embedding = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_model.encode.return_value = mock_embedding
        
        engine = ScoringEngine(model_type="semantic")
        
        # Mock the cosine similarity computation
        with patch('ml.inference.scoring_engine.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.85]]  # High similarity
            
            result = engine.score_candidate(sample_resume, sample_job)
            
            assert 'overall_score' in result
            assert 'skills_score' in result
            assert 'experience_score' in result
            assert 'education_score' in result
            
            # All scores should be in valid range
            assert 0 <= result['overall_score'] <= 1
            assert all(0 <= score <= 1 for score in result.values())
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_embeddings_generation(self, mock_transformer_class, sample_resume, sample_job):
        """Test semantic embeddings generation"""
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
        # Mock different embeddings for different sections
        def mock_encode(texts):
            # Return different embeddings based on input
            if isinstance(texts, list) and len(texts) == 1:
                text = texts[0]
                if "Python" in text:
                    return np.array([[0.8, 0.2, 0.1]])
                elif "Software Engineer" in text:
                    return np.array([[0.6, 0.3, 0.2]])
                else:
                    return np.array([[0.3, 0.4, 0.5]])
            return np.array([[0.5, 0.5, 0.5]])
        
        mock_model.encode.side_effect = mock_encode
        
        engine = ScoringEngine(model_type="semantic")
        
        with patch('ml.inference.scoring_engine.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.75]]
            
            result = engine.score_candidate(sample_resume, sample_job)
            
            # Verify encode was called for different sections
            assert mock_model.encode.call_count >= 6  # 3 sections * 2 (resume + job)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert len(result) == 4  # overall + 3 section scores
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_vs_tfidf_consistency(self, mock_transformer_class, sample_resume, sample_job):
        """Test that semantic and TF-IDF engines return consistent result structure"""
        # Test TF-IDF engine
        tfidf_engine = ScoringEngine(model_type="tfidf")
        tfidf_result = tfidf_engine.score_candidate(sample_resume, sample_job)
        
        # Test semantic engine (mocked)
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        semantic_engine = ScoringEngine(model_type="semantic")
        
        with patch('ml.inference.scoring_engine.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.7]]
            semantic_result = semantic_engine.score_candidate(sample_resume, sample_job)
        
        # Both should have same structure
        assert set(tfidf_result.keys()) == set(semantic_result.keys())
        assert all(0 <= score <= 1 for score in tfidf_result.values())
        assert all(0 <= score <= 1 for score in semantic_result.values())
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_batch_scoring(self, mock_transformer_class, sample_resume, sample_job):
        """Test semantic batch scoring"""
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        engine = ScoringEngine(model_type="semantic")
        resumes = [sample_resume, sample_resume]  # Duplicate for testing
        
        with patch('ml.inference.scoring_engine.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.8]]
            
            scores = engine.batch_score_candidates(resumes, sample_job)
            
            assert len(scores) == 2
            assert all('overall_score' in score for score in scores)
            assert all(0 <= score['overall_score'] <= 1 for score in scores)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_error_handling(self, mock_transformer_class, sample_resume, sample_job):
        """Test error handling in semantic scoring"""
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
        # Mock encode to raise an exception
        mock_model.encode.side_effect = Exception("Encoding failed")
        
        engine = ScoringEngine(model_type="semantic")
        
        # Should handle errors gracefully and return zero scores
        result = engine.score_candidate(sample_resume, sample_job)
        
        # Should still return valid structure with zero scores
        assert 'overall_score' in result
        assert result['skills_score'] == 0.0
        assert result['experience_score'] == 0.0
        assert result['education_score'] == 0.0
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_empty_sections_handling(self, mock_transformer_class, sample_job):
        """Test semantic scoring with empty resume sections"""
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
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
        
        engine = ScoringEngine(model_type="semantic")
        result = engine.score_candidate(empty_resume, sample_job)
        
        # Should handle empty sections gracefully
        assert result['overall_score'] == 0.0
        assert result['skills_score'] == 0.0
        assert result['experience_score'] == 0.0
        assert result['education_score'] == 0.0
    
    def test_model_info_semantic(self):
        """Test model info for semantic engine"""
        # This will fall back to TF-IDF in test environment
        engine = ScoringEngine(model_type="semantic")
        info = engine.get_model_info()
        
        assert 'model_type' in info
        assert 'version' in info
        assert 'description' in info
        assert info['version'] == '1.0.0'
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_model_loading(self, mock_transformer_class):
        """Test that semantic model is loaded with correct parameters"""
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
        engine = ScoringEngine(model_type="semantic")
        
        # Verify the model was loaded with correct parameters
        mock_transformer_class.assert_called_once_with('all-MiniLM-L6-v2')
        assert engine.semantic_model == mock_model
        assert engine.model_type == "semantic"
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_semantic_section_weights_applied(self, mock_transformer_class, sample_resume, sample_job):
        """Test that section weights are properly applied in semantic scoring"""
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        engine = ScoringEngine(model_type="semantic")
        
        # Mock different similarities for different sections
        similarity_values = [0.9, 0.6, 0.3]  # skills, experience, education
        similarity_index = 0
        
        def mock_cosine_similarity(a, b):
            nonlocal similarity_index
            result = [[similarity_values[similarity_index % len(similarity_values)]]]
            similarity_index += 1
            return result
        
        with patch('ml.inference.scoring_engine.cosine_similarity', side_effect=mock_cosine_similarity):
            result = engine.score_candidate(sample_resume, sample_job)
            
            # Calculate expected weighted score
            expected = (0.9 * 0.4 + 0.6 * 0.4 + 0.3 * 0.2)  # weights: skills=0.4, exp=0.4, edu=0.2
            
            # Allow for small floating point differences
            assert abs(result['overall_score'] - expected) < 0.01