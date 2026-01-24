"""Property-based tests for scoring service functionality"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from uuid import uuid4
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from backend.app.services.scoring_service import ScoringService
from backend.app.repositories.score_repository import ScoreRepository
from backend.app.repositories.candidate_repository import CandidateRepository
from backend.app.repositories.job_repository import JobRepository
from backend.app.models.score import Score
from backend.app.models.candidate import Candidate
from backend.app.models.job import Job, ExperienceLevel, JobStatus
from backend.app.schemas.resume import ParsedResume, Skill
from ml.inference.scoring_engine import ScoringEngine
from backend.app.core.exceptions import NotFoundException, ValidationException


# Test data strategies
scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
candidate_ids = st.builds(uuid4)
job_ids = st.builds(uuid4)
score_ids = st.builds(uuid4)


class MockScoringEngine:
    """Mock scoring engine for testing"""
    
    def __init__(self, score_value=0.75):
        self.score_value = score_value
    
    def score_candidate(self, resume, job):
        return {
            'overall_score': self.score_value,
            'skills_score': self.score_value,
            'experience_score': self.score_value,
            'education_score': self.score_value
        }
    
    def explain_score(self, resume, job, scores, detailed=False):
        """Generate explanation for score"""
        if detailed:
            return {
                'explanation': f"Detailed explanation for score {scores['overall_score']:.2f}",
                'match_level': 'good' if scores['overall_score'] >= 0.6 else 'moderate',
                'overall_score': scores['overall_score'],
                'section_analysis': [
                    {
                        'section': 'skills',
                        'score': scores['skills_score'],
                        'weight': 0.4,
                        'contribution': scores['skills_score'] * 0.4,
                        'key_matches': ['Python', 'SQL'],
                        'missing_elements': ['Docker']
                    }
                ]
            }
        else:
            return f"Test explanation for score {scores['overall_score']:.2f}"
    
    def generate_detailed_explanation(self, resume, job, scores):
        """Generate detailed explanation"""
        return self.explain_score(resume, job, scores, detailed=True)
    
    def get_model_info(self):
        return {
            'model_type': 'test',
            'version': '1.0.0',
            'description': 'Test scoring engine'
        }


def create_mock_candidate(candidate_id, has_parsed_data=True):
    """Create a mock candidate"""
    parsed_data = None
    if has_parsed_data:
        parsed_data = {
            'raw_text': 'Test candidate',
            'sections': {'skills': 'Python Java'},
            'work_experience': [],
            'education': [],
            'skills': [{'skill': 'Python', 'confidence': 0.8}],
            'certifications': [],
            'low_confidence_fields': [],
            'file_format': 'pdf'
        }
    
    return Candidate(
        id=candidate_id,
        name='Test Candidate',
        email='test@example.com',
        resume_file_path='test.pdf',
        parsed_data=parsed_data,
        skills=['Python', 'Java'],
        experience_years=3,
        education_level='Bachelor'
    )


def create_mock_job(job_id):
    """Create a mock job"""
    return Job(
        id=job_id,
        title='Test Job',
        description='Test job description',
        required_skills=['Python', 'Java'],
        experience_level=ExperienceLevel.MID,
        status=JobStatus.ACTIVE,
        created_by=uuid4()
    )


def create_mock_score(score_id, candidate_id, job_id, score_value=0.75):
    """Create a mock score"""
    return Score(
        id=score_id,
        candidate_id=candidate_id,
        job_id=job_id,
        score=score_value,
        model_version='1.0.0',
        explanation='Test explanation',
        is_current=True,
        created_at=datetime.utcnow()
    )


class TestScoringServiceProperties:
    """Property-based tests for scoring service"""
    
    def create_scoring_service(self, score_value=0.75):
        """Create a scoring service with mocked dependencies"""
        score_repo = Mock(spec=ScoreRepository)
        candidate_repo = Mock(spec=CandidateRepository)
        job_repo = Mock(spec=JobRepository)
        scoring_engine = MockScoringEngine(score_value)
        
        return ScoringService(
            score_repository=score_repo,
            candidate_repository=candidate_repo,
            job_repository=job_repo,
            scoring_engine=scoring_engine
        )
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids,
        score_value=scores
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_18_score_persistence_round_trip(
        self, candidate_id, job_id, score_value
    ):
        """
        **Feature: talentflow-ai, Property 18: Score persistence round-trip**
        
        For any completed scoring operation, the score and timestamp should be 
        persisted to the database and be retrievable using the score ID.
        
        **Validates: Requirements 3.4**
        """
        service = self.create_scoring_service(score_value)
        
        # Mock candidate and job
        candidate = create_mock_candidate(candidate_id)
        job = create_mock_job(job_id)
        
        # Mock repository responses
        service.candidate_repo.get_by_id = AsyncMock(return_value=candidate)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=None)
        
        # Mock score creation
        created_score = create_mock_score(uuid4(), candidate_id, job_id, score_value)
        service.score_repo.create = AsyncMock(return_value=created_score)
        
        # Score candidate
        result_score = await service.score_candidate_for_job(
            candidate_id=candidate_id,
            job_id=job_id
        )
        
        # Verify score was created and can be retrieved
        assert result_score is not None
        assert result_score.candidate_id == candidate_id
        assert result_score.job_id == job_id
        assert result_score.score == score_value
        assert result_score.is_current is True
        assert result_score.created_at is not None
        
        # Verify repository was called to create score
        service.score_repo.create.assert_called_once()
        create_call_args = service.score_repo.create.call_args[0][0]
        assert create_call_args['candidate_id'] == candidate_id
        assert create_call_args['job_id'] == job_id
        assert create_call_args['score'] == score_value
        assert create_call_args['is_current'] is True
    
    @given(
        job_id=job_ids,
        candidate_scores=st.lists(
            st.tuples(candidate_ids, scores),
            min_size=2,
            max_size=10,
            unique_by=lambda x: x[0]  # Unique candidate IDs
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_17_candidate_ranking_order(
        self, job_id, candidate_scores
    ):
        """
        **Feature: talentflow-ai, Property 17: Candidate ranking order**
        
        For any set of candidates scored for a job, when retrieved as a ranked list, 
        the candidates should be ordered in descending order by their scores.
        
        **Validates: Requirements 3.3**
        """
        service = self.create_scoring_service()
        
        # Mock job
        job = create_mock_job(job_id)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        
        # Create mock rankings from candidate scores
        rankings = []
        for i, (candidate_id, score) in enumerate(candidate_scores):
            rankings.append({
                'rank': i + 1,
                'candidate_id': str(candidate_id),
                'candidate_name': f'Candidate {i}',
                'candidate_email': f'candidate{i}@test.com',
                'score': score,
                'explanation': f'Test explanation {i}',
                'created_at': datetime.utcnow()
            })
        
        # Sort rankings by score descending (as the repository should do)
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # Mock repository response
        service.score_repo.get_candidate_rankings_for_job = AsyncMock(return_value=rankings)
        
        # Get ranked candidates
        result = await service.get_ranked_candidates_for_job(job_id=job_id)
        
        # Verify ranking order
        assert len(result) == len(candidate_scores)
        
        for i in range(len(result) - 1):
            current_score = result[i]['score']
            next_score = result[i + 1]['score']
            assert current_score >= next_score, f"Ranking order violated at position {i}"
        
        # Verify all candidates are present
        result_candidate_ids = {r['candidate_id'] for r in result}
        expected_candidate_ids = {str(cid) for cid, _ in candidate_scores}
        assert result_candidate_ids == expected_candidate_ids
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids,
        original_score=scores,
        updated_score=scores
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_20_score_invalidation_on_job_update(
        self, candidate_id, job_id, original_score, updated_score
    ):
        """
        **Feature: talentflow-ai, Property 20: Score invalidation on job update**
        
        For any job description update, all existing scores associated with that job 
        should have their is_current flag set to false.
        
        **Validates: Requirements 3.6**
        """
        service = self.create_scoring_service()
        
        # Mock job
        job = create_mock_job(job_id)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        
        # Mock invalidation
        invalidated_count = 5  # Assume 5 scores were invalidated
        service.score_repo.invalidate_scores_for_job = AsyncMock(return_value=invalidated_count)
        
        # Invalidate scores for job
        result_count = await service.invalidate_scores_for_job(job_id)
        
        # Verify invalidation was called and returned count
        assert result_count == invalidated_count
        service.score_repo.invalidate_scores_for_job.assert_called_once_with(job_id)
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids,
        score_value=scores
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_score_bounds_validation(
        self, candidate_id, job_id, score_value
    ):
        """
        **Property: Score bounds validation**
        
        For any scoring operation, the resulting score should always be in the 
        range [0.0, 1.0] inclusive.
        """
        service = self.create_scoring_service(score_value)
        
        # Mock candidate and job
        candidate = create_mock_candidate(candidate_id)
        job = create_mock_job(job_id)
        
        service.candidate_repo.get_by_id = AsyncMock(return_value=candidate)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=None)
        
        # Mock score creation
        created_score = create_mock_score(uuid4(), candidate_id, job_id, score_value)
        service.score_repo.create = AsyncMock(return_value=created_score)
        
        # Score candidate
        result_score = await service.score_candidate_for_job(
            candidate_id=candidate_id,
            job_id=job_id
        )
        
        # Verify score is in valid range
        assert 0.0 <= result_score.score <= 1.0
        
        # Verify the score matches what the engine produced
        assert result_score.score == score_value
    
    @given(
        candidate_ids_list=st.lists(candidate_ids, min_size=1, max_size=5, unique=True),
        job_id=job_ids,
        score_values=st.lists(scores, min_size=1, max_size=5)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_batch_scoring_consistency(
        self, candidate_ids_list, job_id, score_values
    ):
        """
        **Property: Batch scoring consistency**
        
        For any set of candidates and a job, batch scoring should produce 
        consistent results for each candidate.
        """
        # Ensure we have enough score values
        if len(score_values) < len(candidate_ids_list):
            score_values = score_values * ((len(candidate_ids_list) // len(score_values)) + 1)
        score_values = score_values[:len(candidate_ids_list)]
        
        service = self.create_scoring_service()
        
        # Mock job
        job = create_mock_job(job_id)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        
        # Mock candidates and scores
        mock_scores = []
        for i, (candidate_id, score_value) in enumerate(zip(candidate_ids_list, score_values)):
            candidate = create_mock_candidate(candidate_id)
            service.candidate_repo.get_by_id = AsyncMock(return_value=candidate)
            
            score = create_mock_score(uuid4(), candidate_id, job_id, score_value)
            mock_scores.append(score)
        
        # Mock repository calls
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=None)
        service.score_repo.create = AsyncMock(side_effect=mock_scores)
        
        # Batch score candidates
        result_scores = await service.batch_score_candidates_for_job(
            candidate_ids=candidate_ids_list,
            job_id=job_id
        )
        
        # Verify all candidates were scored
        assert len(result_scores) == len(candidate_ids_list)
        
        # Verify each score is valid
        for score in result_scores:
            assert 0.0 <= score.score <= 1.0
            assert score.job_id == job_id
            assert score.candidate_id in candidate_ids_list
            assert score.is_current is True
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_not_found_error_handling(
        self, candidate_id, job_id
    ):
        """
        **Property: Not found error handling**
        
        For any scoring request with non-existent candidate or job, 
        the system should raise NotFoundException.
        """
        service = self.create_scoring_service()
        
        # Test with non-existent candidate
        service.candidate_repo.get_by_id = AsyncMock(return_value=None)
        service.job_repo.get_by_id = AsyncMock(return_value=create_mock_job(job_id))
        
        with pytest.raises(NotFoundException, match="Candidate not found"):
            await service.score_candidate_for_job(candidate_id, job_id)
        
        # Test with non-existent job
        service.candidate_repo.get_by_id = AsyncMock(return_value=create_mock_candidate(candidate_id))
        service.job_repo.get_by_id = AsyncMock(return_value=None)
        
        with pytest.raises(NotFoundException, match="Job not found"):
            await service.score_candidate_for_job(candidate_id, job_id)
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_validation_error_handling(
        self, candidate_id, job_id
    ):
        """
        **Property: Validation error handling**
        
        For any scoring request with candidate lacking parsed data, 
        the system should raise ValidationException.
        """
        service = self.create_scoring_service()
        
        # Mock candidate without parsed data
        candidate = create_mock_candidate(candidate_id, has_parsed_data=False)
        job = create_mock_job(job_id)
        
        service.candidate_repo.get_by_id = AsyncMock(return_value=candidate)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=None)
        
        with pytest.raises(ValidationException, match="has no parsed resume data"):
            await service.score_candidate_for_job(candidate_id, job_id)
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids,
        score_value=scores
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_existing_score_reuse(
        self, candidate_id, job_id, score_value
    ):
        """
        **Property: Existing score reuse**
        
        For any scoring request where a current score already exists, 
        the system should return the existing score without recomputing.
        """
        service = self.create_scoring_service()
        
        # Mock existing score
        existing_score = create_mock_score(uuid4(), candidate_id, job_id, score_value)
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=existing_score)
        
        # Score candidate (should return existing score)
        result_score = await service.score_candidate_for_job(
            candidate_id=candidate_id,
            job_id=job_id,
            force_rescore=False
        )
        
        # Verify existing score was returned
        assert result_score == existing_score
        assert result_score.score == score_value
        
        # Verify no new score was created
        service.score_repo.create.assert_not_called()
    
    @given(
        job_id=job_ids,
        min_score=st.one_of(st.none(), scores),
        limit=st.integers(min_value=1, max_value=100)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_ranking_filters_and_limits(
        self, job_id, min_score, limit
    ):
        """
        **Property: Ranking filters and limits**
        
        For any ranking request with filters and limits, the system should 
        respect the minimum score threshold and limit parameters.
        """
        service = self.create_scoring_service()
        
        # Mock job
        job = create_mock_job(job_id)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        
        # Create mock rankings with various scores
        all_rankings = []
        for i in range(limit + 10):  # More than limit
            score = 0.1 + (i * 0.08)  # Scores from 0.1 to ~0.9
            all_rankings.append({
                'rank': i + 1,
                'candidate_id': str(uuid4()),
                'candidate_name': f'Candidate {i}',
                'candidate_email': f'candidate{i}@test.com',
                'score': min(score, 1.0),
                'explanation': f'Test explanation {i}',
                'created_at': datetime.utcnow()
            })
        
        # Filter by min_score if specified
        if min_score is not None:
            filtered_rankings = [r for r in all_rankings if r['score'] >= min_score]
        else:
            filtered_rankings = all_rankings
        
        # Apply limit
        limited_rankings = filtered_rankings[:limit]
        
        # Mock repository response
        service.score_repo.get_candidate_rankings_for_job = AsyncMock(return_value=limited_rankings)
        
        # Get ranked candidates
        result = await service.get_ranked_candidates_for_job(
            job_id=job_id,
            min_score=min_score,
            limit=limit
        )
        
        # Verify results respect filters and limits
        assert len(result) <= limit
        
        if min_score is not None:
            for ranking in result:
                assert ranking['score'] >= min_score
        
        # Verify ordering is maintained
        for i in range(len(result) - 1):
            assert result[i]['score'] >= result[i + 1]['score']
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids,
        score_value=scores
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_19_score_explanation_generation(
        self, candidate_id, job_id, score_value
    ):
        """
        **Feature: talentflow-ai, Property 19: Score explanation generation**
        
        For any scoring request with explanation enabled, the system should generate 
        a non-empty natural language explanation for the scoring decision.
        
        **Validates: Requirements 3.5**
        """
        service = self.create_scoring_service(score_value)
        
        # Mock candidate and job
        candidate = create_mock_candidate(candidate_id)
        job = create_mock_job(job_id)
        
        service.candidate_repo.get_by_id = AsyncMock(return_value=candidate)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=None)
        
        # Mock score creation with explanation
        created_score = create_mock_score(uuid4(), candidate_id, job_id, score_value)
        created_score.explanation = f"Test explanation for score {score_value:.2f}"
        service.score_repo.create = AsyncMock(return_value=created_score)
        
        # Score candidate with explanation enabled
        result_score = await service.score_candidate_for_job(
            candidate_id=candidate_id,
            job_id=job_id,
            generate_explanation=True
        )
        
        # Verify explanation was generated
        assert result_score.explanation is not None
        assert isinstance(result_score.explanation, str)
        assert len(result_score.explanation.strip()) > 0
        
        # Verify explanation contains score information
        score_str = f"{score_value:.2f}"
        assert score_str in result_score.explanation or str(score_value) in result_score.explanation
        
        # Verify explanation is meaningful (contains common explanation terms)
        explanation_lower = result_score.explanation.lower()
        meaningful_terms = [
            'match', 'score', 'candidate', 'skills', 'experience', 
            'education', 'excellent', 'good', 'moderate', 'poor',
            'strong', 'weak', 'relevant', 'suitable'
        ]
        
        # At least one meaningful term should be present
        assert any(term in explanation_lower for term in meaningful_terms), \
            f"Explanation lacks meaningful content: {result_score.explanation}"
        
        # Verify repository was called to create score with explanation
        service.score_repo.create.assert_called_once()
        create_call_args = service.score_repo.create.call_args[0][0]
        assert create_call_args['explanation'] is not None
        assert len(create_call_args['explanation'].strip()) > 0
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids,
        score_value=scores
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_19_detailed_explanation_generation(
        self, candidate_id, job_id, score_value
    ):
        """
        **Feature: talentflow-ai, Property 19: Score explanation generation (detailed)**
        
        For any scoring request with detailed explanation enabled, the system should 
        generate a comprehensive explanation with section analysis and improvement suggestions.
        
        **Validates: Requirements 3.5**
        """
        service = self.create_scoring_service(score_value)
        
        # Mock candidate and job
        candidate = create_mock_candidate(candidate_id)
        job = create_mock_job(job_id)
        
        service.candidate_repo.get_by_id = AsyncMock(return_value=candidate)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        
        # Mock existing score or create new one
        existing_score = create_mock_score(uuid4(), candidate_id, job_id, score_value)
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=existing_score)
        
        # Generate detailed explanation
        detailed_explanation = await service.generate_detailed_explanation(
            candidate_id=candidate_id,
            job_id=job_id
        )
        
        # Verify detailed explanation structure
        assert isinstance(detailed_explanation, dict)
        
        # Required fields for detailed explanation
        required_fields = ['explanation', 'match_level', 'overall_score']
        for field in required_fields:
            assert field in detailed_explanation
            assert detailed_explanation[field] is not None
        
        # Verify explanation is non-empty
        assert isinstance(detailed_explanation['explanation'], str)
        assert len(detailed_explanation['explanation'].strip()) > 0
        
        # Verify match level is valid
        valid_match_levels = ['excellent', 'good', 'moderate', 'poor']
        assert detailed_explanation['match_level'] in valid_match_levels
        
        # Verify overall score matches
        assert detailed_explanation['overall_score'] == score_value
        
        # Verify metadata fields are present
        metadata_fields = ['score_id', 'candidate_id', 'job_id', 'candidate_name', 'job_title']
        for field in metadata_fields:
            assert field in detailed_explanation
            assert detailed_explanation[field] is not None
        
        # If section analysis is present, verify its structure
        if 'section_analysis' in detailed_explanation:
            section_analysis = detailed_explanation['section_analysis']
            assert isinstance(section_analysis, list)
            
            for section in section_analysis:
                assert isinstance(section, dict)
                section_fields = ['section', 'score', 'weight', 'contribution']
                for field in section_fields:
                    assert field in section
                    assert isinstance(section[field], (int, float, str))
                
                # Verify score bounds for sections
                if isinstance(section['score'], (int, float)):
                    assert 0.0 <= section['score'] <= 1.0
                if isinstance(section['weight'], (int, float)):
                    assert 0.0 <= section['weight'] <= 1.0
                if isinstance(section['contribution'], (int, float)):
                    assert 0.0 <= section['contribution'] <= 1.0
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids,
        score_value=scores
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_19_explanation_without_flag(
        self, candidate_id, job_id, score_value
    ):
        """
        **Feature: talentflow-ai, Property 19: Score explanation generation (negative case)**
        
        For any scoring request without explanation enabled, the system should NOT 
        generate an explanation (explanation should be None).
        
        **Validates: Requirements 3.5**
        """
        service = self.create_scoring_service(score_value)
        
        # Mock candidate and job
        candidate = create_mock_candidate(candidate_id)
        job = create_mock_job(job_id)
        
        service.candidate_repo.get_by_id = AsyncMock(return_value=candidate)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=None)
        
        # Mock score creation without explanation
        created_score = create_mock_score(uuid4(), candidate_id, job_id, score_value)
        created_score.explanation = None  # No explanation when flag is False
        service.score_repo.create = AsyncMock(return_value=created_score)
        
        # Score candidate without explanation enabled (default)
        result_score = await service.score_candidate_for_job(
            candidate_id=candidate_id,
            job_id=job_id,
            generate_explanation=False
        )
        
        # Verify no explanation was generated
        assert result_score.explanation is None
        
        # Verify repository was called to create score without explanation
        service.score_repo.create.assert_called_once()
        create_call_args = service.score_repo.create.call_args[0][0]
        assert create_call_args['explanation'] is None
    
    @given(
        candidate_ids_list=st.lists(candidate_ids, min_size=2, max_size=5, unique=True),
        job_id=job_ids,
        score_values=st.lists(scores, min_size=2, max_size=5)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_19_batch_explanation_generation(
        self, candidate_ids_list, job_id, score_values
    ):
        """
        **Feature: talentflow-ai, Property 19: Score explanation generation (batch)**
        
        For any batch scoring request with explanations enabled, the system should 
        generate explanations for all candidates in the batch.
        
        **Validates: Requirements 3.5**
        """
        # Ensure we have enough score values
        if len(score_values) < len(candidate_ids_list):
            score_values = score_values * ((len(candidate_ids_list) // len(score_values)) + 1)
        score_values = score_values[:len(candidate_ids_list)]
        
        service = self.create_scoring_service()
        
        # Mock job
        job = create_mock_job(job_id)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        
        # Mock candidates and scores with explanations
        mock_scores = []
        for i, (candidate_id, score_value) in enumerate(zip(candidate_ids_list, score_values)):
            candidate = create_mock_candidate(candidate_id)
            service.candidate_repo.get_by_id = AsyncMock(return_value=candidate)
            
            score = create_mock_score(uuid4(), candidate_id, job_id, score_value)
            score.explanation = f"Batch explanation for candidate {i} with score {score_value:.2f}"
            mock_scores.append(score)
        
        # Mock repository calls
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=None)
        service.score_repo.create = AsyncMock(side_effect=mock_scores)
        
        # Batch score candidates with explanations
        result_scores = await service.batch_score_candidates_for_job(
            candidate_ids=candidate_ids_list,
            job_id=job_id,
            generate_explanations=True
        )
        
        # Verify all candidates have explanations
        assert len(result_scores) == len(candidate_ids_list)
        
        for i, score in enumerate(result_scores):
            assert score.explanation is not None
            assert isinstance(score.explanation, str)
            assert len(score.explanation.strip()) > 0
            
            # Verify explanation contains score information
            score_str = f"{score_values[i]:.2f}"
            assert (score_str in score.explanation or 
                   str(score_values[i]) in score.explanation or
                   f"candidate {i}" in score.explanation.lower())
    
    @given(
        candidate_id=candidate_ids,
        job_id=job_ids,
        score_value=scores
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_19_explanation_consistency(
        self, candidate_id, job_id, score_value
    ):
        """
        **Feature: talentflow-ai, Property 19: Score explanation generation (consistency)**
        
        For any candidate-job pair, generating explanations multiple times should 
        produce consistent explanations (deterministic behavior).
        
        **Validates: Requirements 3.5**
        """
        service = self.create_scoring_service(score_value)
        
        # Mock candidate and job
        candidate = create_mock_candidate(candidate_id)
        job = create_mock_job(job_id)
        
        service.candidate_repo.get_by_id = AsyncMock(return_value=candidate)
        service.job_repo.get_by_id = AsyncMock(return_value=job)
        
        # Mock existing score
        existing_score = create_mock_score(uuid4(), candidate_id, job_id, score_value)
        service.score_repo.get_by_candidate_and_job = AsyncMock(return_value=existing_score)
        
        # Generate detailed explanation twice
        explanation1 = await service.generate_detailed_explanation(
            candidate_id=candidate_id,
            job_id=job_id
        )
        
        explanation2 = await service.generate_detailed_explanation(
            candidate_id=candidate_id,
            job_id=job_id
        )
        
        # Explanations should be consistent
        assert explanation1['explanation'] == explanation2['explanation']
        assert explanation1['match_level'] == explanation2['match_level']
        assert explanation1['overall_score'] == explanation2['overall_score']
        
        # If section analysis is present, it should be identical
        if 'section_analysis' in explanation1 and 'section_analysis' in explanation2:
            assert explanation1['section_analysis'] == explanation2['section_analysis']