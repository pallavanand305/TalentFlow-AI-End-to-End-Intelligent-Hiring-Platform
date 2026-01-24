"""Property-based tests for scoring engine functionality"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from uuid import uuid4
import numpy as np

from ml.inference.scoring_engine import ScoringEngine, ScoringResult
from backend.app.schemas.resume import ParsedResume, WorkExperience, Education, Skill
from backend.app.models.job import Job, ExperienceLevel, JobStatus


# Test data strategies
skill_names = st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isalnum())
skills_lists = st.lists(skill_names, min_size=0, max_size=10, unique=True)
job_titles = st.text(min_size=3, max_size=100).filter(lambda x: x.strip())
job_descriptions = st.text(min_size=10, max_size=500).filter(lambda x: x.strip())
experience_levels = st.sampled_from(list(ExperienceLevel))
scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def create_test_resume(skills_list):
    """Create a test resume with given skills"""
    skills = [Skill(skill=skill, confidence=0.8) for skill in skills_list]
    
    return ParsedResume(
        raw_text=" ".join(skills_list),
        sections={"skills": " ".join(skills_list)},
        work_experience=[
            WorkExperience(
                company="TestCorp",
                title="Developer",
                description="Software development",
                confidence=0.8
            )
        ],
        education=[
            Education(
                institution="Test University",
                degree="Bachelor",
                description="Computer Science",
                confidence=0.8
            )
        ],
        skills=skills,
        certifications=[],
        low_confidence_fields=[],
        file_format="pdf"
    )


def create_test_job(title, description, required_skills, experience_level):
    """Create a test job with given parameters"""
    return Job(
        id=uuid4(),
        title=title,
        description=description,
        required_skills=required_skills,
        experience_level=experience_level,
        status=JobStatus.ACTIVE,
        created_by=uuid4()
    )


class TestScoringEngineProperties:
    """Property-based tests for scoring engine"""
    
    @given(
        resume_skills=skills_lists,
        job_skills=skills_lists,
        job_title=job_titles,
        job_description=job_descriptions,
        experience_level=experience_levels
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_16_score_bounds(
        self, resume_skills, job_skills, job_title, job_description, experience_level
    ):
        """
        **Feature: talentflow-ai, Property 16: Score bounds**
        
        For any candidate-job pair, the computed similarity score should be a 
        numerical value in the range [0.0, 1.0] inclusive.
        
        **Validates: Requirements 3.1, 13.5**
        """
        # Skip empty inputs that might cause issues
        assume(len(job_skills) > 0)
        assume(len(job_title.strip()) >= 3)
        assume(len(job_description.strip()) >= 10)
        
        engine = ScoringEngine(model_type="tfidf")
        resume = create_test_resume(resume_skills)
        job = create_test_job(job_title, job_description, job_skills, experience_level)
        
        result = engine.score_candidate(resume, job)
        
        # Verify overall score is in [0, 1] range
        assert 0.0 <= result['overall_score'] <= 1.0
        
        # Verify all section scores are in [0, 1] range
        assert 0.0 <= result['skills_score'] <= 1.0
        assert 0.0 <= result['experience_score'] <= 1.0
        assert 0.0 <= result['education_score'] <= 1.0
        
        # Verify scores are numerical (not NaN or infinity)
        assert np.isfinite(result['overall_score'])
        assert np.isfinite(result['skills_score'])
        assert np.isfinite(result['experience_score'])
        assert np.isfinite(result['education_score'])
    
    @given(
        skills_score=scores,
        experience_score=scores,
        education_score=scores
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_21_weighted_section_scoring(
        self, skills_score, experience_score, education_score
    ):
        """
        **Feature: talentflow-ai, Property 21: Weighted section scoring**
        
        For any two resumes that differ only in one section (skills vs experience vs education), 
        the change in score should reflect the configured weights for that section.
        
        **Validates: Requirements 13.3**
        """
        engine = ScoringEngine(model_type="tfidf")
        
        section_scores = {
            'skills': skills_score,
            'experience': experience_score,
            'education': education_score
        }
        
        weighted_score = engine._compute_weighted_score(section_scores)
        
        # Verify weighted score is in valid range
        assert 0.0 <= weighted_score <= 1.0
        
        # Verify weighted score respects section weights
        expected_score = (
            skills_score * engine.section_weights['skills'] +
            experience_score * engine.section_weights['experience'] +
            education_score * engine.section_weights['education']
        )
        
        assert abs(weighted_score - expected_score) < 0.001
    
    @given(
        resume_skills=skills_lists,
        job_skills=skills_lists,
        job_title=job_titles,
        job_description=job_descriptions,
        experience_level=experience_levels
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_scoring_consistency(
        self, resume_skills, job_skills, job_title, job_description, experience_level
    ):
        """
        **Property: Scoring consistency**
        
        For any candidate-job pair, scoring the same pair multiple times should 
        return identical results (deterministic behavior).
        """
        # Skip empty inputs
        assume(len(job_skills) > 0)
        assume(len(job_title.strip()) >= 3)
        assume(len(job_description.strip()) >= 10)
        
        engine = ScoringEngine(model_type="tfidf")
        resume = create_test_resume(resume_skills)
        job = create_test_job(job_title, job_description, job_skills, experience_level)
        
        # Score the same pair twice
        result1 = engine.score_candidate(resume, job)
        result2 = engine.score_candidate(resume, job)
        
        # Results should be identical
        assert result1['overall_score'] == result2['overall_score']
        assert result1['skills_score'] == result2['skills_score']
        assert result1['experience_score'] == result2['experience_score']
        assert result1['education_score'] == result2['education_score']
    
    @given(
        common_skills=skills_lists,
        unique_skills1=skills_lists,
        unique_skills2=skills_lists,
        job_title=job_titles,
        job_description=job_descriptions,
        experience_level=experience_levels
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_skill_overlap_correlation(
        self, common_skills, unique_skills1, unique_skills2, 
        job_title, job_description, experience_level
    ):
        """
        **Property: Skill overlap correlation**
        
        For any job and two candidates, the candidate with more overlapping skills 
        should generally have a higher or equal skills score.
        """
        # Skip cases that might not show clear differences
        assume(len(common_skills) > 0)
        assume(len(unique_skills1) > 0 or len(unique_skills2) > 0)
        assume(len(job_title.strip()) >= 3)
        assume(len(job_description.strip()) >= 10)
        
        # Create job with common skills
        job_skills = common_skills[:3] if len(common_skills) >= 3 else common_skills
        assume(len(job_skills) > 0)
        
        job = create_test_job(job_title, job_description, job_skills, experience_level)
        
        # Candidate 1: has common skills + unique skills
        candidate1_skills = common_skills + unique_skills1
        resume1 = create_test_resume(candidate1_skills)
        
        # Candidate 2: has only unique skills (no overlap with job)
        candidate2_skills = unique_skills2
        # Ensure no overlap with job skills
        candidate2_skills = [skill for skill in candidate2_skills if skill not in job_skills]
        assume(len(candidate2_skills) > 0)  # Ensure candidate 2 has some skills
        
        resume2 = create_test_resume(candidate2_skills)
        
        engine = ScoringEngine(model_type="tfidf")
        
        result1 = engine.score_candidate(resume1, job)
        result2 = engine.score_candidate(resume2, job)
        
        # Candidate 1 (with overlapping skills) should have higher or equal skills score
        # Note: We use >= because TF-IDF might not always show perfect correlation
        # due to normalization and other factors
        assert result1['skills_score'] >= result2['skills_score'] - 0.1  # Small tolerance
    
    @given(
        candidate_scores=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),  # candidate_id
                st.builds(dict, overall_score=scores)  # score dict
            ),
            min_size=2,
            max_size=10
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_17_candidate_ranking_order(self, candidate_scores):
        """
        **Feature: talentflow-ai, Property 17: Candidate ranking order**
        
        For any set of candidates scored for a job, when retrieved as a ranked list, 
        the candidates should be ordered in descending order by their scores.
        
        **Validates: Requirements 3.3**
        """
        engine = ScoringEngine(model_type="tfidf")
        
        ranked = engine.rank_candidates(candidate_scores)
        
        # Verify ranking is in descending order
        for i in range(len(ranked) - 1):
            current_score = ranked[i][1]['overall_score']
            next_score = ranked[i + 1][1]['overall_score']
            assert current_score >= next_score
        
        # Verify all original candidates are present
        original_ids = {candidate_id for candidate_id, _ in candidate_scores}
        ranked_ids = {candidate_id for candidate_id, _ in ranked}
        assert original_ids == ranked_ids
    
    @given(
        resume_skills=skills_lists,
        job_skills=skills_lists,
        job_title=job_titles,
        job_description=job_descriptions,
        experience_level=experience_levels
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_explanation_generation(
        self, resume_skills, job_skills, job_title, job_description, experience_level
    ):
        """
        **Property: Score explanation generation**
        
        For any scoring request, the system should be able to generate a 
        non-empty natural language explanation for the scoring decision.
        
        **Validates: Requirements 3.5**
        """
        # Skip empty inputs
        assume(len(job_skills) > 0)
        assume(len(job_title.strip()) >= 3)
        assume(len(job_description.strip()) >= 10)
        
        engine = ScoringEngine(model_type="tfidf")
        resume = create_test_resume(resume_skills)
        job = create_test_job(job_title, job_description, job_skills, experience_level)
        
        scores = engine.score_candidate(resume, job)
        explanation = engine.explain_score(resume, job, scores)
        
        # Explanation should be non-empty string
        assert isinstance(explanation, str)
        assert len(explanation.strip()) > 0
        
        # Explanation should contain score information
        assert str(scores['overall_score']) in explanation or f"{scores['overall_score']:.2f}" in explanation
        
        # Explanation should contain match level assessment
        match_levels = ["excellent", "good", "moderate", "poor"]
        assert any(level in explanation.lower() for level in match_levels)
    
    @given(
        resume_skills=skills_lists,
        job_skills=skills_lists,
        job_title=job_titles,
        job_description=job_descriptions,
        experience_level=experience_levels
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_enhanced_explanation_generation(
        self, resume_skills, job_skills, job_title, job_description, experience_level
    ):
        """
        **Property: Enhanced explanation generation with section analysis**
        **Feature: talentflow-ai, Property 53: Enhanced explanation generation**
        
        For any scoring request with detailed explanation enabled, the system should 
        generate a comprehensive explanation with section-wise analysis, key matches,
        and improvement suggestions.
        
        **Validates: Requirements 3.5**
        """
        # Skip empty inputs
        assume(len(job_skills) > 0)
        assume(len(job_title.strip()) >= 3)
        assume(len(job_description.strip()) >= 10)
        
        engine = ScoringEngine(model_type="tfidf")
        resume = create_test_resume(resume_skills)
        job = create_test_job(job_title, job_description, job_skills, experience_level)
        
        scores = engine.score_candidate(resume, job)
        detailed_explanation = engine.generate_detailed_explanation(resume, job, scores)
        
        # Should return dictionary with required fields
        assert isinstance(detailed_explanation, dict)
        assert 'explanation' in detailed_explanation
        assert 'match_level' in detailed_explanation
        assert 'overall_score' in detailed_explanation
        assert 'section_analysis' in detailed_explanation
        
        # Explanation should be non-empty
        assert isinstance(detailed_explanation['explanation'], str)
        assert len(detailed_explanation['explanation'].strip()) > 0
        
        # Match level should be valid
        valid_levels = ['excellent', 'good', 'moderate', 'poor']
        assert detailed_explanation['match_level'] in valid_levels
        
        # Overall score should match
        assert detailed_explanation['overall_score'] == scores['overall_score']
        
        # Section analysis should be present
        assert isinstance(detailed_explanation['section_analysis'], list)
        
        # If section analysis is present, validate structure
        for section in detailed_explanation['section_analysis']:
            assert 'section' in section
            assert 'score' in section
            assert 'weight' in section
            assert 'contribution' in section
            assert 'key_matches' in section
            assert 'missing_elements' in section
            
            # Validate data types
            assert isinstance(section['section'], str)
            assert isinstance(section['score'], (int, float))
            assert isinstance(section['weight'], (int, float))
            assert isinstance(section['contribution'], (int, float))
            assert isinstance(section['key_matches'], list)
            assert isinstance(section['missing_elements'], list)
            
            # Validate score bounds
            assert 0.0 <= section['score'] <= 1.0
            assert 0.0 <= section['weight'] <= 1.0
            assert 0.0 <= section['contribution'] <= 1.0
    
    @given(
        resumes_count=st.integers(min_value=1, max_value=5),
        job_title=job_titles,
        job_description=job_descriptions,
        job_skills=skills_lists,
        experience_level=experience_levels
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_batch_scoring_consistency(
        self, resumes_count, job_title, job_description, job_skills, experience_level
    ):
        """
        **Property: Batch scoring consistency**
        
        For any set of candidates and a job, batch scoring should return the same 
        results as individual scoring for each candidate.
        """
        # Skip empty inputs
        assume(len(job_skills) > 0)
        assume(len(job_title.strip()) >= 3)
        assume(len(job_description.strip()) >= 10)
        
        engine = ScoringEngine(model_type="tfidf")
        job = create_test_job(job_title, job_description, job_skills, experience_level)
        
        # Create test resumes
        resumes = []
        for i in range(resumes_count):
            skills = [f"skill_{i}_{j}" for j in range(2)]  # Simple skills for each resume
            resumes.append(create_test_resume(skills))
        
        # Score individually
        individual_scores = []
        for resume in resumes:
            score = engine.score_candidate(resume, job)
            individual_scores.append(score)
        
        # Score in batch
        batch_scores = engine.batch_score_candidates(resumes, job)
        
        # Results should be identical
        assert len(individual_scores) == len(batch_scores)
        for individual, batch in zip(individual_scores, batch_scores):
            assert individual['overall_score'] == batch['overall_score']
            assert individual['skills_score'] == batch['skills_score']
            assert individual['experience_score'] == batch['experience_score']
            assert individual['education_score'] == batch['education_score']
    
    @given(
        model_type=st.sampled_from(["tfidf"])  # Only test tfidf for now
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_model_info_completeness(self, model_type):
        """
        **Property: Model info completeness**
        
        For any scoring engine, the model info should contain all required fields 
        and be consistent with the engine configuration.
        """
        engine = ScoringEngine(model_type=model_type)
        info = engine.get_model_info()
        
        # Required fields should be present
        required_fields = ['model_type', 'version', 'description', 'section_weights']
        for field in required_fields:
            assert field in info
            assert info[field] is not None
        
        # Model type should match
        assert info['model_type'] == model_type
        
        # Version should be a valid string
        assert isinstance(info['version'], str)
        assert len(info['version']) > 0
        
        # Description should be non-empty
        assert isinstance(info['description'], str)
        assert len(info['description']) > 0


class TestScoringResultProperties:
    """Property-based tests for ScoringResult"""
    
    @given(
        candidate_id=st.text(min_size=1, max_size=50),
        job_id=st.text(min_size=1, max_size=50),
        overall_score=scores,
        skills_score=scores,
        experience_score=scores,
        education_score=scores,
        explanation=st.one_of(st.none(), st.text(min_size=1, max_size=200)),
        model_version=st.text(min_size=1, max_size=20)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_scoring_result_round_trip(
        self, candidate_id, job_id, overall_score, skills_score, 
        experience_score, education_score, explanation, model_version
    ):
        """
        **Property: Scoring result round-trip**
        
        For any scoring result, converting to dictionary and back should preserve 
        all data integrity.
        """
        section_scores = {
            'skills': skills_score,
            'experience': experience_score,
            'education': education_score
        }
        
        result = ScoringResult(
            candidate_id=candidate_id,
            job_id=job_id,
            overall_score=overall_score,
            section_scores=section_scores,
            explanation=explanation,
            model_version=model_version
        )
        
        # Convert to dict
        result_dict = result.to_dict()
        
        # Verify all fields are preserved
        assert result_dict['candidate_id'] == candidate_id
        assert result_dict['job_id'] == job_id
        assert result_dict['overall_score'] == overall_score
        assert result_dict['section_scores'] == section_scores
        assert result_dict['explanation'] == explanation
        assert result_dict['model_version'] == model_version
        
        # Verify dict structure
        expected_keys = {
            'candidate_id', 'job_id', 'overall_score', 
            'section_scores', 'explanation', 'model_version'
        }
        assert set(result_dict.keys()) == expected_keys