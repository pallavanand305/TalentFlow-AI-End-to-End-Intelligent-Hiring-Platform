"""Integration tests for explanation API endpoints"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.score import ScoreExplanationRequest, ScoreExplanationResponse


class TestExplanationAPI:
    """Test explanation API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test-token"}
    
    @pytest.fixture
    def sample_score_id(self):
        """Sample score UUID"""
        return uuid4()
    
    @pytest.fixture
    def sample_candidate_id(self):
        """Sample candidate UUID"""
        return uuid4()
    
    @pytest.fixture
    def sample_job_id(self):
        """Sample job UUID"""
        return uuid4()
    
    @patch('backend.app.api.scores.get_current_user')
    @patch('backend.app.api.scores.require_any_role')
    @patch('backend.app.api.scores.get_scoring_service')
    def test_generate_score_explanation_basic(
        self, 
        mock_get_service, 
        mock_require_role, 
        mock_get_user,
        client, 
        auth_headers, 
        sample_score_id
    ):
        """Test basic score explanation generation"""
        # Mock user
        mock_user = Mock()
        mock_user.id = uuid4()
        mock_get_user.return_value = mock_user
        mock_require_role.return_value = mock_user
        
        # Mock score
        mock_score = Mock()
        mock_score.id = sample_score_id
        mock_score.candidate_id = uuid4()
        mock_score.job_id = uuid4()
        mock_score.score = 0.75
        
        # Mock scoring service
        mock_service = Mock()
        mock_service.get_score_by_id.return_value = mock_score
        mock_service.generate_detailed_explanation.return_value = {
            'explanation': 'This candidate shows a good match for the position.',
            'match_level': 'good',
            'overall_score': 0.75,
            'section_analysis': [
                {
                    'section': 'skills',
                    'score': 0.8,
                    'weight': 0.4,
                    'contribution': 0.32,
                    'key_matches': ['Python', 'SQL'],
                    'missing_elements': ['Docker']
                }
            ]
        }
        mock_get_service.return_value = mock_service
        
        # Make request
        request_data = {
            'score_id': str(sample_score_id),
            'detailed': False
        }
        
        response = client.post(
            f"/api/v1/scores/{sample_score_id}/explain",
            json=request_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert 'score_id' in data
        assert 'score' in data
        assert 'explanation' in data
        assert data['explanation'] == 'This candidate shows a good match for the position.'
        assert data['score'] == 0.75
        assert data['section_scores'] is None  # Not detailed
        assert data['key_matches'] is None  # Not detailed
    
    @patch('backend.app.api.scores.get_current_user')
    @patch('backend.app.api.scores.require_any_role')
    @patch('backend.app.api.scores.get_scoring_service')
    def test_generate_score_explanation_detailed(
        self, 
        mock_get_service, 
        mock_require_role, 
        mock_get_user,
        client, 
        auth_headers, 
        sample_score_id
    ):
        """Test detailed score explanation generation"""
        # Mock user
        mock_user = Mock()
        mock_user.id = uuid4()
        mock_get_user.return_value = mock_user
        mock_require_role.return_value = mock_user
        
        # Mock score
        mock_score = Mock()
        mock_score.id = sample_score_id
        mock_score.candidate_id = uuid4()
        mock_score.job_id = uuid4()
        mock_score.score = 0.75
        
        # Mock scoring service
        mock_service = Mock()
        mock_service.get_score_by_id.return_value = mock_score
        mock_service.generate_detailed_explanation.return_value = {
            'explanation': 'This candidate shows a good match for the position.',
            'match_level': 'good',
            'overall_score': 0.75,
            'section_analysis': [
                {
                    'section': 'skills',
                    'score': 0.8,
                    'weight': 0.4,
                    'contribution': 0.32,
                    'key_matches': ['Python', 'SQL'],
                    'missing_elements': ['Docker']
                },
                {
                    'section': 'experience',
                    'score': 0.7,
                    'weight': 0.4,
                    'contribution': 0.28,
                    'key_matches': ['Software Engineer'],
                    'missing_elements': []
                }
            ],
            'improvement_suggestions': ['Consider learning Docker for containerization']
        }
        mock_get_service.return_value = mock_service
        
        # Make request
        request_data = {
            'score_id': str(sample_score_id),
            'detailed': True
        }
        
        response = client.post(
            f"/api/v1/scores/{sample_score_id}/explain",
            json=request_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert 'score_id' in data
        assert 'score' in data
        assert 'explanation' in data
        assert 'section_scores' in data
        assert 'key_matches' in data
        assert 'improvement_suggestions' in data
        
        # Verify detailed data
        assert data['section_scores']['skills'] == 0.8
        assert data['section_scores']['experience'] == 0.7
        assert 'Python' in data['key_matches']
        assert 'SQL' in data['key_matches']
        assert len(data['improvement_suggestions']) == 1
    
    @patch('backend.app.api.scores.get_current_user')
    @patch('backend.app.api.scores.require_any_role')
    @patch('backend.app.api.scores.get_scoring_service')
    def test_generate_candidate_job_explanation(
        self, 
        mock_get_service, 
        mock_require_role, 
        mock_get_user,
        client, 
        auth_headers, 
        sample_candidate_id,
        sample_job_id
    ):
        """Test candidate-job explanation generation"""
        # Mock user
        mock_user = Mock()
        mock_user.id = uuid4()
        mock_get_user.return_value = mock_user
        mock_require_role.return_value = mock_user
        
        # Mock scoring service
        mock_service = Mock()
        mock_service.generate_detailed_explanation.return_value = {
            'score_id': str(uuid4()),
            'explanation': 'This candidate shows a moderate match for the position.',
            'match_level': 'moderate',
            'overall_score': 0.55,
            'section_analysis': [
                {
                    'section': 'skills',
                    'score': 0.6,
                    'weight': 0.4,
                    'contribution': 0.24,
                    'key_matches': ['JavaScript'],
                    'missing_elements': ['React', 'Node.js']
                }
            ],
            'improvement_suggestions': ['Consider learning React for frontend development']
        }
        mock_get_service.return_value = mock_service
        
        # Make request
        response = client.post(
            f"/api/v1/scores/candidates/{sample_candidate_id}/jobs/{sample_job_id}/explain?detailed=true",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert 'score' in data
        assert 'explanation' in data
        assert 'section_scores' in data
        assert 'improvement_suggestions' in data
        
        assert data['score'] == 0.55
        assert 'moderate match' in data['explanation']
        assert data['section_scores']['skills'] == 0.6
        assert len(data['improvement_suggestions']) == 1
    
    @patch('backend.app.api.scores.get_current_user')
    @patch('backend.app.api.scores.require_any_role')
    @patch('backend.app.api.scores.get_scoring_service')
    def test_generate_candidate_job_explanation_basic(
        self, 
        mock_get_service, 
        mock_require_role, 
        mock_get_user,
        client, 
        auth_headers, 
        sample_candidate_id,
        sample_job_id
    ):
        """Test basic candidate-job explanation generation"""
        # Mock user
        mock_user = Mock()
        mock_user.id = uuid4()
        mock_get_user.return_value = mock_user
        mock_require_role.return_value = mock_user
        
        # Mock scoring service
        mock_service = Mock()
        mock_service.generate_detailed_explanation.return_value = {
            'score_id': str(uuid4()),
            'explanation': 'This candidate shows an excellent match for the position.',
            'match_level': 'excellent',
            'overall_score': 0.85,
            'section_analysis': []
        }
        mock_get_service.return_value = mock_service
        
        # Make request (detailed=false by default)
        response = client.post(
            f"/api/v1/scores/candidates/{sample_candidate_id}/jobs/{sample_job_id}/explain",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data['score'] == 0.85
        assert 'excellent match' in data['explanation']
        assert data['section_scores'] is None  # Not detailed
        assert data['key_matches'] is None  # Not detailed
        assert data['improvement_suggestions'] is None  # Not detailed
    
    @patch('backend.app.api.scores.get_current_user')
    @patch('backend.app.api.scores.require_any_role')
    @patch('backend.app.api.scores.get_scoring_service')
    def test_score_explanation_not_found(
        self, 
        mock_get_service, 
        mock_require_role, 
        mock_get_user,
        client, 
        auth_headers, 
        sample_score_id
    ):
        """Test score explanation when score not found"""
        # Mock user
        mock_user = Mock()
        mock_user.id = uuid4()
        mock_get_user.return_value = mock_user
        mock_require_role.return_value = mock_user
        
        # Mock scoring service to raise NotFoundException
        from backend.app.core.exceptions import NotFoundException
        mock_service = Mock()
        mock_service.get_score_by_id.side_effect = NotFoundException("Score not found")
        mock_get_service.return_value = mock_service
        
        # Make request
        request_data = {
            'score_id': str(sample_score_id),
            'detailed': False
        }
        
        response = client.post(
            f"/api/v1/scores/{sample_score_id}/explain",
            json=request_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 404
        assert "Score not found" in response.json()['detail']
    
    @patch('backend.app.api.scores.get_current_user')
    @patch('backend.app.api.scores.require_any_role')
    @patch('backend.app.api.scores.get_scoring_service')
    def test_candidate_job_explanation_not_found(
        self, 
        mock_get_service, 
        mock_require_role, 
        mock_get_user,
        client, 
        auth_headers, 
        sample_candidate_id,
        sample_job_id
    ):
        """Test candidate-job explanation when candidate/job not found"""
        # Mock user
        mock_user = Mock()
        mock_user.id = uuid4()
        mock_get_user.return_value = mock_user
        mock_require_role.return_value = mock_user
        
        # Mock scoring service to raise NotFoundException
        from backend.app.core.exceptions import NotFoundException
        mock_service = Mock()
        mock_service.generate_detailed_explanation.side_effect = NotFoundException("Candidate not found")
        mock_get_service.return_value = mock_service
        
        # Make request
        response = client.post(
            f"/api/v1/scores/candidates/{sample_candidate_id}/jobs/{sample_job_id}/explain",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 404
        assert "Candidate not found" in response.json()['detail']
    
    @patch('backend.app.api.scores.get_current_user')
    @patch('backend.app.api.scores.require_any_role')
    @patch('backend.app.api.scores.get_scoring_service')
    def test_explanation_generation_error(
        self, 
        mock_get_service, 
        mock_require_role, 
        mock_get_user,
        client, 
        auth_headers, 
        sample_score_id
    ):
        """Test explanation generation when service error occurs"""
        # Mock user
        mock_user = Mock()
        mock_user.id = uuid4()
        mock_get_user.return_value = mock_user
        mock_require_role.return_value = mock_user
        
        # Mock score
        mock_score = Mock()
        mock_score.id = sample_score_id
        mock_score.candidate_id = uuid4()
        mock_score.job_id = uuid4()
        mock_score.score = 0.75
        
        # Mock scoring service to raise exception
        mock_service = Mock()
        mock_service.get_score_by_id.return_value = mock_score
        mock_service.generate_detailed_explanation.side_effect = Exception("Service error")
        mock_get_service.return_value = mock_service
        
        # Make request
        request_data = {
            'score_id': str(sample_score_id),
            'detailed': False
        }
        
        response = client.post(
            f"/api/v1/scores/{sample_score_id}/explain",
            json=request_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 500
        assert "Failed to generate score explanation" in response.json()['detail']
    
    def test_explanation_request_schema_validation(self):
        """Test explanation request schema validation"""
        # Valid request
        valid_request = ScoreExplanationRequest(
            score_id=uuid4(),
            detailed=True
        )
        assert valid_request.detailed is True
        
        # Default values
        default_request = ScoreExplanationRequest(score_id=uuid4())
        assert default_request.detailed is False
    
    def test_explanation_response_schema_validation(self):
        """Test explanation response schema validation"""
        # Valid response
        response_data = {
            'score_id': uuid4(),
            'score': 0.75,
            'explanation': 'Test explanation',
            'section_scores': {'skills': 0.8, 'experience': 0.7},
            'key_matches': ['Python', 'SQL'],
            'improvement_suggestions': ['Learn Docker']
        }
        
        response = ScoreExplanationResponse(**response_data)
        assert response.score == 0.75
        assert response.explanation == 'Test explanation'
        assert len(response.section_scores) == 2
        assert len(response.key_matches) == 2
        assert len(response.improvement_suggestions) == 1
        
        # Minimal response
        minimal_response = ScoreExplanationResponse(
            score_id=uuid4(),
            score=0.5,
            explanation='Minimal explanation'
        )
        assert minimal_response.section_scores is None
        assert minimal_response.key_matches is None
        assert minimal_response.improvement_suggestions is None


class TestExplanationAPIIntegration:
    """Integration tests for explanation API with real scoring engine"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test-token"}
    
    @patch('backend.app.api.scores.get_current_user')
    @patch('backend.app.api.scores.require_any_role')
    def test_explanation_api_openapi_schema(
        self, 
        mock_require_role, 
        mock_get_user,
        client
    ):
        """Test that explanation endpoints are included in OpenAPI schema"""
        # Mock user
        mock_user = Mock()
        mock_user.id = uuid4()
        mock_get_user.return_value = mock_user
        mock_require_role.return_value = mock_user
        
        # Get OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        paths = schema.get('paths', {})
        
        # Check that explanation endpoints are documented
        score_explain_path = '/api/v1/scores/{score_id}/explain'
        candidate_job_explain_path = '/api/v1/scores/candidates/{candidate_id}/jobs/{job_id}/explain'
        
        # Note: The actual path format in OpenAPI might be slightly different
        # This test verifies the endpoints are documented
        explanation_endpoints = [
            path for path in paths.keys() 
            if 'explain' in path and 'scores' in path
        ]
        
        assert len(explanation_endpoints) >= 2, f"Expected explanation endpoints in: {list(paths.keys())}"