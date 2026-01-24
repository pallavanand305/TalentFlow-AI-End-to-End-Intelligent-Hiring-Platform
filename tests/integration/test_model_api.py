"""Integration tests for model management API endpoints"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import patch, MagicMock
import json

from backend.app.main import app
from backend.app.core.database import get_db_session
from backend.app.models.user import User, UserRole
from backend.app.services.auth_service import AuthService


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
async def admin_user(db_session: AsyncSession):
    """Create admin user for testing"""
    user = User(
        username="admin_user",
        email="admin@test.com",
        password_hash="hashed_password",
        role=UserRole.ADMIN
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def recruiter_user(db_session: AsyncSession):
    """Create recruiter user for testing"""
    user = User(
        username="recruiter_user",
        email="recruiter@test.com",
        password_hash="hashed_password",
        role=UserRole.RECRUITER
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def admin_headers(admin_user):
    """Create authorization headers for admin user"""
    auth_service = AuthService()
    token = auth_service.create_access_token(str(admin_user.id), admin_user.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def recruiter_headers(recruiter_user):
    """Create authorization headers for recruiter user"""
    auth_service = AuthService()
    token = auth_service.create_access_token(str(recruiter_user.id), recruiter_user.role.value)
    return {"Authorization": f"Bearer {token}"}


class TestModelListIntegration:
    """Integration tests for model listing endpoint"""
    
    @patch('backend.app.services.model_registry.model_registry.list_models')
    async def test_list_models_integration(self, mock_list_models, client, admin_headers):
        """Test complete model listing workflow"""
        # Mock MLflow response
        mock_models_data = [
            {
                "name": "candidate_scoring_model",
                "description": "ML model for candidate-job matching",
                "creation_timestamp": "2024-01-15T10:00:00Z",
                "last_updated_timestamp": "2024-01-15T12:00:00Z",
                "latest_versions": [
                    {
                        "version": "1",
                        "stage": "Production",
                        "run_id": "abc123"
                    },
                    {
                        "version": "2", 
                        "stage": "Staging",
                        "run_id": "def456"
                    }
                ]
            },
            {
                "name": "resume_parser_model",
                "description": "NLP model for resume parsing",
                "creation_timestamp": "2024-01-10T09:00:00Z",
                "last_updated_timestamp": "2024-01-12T11:00:00Z",
                "latest_versions": [
                    {
                        "version": "3",
                        "stage": "Production", 
                        "run_id": "ghi789"
                    }
                ]
            }
        ]
        mock_list_models.return_value = mock_models_data
        
        # Make request
        response = client.get("/api/v1/models", headers=admin_headers)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 2
        
        # Check first model
        scoring_model = data[0]
        assert scoring_model["name"] == "candidate_scoring_model"
        assert scoring_model["description"] == "ML model for candidate-job matching"
        assert len(scoring_model["latest_versions"]) == 2
        
        production_version = next(v for v in scoring_model["latest_versions"] if v["stage"] == "Production")
        assert production_version["version"] == "1"
        assert production_version["run_id"] == "abc123"
        
        # Check second model
        parser_model = data[1]
        assert parser_model["name"] == "resume_parser_model"
        assert len(parser_model["latest_versions"]) == 1
        
        mock_list_models.assert_called_once()
    
    async def test_list_models_unauthorized(self, client):
        """Test that listing models requires authentication"""
        response = client.get("/api/v1/models")
        assert response.status_code == 401
    
    @patch('backend.app.services.model_registry.model_registry.list_models')
    async def test_list_models_recruiter_access(self, mock_list_models, client, recruiter_headers):
        """Test that recruiters can list models"""
        mock_list_models.return_value = []
        
        response = client.get("/api/v1/models", headers=recruiter_headers)
        assert response.status_code == 200


class TestModelDetailsIntegration:
    """Integration tests for model details endpoint"""
    
    @patch('backend.app.services.model_registry.model_registry.get_model_info')
    async def test_get_model_details_with_version(self, mock_get_model_info, client, admin_headers):
        """Test getting specific model version details"""
        # Mock MLflow response
        mock_model_info = {
            "name": "candidate_scoring_model",
            "version": "2",
            "stage": "Staging",
            "run_id": "def456",
            "created_at": "2024-01-15T12:00:00Z",
            "description": "Updated scoring model with improved accuracy",
            "metrics": {
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.89,
                "f1_score": 0.87
            },
            "params": {
                "learning_rate": 0.001,
                "max_depth": 10,
                "n_estimators": 100
            },
            "tags": {
                "environment": "staging",
                "model_type": "xgboost"
            }
        }
        mock_get_model_info.return_value = mock_model_info
        
        # Make request
        response = client.get(
            "/api/v1/models/candidate_scoring_model?version=2",
            headers=admin_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "candidate_scoring_model"
        assert data["version"] == "2"
        assert data["stage"] == "Staging"
        assert data["run_id"] == "def456"
        assert data["description"] == "Updated scoring model with improved accuracy"
        
        # Check metrics
        assert data["metrics"]["accuracy"] == 0.87
        assert data["metrics"]["f1_score"] == 0.87
        
        # Check parameters
        assert data["params"]["learning_rate"] == 0.001
        assert data["params"]["max_depth"] == 10
        
        # Check tags
        assert data["tags"]["environment"] == "staging"
        assert data["tags"]["model_type"] == "xgboost"
        
        mock_get_model_info.assert_called_once_with("candidate_scoring_model", "2")
    
    @patch('backend.app.services.model_registry.model_registry.get_model_info')
    async def test_get_model_details_without_version(self, mock_get_model_info, client, admin_headers):
        """Test getting model details without specific version"""
        # Mock MLflow response for model-level info
        mock_model_info = {
            "name": "candidate_scoring_model",
            "description": "ML model for candidate-job matching",
            "creation_timestamp": "2024-01-15T10:00:00Z",
            "last_updated_timestamp": "2024-01-15T12:00:00Z",
            "latest_versions": [
                {
                    "version": "1",
                    "stage": "Production",
                    "run_id": "abc123"
                },
                {
                    "version": "2",
                    "stage": "Staging", 
                    "run_id": "def456"
                }
            ]
        }
        mock_get_model_info.return_value = mock_model_info
        
        # Make request
        response = client.get(
            "/api/v1/models/candidate_scoring_model",
            headers=admin_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "candidate_scoring_model"
        assert data["version"] is None  # No specific version requested
        assert data["description"] == "ML model for candidate-job matching"
        assert len(data["latest_versions"]) == 2
        
        production_version = next(v for v in data["latest_versions"] if v["stage"] == "Production")
        assert production_version["version"] == "1"
        
        mock_get_model_info.assert_called_once_with("candidate_scoring_model", None)
    
    @patch('backend.app.services.model_registry.model_registry.get_model_info')
    async def test_get_model_details_not_found(self, mock_get_model_info, client, admin_headers):
        """Test getting details for non-existent model"""
        mock_get_model_info.return_value = {}
        
        response = client.get(
            "/api/v1/models/nonexistent_model",
            headers=admin_headers
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]


class TestModelPromotionIntegration:
    """Integration tests for model promotion endpoint"""
    
    @patch('backend.app.services.model_registry.model_registry.get_model_info')
    @patch('backend.app.services.model_registry.model_registry.promote_model')
    async def test_promote_model_to_production(self, mock_promote_model, mock_get_model_info, 
                                              client, admin_headers):
        """Test promoting a model to production"""
        # Mock model exists
        mock_get_model_info.return_value = {
            "name": "candidate_scoring_model",
            "version": "2",
            "stage": "Staging"
        }
        mock_promote_model.return_value = True
        
        # Make request
        request_data = {
            "model_name": "candidate_scoring_model",
            "version": "2",
            "stage": "Production",
            "archive_existing": True
        }
        response = client.post(
            "/api/v1/models/promote",
            headers=admin_headers,
            json=request_data
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_name"] == "candidate_scoring_model"
        assert data["version"] == "2"
        assert data["stage"] == "Production"
        assert "Successfully promoted" in data["message"]
        
        mock_promote_model.assert_called_once_with(
            model_name="candidate_scoring_model",
            version="2",
            stage="Production",
            archive_existing=True
        )
    
    async def test_promote_model_requires_admin(self, client, recruiter_headers):
        """Test that model promotion requires admin role"""
        request_data = {
            "model_name": "test_model",
            "version": "1",
            "stage": "Production"
        }
        response = client.post(
            "/api/v1/models/promote",
            headers=recruiter_headers,
            json=request_data
        )
        
        assert response.status_code == 403  # Forbidden
    
    async def test_promote_model_invalid_stage(self, client, admin_headers):
        """Test promotion with invalid stage"""
        request_data = {
            "model_name": "test_model",
            "version": "1",
            "stage": "InvalidStage"
        }
        response = client.post(
            "/api/v1/models/promote",
            headers=admin_headers,
            json=request_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid stage" in data["detail"]


class TestModelComparisonIntegration:
    """Integration tests for model comparison endpoint"""
    
    @patch('backend.app.services.model_registry.model_registry.compare_models')
    async def test_compare_models_success(self, mock_compare_models, client, admin_headers):
        """Test successful model comparison"""
        import pandas as pd
        
        # Mock comparison data
        comparison_df = pd.DataFrame([
            {
                "version": "1",
                "run_id": "abc123",
                "stage": "Production",
                "created_at": "2024-01-15T10:00:00Z",
                "accuracy": 0.82,
                "precision": 0.80,
                "recall": 0.84,
                "f1_score": 0.82,
                "learning_rate": 0.01,
                "max_depth": 8
            },
            {
                "version": "2",
                "run_id": "def456", 
                "stage": "Staging",
                "created_at": "2024-01-15T12:00:00Z",
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.89,
                "f1_score": 0.87,
                "learning_rate": 0.001,
                "max_depth": 10
            }
        ])
        mock_compare_models.return_value = comparison_df
        
        # Make request
        request_data = {
            "model_name": "candidate_scoring_model",
            "versions": ["1", "2"]
        }
        response = client.post(
            "/api/v1/models/compare",
            headers=admin_headers,
            json=request_data
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_name"] == "candidate_scoring_model"
        assert len(data["comparison_data"]) == 2
        
        # Check version 1 data
        v1_data = next(d for d in data["comparison_data"] if d["version"] == "1")
        assert v1_data["accuracy"] == 0.82
        assert v1_data["learning_rate"] == 0.01
        assert v1_data["stage"] == "Production"
        
        # Check version 2 data
        v2_data = next(d for d in data["comparison_data"] if d["version"] == "2")
        assert v2_data["accuracy"] == 0.87
        assert v2_data["learning_rate"] == 0.001
        assert v2_data["stage"] == "Staging"
        
        mock_compare_models.assert_called_once_with(
            model_name="candidate_scoring_model",
            versions=["1", "2"]
        )


class TestModelRegistrationIntegration:
    """Integration tests for model registration endpoint"""
    
    @patch('backend.app.services.model_registry.model_registry.register_model')
    async def test_register_model_success(self, mock_register_model, client, admin_headers):
        """Test successful model registration"""
        mock_register_model.return_value = "3"
        
        # Make request
        request_data = {
            "run_id": "ghi789",
            "model_name": "new_scoring_model",
            "description": "Improved scoring model with better accuracy"
        }
        response = client.post(
            "/api/v1/models/register",
            headers=admin_headers,
            json=request_data
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_name"] == "new_scoring_model"
        assert data["version"] == "3"
        assert data["run_id"] == "ghi789"
        assert "Successfully registered" in data["message"]
        
        mock_register_model.assert_called_once_with(
            run_id="ghi789",
            model_name="new_scoring_model",
            description="Improved scoring model with better accuracy"
        )
    
    async def test_register_model_requires_admin(self, client, recruiter_headers):
        """Test that model registration requires admin role"""
        request_data = {
            "run_id": "test_run",
            "model_name": "test_model"
        }
        response = client.post(
            "/api/v1/models/register",
            headers=recruiter_headers,
            json=request_data
        )
        
        assert response.status_code == 403  # Forbidden


class TestMLflowHealthIntegration:
    """Integration tests for MLflow health endpoint"""
    
    @patch('backend.app.services.model_registry.model_registry.health_check')
    async def test_mlflow_health_check_healthy(self, mock_health_check, client, admin_headers):
        """Test MLflow health check when healthy"""
        mock_health_data = {
            "status": "healthy",
            "tracking_uri": "http://localhost:5000",
            "experiments_count": 3,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        mock_health_check.return_value = mock_health_data
        
        # Make request
        response = client.get("/api/v1/models/health", headers=admin_headers)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["tracking_uri"] == "http://localhost:5000"
        assert data["experiments_count"] == 3
        assert "timestamp" in data
        
        mock_health_check.assert_called_once()
    
    @patch('backend.app.services.model_registry.model_registry.health_check')
    async def test_mlflow_health_check_unhealthy(self, mock_health_check, client, admin_headers):
        """Test MLflow health check when unhealthy"""
        mock_health_check.side_effect = Exception("Connection timeout")
        
        # Make request
        response = client.get("/api/v1/models/health", headers=admin_headers)
        
        # Assertions
        assert response.status_code == 200  # Health endpoint should always return 200
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert "error" in data


class TestModelAPIWorkflow:
    """Integration tests for complete model management workflows"""
    
    @patch('backend.app.services.model_registry.model_registry.list_models')
    @patch('backend.app.services.model_registry.model_registry.get_model_info')
    @patch('backend.app.services.model_registry.model_registry.promote_model')
    async def test_complete_model_promotion_workflow(self, mock_promote_model, mock_get_model_info,
                                                    mock_list_models, client, admin_headers):
        """Test complete workflow: list models -> get details -> promote"""
        # Step 1: List models
        mock_list_models.return_value = [
            {
                "name": "scoring_model",
                "description": "Candidate scoring model",
                "creation_timestamp": "2024-01-15T10:00:00Z",
                "last_updated_timestamp": "2024-01-15T12:00:00Z",
                "latest_versions": [
                    {"version": "1", "stage": "Production", "run_id": "abc123"},
                    {"version": "2", "stage": "Staging", "run_id": "def456"}
                ]
            }
        ]
        
        response = client.get("/api/v1/models", headers=admin_headers)
        assert response.status_code == 200
        models = response.json()
        assert len(models) == 1
        
        # Step 2: Get details for staging version
        mock_get_model_info.return_value = {
            "name": "scoring_model",
            "version": "2",
            "stage": "Staging",
            "run_id": "def456",
            "metrics": {"accuracy": 0.87, "f1_score": 0.85}
        }
        
        response = client.get("/api/v1/models/scoring_model?version=2", headers=admin_headers)
        assert response.status_code == 200
        details = response.json()
        assert details["stage"] == "Staging"
        assert details["metrics"]["accuracy"] == 0.87
        
        # Step 3: Promote to production
        mock_promote_model.return_value = True
        
        promote_request = {
            "model_name": "scoring_model",
            "version": "2",
            "stage": "Production",
            "archive_existing": True
        }
        response = client.post("/api/v1/models/promote", headers=admin_headers, json=promote_request)
        assert response.status_code == 200
        promotion_result = response.json()
        assert promotion_result["success"] is True
        assert promotion_result["stage"] == "Production"