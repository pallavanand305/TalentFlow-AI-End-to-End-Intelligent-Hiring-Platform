"""Unit tests for model management API endpoints"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime
import pandas as pd

from backend.app.main import app
from backend.app.models.user import UserRole


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def mock_current_user():
    """Mock current user fixture"""
    user = MagicMock()
    user.id = "test-user-id"
    user.username = "testuser"
    user.role = UserRole.ADMIN
    return user


@pytest.fixture
def mock_model_registry():
    """Mock model registry fixture"""
    with patch('backend.app.api.models.model_registry') as mock:
        yield mock


class TestListModels:
    """Test cases for GET /api/v1/models"""
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_list_models_success(self, mock_require_role, mock_get_current_user, 
                                client, mock_current_user, mock_model_registry):
        """Test successful model listing"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        
        mock_models_data = [
            {
                "name": "scoring_model",
                "description": "Candidate scoring model",
                "creation_timestamp": datetime.now(),
                "last_updated_timestamp": datetime.now(),
                "latest_versions": [
                    {"version": "1", "stage": "Production", "run_id": "run123"},
                    {"version": "2", "stage": "Staging", "run_id": "run456"}
                ]
            }
        ]
        mock_model_registry.list_models.return_value = mock_models_data
        
        # Make request
        response = client.get("/api/v1/models")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "scoring_model"
        assert data[0]["description"] == "Candidate scoring model"
        assert len(data[0]["latest_versions"]) == 2
        
        mock_model_registry.list_models.assert_called_once()
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_list_models_empty(self, mock_require_role, mock_get_current_user,
                              client, mock_current_user, mock_model_registry):
        """Test listing models when no models exist"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        mock_model_registry.list_models.return_value = []
        
        # Make request
        response = client.get("/api/v1/models")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_list_models_error(self, mock_require_role, mock_get_current_user,
                              client, mock_current_user, mock_model_registry):
        """Test error handling in model listing"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        mock_model_registry.list_models.side_effect = Exception("MLflow connection error")
        
        # Make request
        response = client.get("/api/v1/models")
        
        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve models" in data["detail"]


class TestGetModelDetails:
    """Test cases for GET /api/v1/models/{model_name}"""
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_get_model_details_success(self, mock_require_role, mock_get_current_user,
                                      client, mock_current_user, mock_model_registry):
        """Test successful model details retrieval"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        
        mock_model_info = {
            "name": "scoring_model",
            "version": "1",
            "stage": "Production",
            "run_id": "run123",
            "created_at": datetime.now(),
            "description": "Production scoring model",
            "metrics": {"accuracy": 0.85, "f1_score": 0.82},
            "params": {"learning_rate": 0.01, "max_depth": 5},
            "tags": {"environment": "production"}
        }
        mock_model_registry.get_model_info.return_value = mock_model_info
        
        # Make request
        response = client.get("/api/v1/models/scoring_model?version=1")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "scoring_model"
        assert data["version"] == "1"
        assert data["stage"] == "Production"
        assert data["metrics"]["accuracy"] == 0.85
        assert data["params"]["learning_rate"] == 0.01
        
        mock_model_registry.get_model_info.assert_called_once_with("scoring_model", "1")
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_get_model_details_not_found(self, mock_require_role, mock_get_current_user,
                                        client, mock_current_user, mock_model_registry):
        """Test model not found error"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        mock_model_registry.get_model_info.return_value = {}
        
        # Make request
        response = client.get("/api/v1/models/nonexistent_model")
        
        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_get_model_details_without_version(self, mock_require_role, mock_get_current_user,
                                              client, mock_current_user, mock_model_registry):
        """Test getting model details without specific version"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        
        mock_model_info = {
            "name": "scoring_model",
            "description": "Scoring model",
            "creation_timestamp": datetime.now(),
            "last_updated_timestamp": datetime.now(),
            "latest_versions": [
                {"version": "1", "stage": "Production", "run_id": "run123"},
                {"version": "2", "stage": "Staging", "run_id": "run456"}
            ]
        }
        mock_model_registry.get_model_info.return_value = mock_model_info
        
        # Make request
        response = client.get("/api/v1/models/scoring_model")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "scoring_model"
        assert data["version"] is None  # No specific version requested
        assert len(data["latest_versions"]) == 2


class TestPromoteModel:
    """Test cases for POST /api/v1/models/promote"""
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_promote_model_success(self, mock_require_role, mock_get_current_user,
                                  client, mock_current_user, mock_model_registry):
        """Test successful model promotion"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        
        mock_model_info = {
            "name": "scoring_model",
            "version": "2",
            "stage": "Staging"
        }
        mock_model_registry.get_model_info.return_value = mock_model_info
        mock_model_registry.promote_model.return_value = True
        
        # Make request
        request_data = {
            "model_name": "scoring_model",
            "version": "2",
            "stage": "Production",
            "archive_existing": True
        }
        response = client.post("/api/v1/models/promote", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == "scoring_model"
        assert data["version"] == "2"
        assert data["stage"] == "Production"
        assert "Successfully promoted" in data["message"]
        
        mock_model_registry.promote_model.assert_called_once_with(
            model_name="scoring_model",
            version="2",
            stage="Production",
            archive_existing=True
        )
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_promote_model_invalid_stage(self, mock_require_role, mock_get_current_user,
                                        client, mock_current_user, mock_model_registry):
        """Test promotion with invalid stage"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        
        # Make request with invalid stage
        request_data = {
            "model_name": "scoring_model",
            "version": "2",
            "stage": "InvalidStage"
        }
        response = client.post("/api/v1/models/promote", json=request_data)
        
        # Assertions
        assert response.status_code == 400
        data = response.json()
        assert "Invalid stage" in data["detail"]
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_promote_model_not_found(self, mock_require_role, mock_get_current_user,
                                    client, mock_current_user, mock_model_registry):
        """Test promotion of non-existent model"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        mock_model_registry.get_model_info.return_value = {}
        
        # Make request
        request_data = {
            "model_name": "nonexistent_model",
            "version": "1",
            "stage": "Production"
        }
        response = client.post("/api/v1/models/promote", json=request_data)
        
        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]


class TestCompareModels:
    """Test cases for POST /api/v1/models/compare"""
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_compare_models_success(self, mock_require_role, mock_get_current_user,
                                   client, mock_current_user, mock_model_registry):
        """Test successful model comparison"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        
        # Create mock comparison DataFrame
        comparison_data = pd.DataFrame([
            {"version": "1", "accuracy": 0.80, "f1_score": 0.78, "learning_rate": 0.01},
            {"version": "2", "accuracy": 0.85, "f1_score": 0.82, "learning_rate": 0.005}
        ])
        mock_model_registry.compare_models.return_value = comparison_data
        
        # Make request
        request_data = {
            "model_name": "scoring_model",
            "versions": ["1", "2"]
        }
        response = client.post("/api/v1/models/compare", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "scoring_model"
        assert len(data["comparison_data"]) == 2
        assert data["comparison_data"][0]["version"] == "1"
        assert data["comparison_data"][1]["accuracy"] == 0.85
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_compare_models_insufficient_versions(self, mock_require_role, mock_get_current_user,
                                                 client, mock_current_user, mock_model_registry):
        """Test comparison with insufficient versions"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        
        # Make request with only one version
        request_data = {
            "model_name": "scoring_model",
            "versions": ["1"]
        }
        response = client.post("/api/v1/models/compare", json=request_data)
        
        # Assertions
        assert response.status_code == 400
        data = response.json()
        assert "At least 2 versions are required" in data["detail"]
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_compare_models_no_data(self, mock_require_role, mock_get_current_user,
                                   client, mock_current_user, mock_model_registry):
        """Test comparison when no data is found"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        mock_model_registry.compare_models.return_value = pd.DataFrame()  # Empty DataFrame
        
        # Make request
        request_data = {
            "model_name": "nonexistent_model",
            "versions": ["1", "2"]
        }
        response = client.post("/api/v1/models/compare", json=request_data)
        
        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "No data found" in data["detail"]


class TestRegisterModel:
    """Test cases for POST /api/v1/models/register"""
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_register_model_success(self, mock_require_role, mock_get_current_user,
                                   client, mock_current_user, mock_model_registry):
        """Test successful model registration"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        mock_model_registry.register_model.return_value = "3"
        
        # Make request
        request_data = {
            "run_id": "run789",
            "model_name": "new_scoring_model",
            "description": "New version of scoring model"
        }
        response = client.post("/api/v1/models/register", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == "new_scoring_model"
        assert data["version"] == "3"
        assert data["run_id"] == "run789"
        assert "Successfully registered" in data["message"]
        
        mock_model_registry.register_model.assert_called_once_with(
            run_id="run789",
            model_name="new_scoring_model",
            description="New version of scoring model"
        )
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_register_model_run_not_found(self, mock_require_role, mock_get_current_user,
                                         client, mock_current_user, mock_model_registry):
        """Test registration with non-existent run"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        mock_model_registry.register_model.side_effect = Exception("Run not found")
        
        # Make request
        request_data = {
            "run_id": "nonexistent_run",
            "model_name": "new_model"
        }
        response = client.post("/api/v1/models/register", json=request_data)
        
        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]


class TestMLflowHealth:
    """Test cases for GET /api/v1/models/health"""
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_mlflow_health_success(self, mock_require_role, mock_get_current_user,
                                  client, mock_current_user, mock_model_registry):
        """Test successful MLflow health check"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        
        health_data = {
            "status": "healthy",
            "tracking_uri": "http://localhost:5000",
            "experiments_count": 5,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        mock_model_registry.health_check.return_value = health_data
        
        # Make request
        response = client.get("/api/v1/models/health")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["tracking_uri"] == "http://localhost:5000"
        assert data["experiments_count"] == 5
    
    @patch('backend.app.api.models.get_current_user')
    @patch('backend.app.api.models.require_role')
    def test_mlflow_health_unhealthy(self, mock_require_role, mock_get_current_user,
                                    client, mock_current_user, mock_model_registry):
        """Test MLflow health check when unhealthy"""
        # Setup mocks
        mock_get_current_user.return_value = mock_current_user
        mock_require_role.return_value = lambda: None
        mock_model_registry.health_check.side_effect = Exception("Connection failed")
        
        # Make request
        response = client.get("/api/v1/models/health")
        
        # Assertions
        assert response.status_code == 200  # Health endpoint should always return 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data


class TestModelAPIAuthentication:
    """Test authentication and authorization for model endpoints"""
    
    def test_list_models_requires_auth(self, client):
        """Test that listing models requires authentication"""
        response = client.get("/api/v1/models")
        assert response.status_code == 401
    
    def test_promote_model_requires_admin(self, client):
        """Test that promoting models requires admin role"""
        # This would be tested with proper auth mocking
        # For now, just verify the endpoint exists
        request_data = {
            "model_name": "test_model",
            "version": "1",
            "stage": "Production"
        }
        response = client.post("/api/v1/models/promote", json=request_data)
        assert response.status_code == 401  # Unauthorized without auth
    
    def test_register_model_requires_admin(self, client):
        """Test that registering models requires admin role"""
        request_data = {
            "run_id": "test_run",
            "model_name": "test_model"
        }
        response = client.post("/api/v1/models/register", json=request_data)
        assert response.status_code == 401  # Unauthorized without auth