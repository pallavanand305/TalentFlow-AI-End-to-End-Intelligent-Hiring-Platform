"""Unit tests for MLflow model registry service"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow

from backend.app.services.model_registry import ModelRegistry
from backend.app.core.config import settings


class TestModelRegistry:
    """Test cases for ModelRegistry service"""
    
    @pytest.fixture
    def mock_mlflow_client(self):
        """Mock MLflow client"""
        with patch('backend.app.services.model_registry.MlflowClient') as mock_client:
            yield mock_client.return_value
    
    @pytest.fixture
    def mock_mlflow(self):
        """Mock MLflow module functions"""
        with patch('backend.app.services.model_registry.mlflow') as mock_mlflow_module:
            # Mock the context manager for start_run
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id_12345"
            mock_context = Mock()
            mock_context.__enter__ = Mock(return_value=mock_run)
            mock_context.__exit__ = Mock(return_value=None)
            mock_mlflow_module.start_run.return_value = mock_context
            yield mock_mlflow_module
    
    @pytest.fixture
    def model_registry(self, mock_mlflow_client):
        """Create ModelRegistry instance with mocked dependencies"""
        with patch('backend.app.services.model_registry.ModelVersionRepository'):
            registry = ModelRegistry()
            registry.client = mock_mlflow_client
            return registry
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple sklearn model for testing"""
        model = LogisticRegression()
        # Create some dummy data to fit the model
        X = [[1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0]
        model.fit(X, y)
        return model
    
    def test_init_sets_tracking_uri(self, mock_mlflow):
        """Test that initialization sets the MLflow tracking URI"""
        with patch('backend.app.services.model_registry.ModelVersionRepository'):
            ModelRegistry()
        
        mock_mlflow.set_tracking_uri.assert_called_once_with(settings.MLFLOW_TRACKING_URI)
    
    @pytest.mark.asyncio
    async def test_log_model_success(self, model_registry, mock_mlflow, sample_model):
        """Test successful model logging"""
        metrics = {"accuracy": 0.85, "f1_score": 0.82}
        params = {"C": 1.0, "solver": "liblinear"}
        
        with patch('backend.app.services.model_registry.mlflow.sklearn.log_model') as mock_log_model:
            run_id = await model_registry.log_model(
                model=sample_model,
                model_name="test_model",
                metrics=metrics,
                params=params
            )
        
        assert run_id == "test_run_id_12345"
        mock_mlflow.log_params.assert_called_once_with(params)
        mock_mlflow.log_metrics.assert_called_once_with(metrics)
        mock_log_model.assert_called_once_with(sample_model, "model")
    
    @pytest.mark.asyncio
    async def test_log_model_with_artifacts(self, model_registry, mock_mlflow, sample_model):
        """Test model logging with artifacts"""
        metrics = {"accuracy": 0.85}
        params = {"C": 1.0}
        artifacts = {"config": "/path/to/config.json"}
        
        with patch('backend.app.services.model_registry.mlflow.sklearn.log_model'):
            await model_registry.log_model(
                model=sample_model,
                model_name="test_model",
                metrics=metrics,
                params=params,
                artifacts=artifacts
            )
        
        mock_mlflow.log_artifact.assert_called_once_with("/path/to/config.json", "config")
    
    @pytest.mark.asyncio
    async def test_load_model_by_stage(self, model_registry):
        """Test loading model by stage"""
        model_name = "test_model"
        stage = "Production"
        
        with patch('backend.app.services.model_registry.mlflow.sklearn.load_model') as mock_load:
            mock_load.return_value = Mock()
            
            model = await model_registry.load_model(
                model_name=model_name,
                stage=stage
            )
        
        expected_uri = f"models:/{model_name}/{stage}"
        mock_load.assert_called_once_with(expected_uri)
        assert model is not None
    
    @pytest.mark.asyncio
    async def test_load_model_latest_version(self, model_registry, mock_mlflow_client):
        """Test loading latest model version"""
        model_name = "test_model"
        
        # Mock the latest version response
        mock_version = Mock()
        mock_version.version = "1"
        mock_mlflow_client.get_latest_versions.return_value = [mock_version]
        
        with patch('backend.app.services.model_registry.mlflow.sklearn.load_model') as mock_load:
            mock_load.return_value = Mock()
            
            model = await model_registry.load_model(
                model_name=model_name,
                version="latest"
            )
        
        expected_uri = f"models:/{model_name}/1"
        mock_load.assert_called_once_with(expected_uri)
    
    @pytest.mark.asyncio
    async def test_register_model_success(self, model_registry, mock_mlflow):
        """Test successful model registration"""
        run_id = "test_run_id"
        model_name = "test_model"
        
        mock_version = Mock()
        mock_version.version = "1"
        mock_mlflow.register_model.return_value = mock_version
        
        version = await model_registry.register_model(
            run_id=run_id,
            model_name=model_name
        )
        
        assert version == "1"
        expected_uri = f"runs:/{run_id}/model"
        mock_mlflow.register_model.assert_called_once_with(
            model_uri=expected_uri,
            name=model_name,
            description=None
        )
    
    @pytest.mark.asyncio
    async def test_promote_model_success(self, model_registry, mock_mlflow_client):
        """Test successful model promotion"""
        model_name = "test_model"
        version = "1"
        stage = "Production"
        
        # Mock existing models in production
        existing_model = Mock()
        existing_model.version = "0"
        mock_mlflow_client.get_latest_versions.return_value = [existing_model]
        
        success = await model_registry.promote_model(
            model_name=model_name,
            version=version,
            stage=stage
        )
        
        assert success is True
        
        # Should archive existing model
        mock_mlflow_client.transition_model_version_stage.assert_any_call(
            name=model_name,
            version="0",
            stage="Archived"
        )
        
        # Should promote new model
        mock_mlflow_client.transition_model_version_stage.assert_any_call(
            name=model_name,
            version=version,
            stage=stage
        )
    
    @pytest.mark.asyncio
    async def test_compare_models_success(self, model_registry, mock_mlflow_client):
        """Test model comparison functionality"""
        model_name = "test_model"
        versions = ["1", "2"]
        
        # Mock model version and run data
        mock_version_1 = Mock()
        mock_version_1.run_id = "run_1"
        mock_version_1.current_stage = "Production"
        mock_version_1.creation_timestamp = 1234567890
        
        mock_version_2 = Mock()
        mock_version_2.run_id = "run_2"
        mock_version_2.current_stage = "Staging"
        mock_version_2.creation_timestamp = 1234567891
        
        mock_run_1 = Mock()
        mock_run_1.data.metrics = {"accuracy": 0.85}
        mock_run_1.data.params = {"C": 1.0}
        
        mock_run_2 = Mock()
        mock_run_2.data.metrics = {"accuracy": 0.87}
        mock_run_2.data.params = {"C": 0.5}
        
        mock_mlflow_client.get_model_version.side_effect = [mock_version_1, mock_version_2]
        mock_mlflow_client.get_run.side_effect = [mock_run_1, mock_run_2]
        
        df = await model_registry.compare_models(model_name, versions)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "version" in df.columns
        assert "accuracy" in df.columns
        assert "C" in df.columns
    
    @pytest.mark.asyncio
    async def test_list_models_success(self, model_registry, mock_mlflow_client):
        """Test listing all models"""
        # Mock registered models
        mock_model = Mock()
        mock_model.name = "test_model"
        mock_model.description = "Test model"
        mock_model.creation_timestamp = 1234567890
        mock_model.last_updated_timestamp = 1234567891
        
        mock_mlflow_client.search_registered_models.return_value = [mock_model]
        
        # Mock latest versions
        mock_version = Mock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.run_id = "run_1"
        
        mock_mlflow_client.get_latest_versions.return_value = [mock_version]
        
        models = await model_registry.list_models()
        
        assert len(models) == 1
        assert models[0]["name"] == "test_model"
        assert models[0]["description"] == "Test model"
        assert len(models[0]["latest_versions"]) == 1
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, model_registry, mock_mlflow_client):
        """Test health check when MLflow is healthy"""
        mock_mlflow_client.search_experiments.return_value = [Mock(), Mock()]
        
        health = await model_registry.health_check()
        
        assert health["status"] == "healthy"
        assert health["experiments_count"] == 2
        assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, model_registry, mock_mlflow_client):
        """Test health check when MLflow is unhealthy"""
        mock_mlflow_client.search_experiments.side_effect = Exception("Connection failed")
        
        health = await model_registry.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_get_model_info_with_version(self, model_registry, mock_mlflow_client):
        """Test getting model info for specific version"""
        model_name = "test_model"
        version = "1"
        
        # Mock model version
        mock_version = Mock()
        mock_version.current_stage = "Production"
        mock_version.run_id = "run_1"
        mock_version.creation_timestamp = 1234567890
        mock_version.description = "Test version"
        
        # Mock run data
        mock_run = Mock()
        mock_run.data.metrics = {"accuracy": 0.85}
        mock_run.data.params = {"C": 1.0}
        mock_run.data.tags = {"env": "prod"}
        
        mock_mlflow_client.get_model_version.return_value = mock_version
        mock_mlflow_client.get_run.return_value = mock_run
        
        info = await model_registry.get_model_info(model_name, version)
        
        assert info["name"] == model_name
        assert info["version"] == version
        assert info["stage"] == "Production"
        assert info["metrics"]["accuracy"] == 0.85
        assert info["params"]["C"] == 1.0
    
    @pytest.mark.asyncio
    async def test_load_model_error_handling(self, model_registry):
        """Test error handling in load_model"""
        with patch('backend.app.services.model_registry.mlflow.sklearn.load_model') as mock_load:
            mock_load.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception, match="Model not found"):
                await model_registry.load_model("nonexistent_model")
    
    @pytest.mark.asyncio
    async def test_log_model_error_handling(self, model_registry, sample_model):
        """Test error handling in log_model"""
        with patch('backend.app.services.model_registry.mlflow.start_run') as mock_start_run:
            mock_start_run.side_effect = Exception("MLflow server unavailable")
            
            with pytest.raises(Exception, match="MLflow server unavailable"):
                await model_registry.log_model(
                    model=sample_model,
                    model_name="test_model",
                    metrics={"accuracy": 0.85},
                    params={"C": 1.0}
                )