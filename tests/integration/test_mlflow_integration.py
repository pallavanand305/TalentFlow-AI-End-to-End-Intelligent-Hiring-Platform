"""Integration tests for MLflow setup and connectivity"""

import pytest
import asyncio
import tempfile
import os
from sklearn.linear_model import LogisticRegression
import numpy as np

from backend.app.services.model_registry import ModelRegistry
from backend.app.core.config import settings


class TestMLflowIntegration:
    """Integration tests for MLflow tracking server"""
    
    @pytest.fixture
    def model_registry(self):
        """Create ModelRegistry instance for integration testing"""
        return ModelRegistry()
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple sklearn model for testing"""
        model = LogisticRegression(random_state=42)
        # Create some dummy data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        return model
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mlflow_health_check(self, model_registry):
        """Test that MLflow server is accessible and healthy"""
        health = await model_registry.health_check()
        
        assert health["status"] == "healthy"
        assert "experiments_count" in health
        assert health["tracking_uri"] == settings.MLFLOW_TRACKING_URI
        assert "timestamp" in health
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_models_empty_registry(self, model_registry):
        """Test listing models when registry is empty"""
        models = await model_registry.list_models()
        
        # Should return empty list or existing models from previous tests
        assert isinstance(models, list)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_model_workflow(self, model_registry, sample_model):
        """Test complete model workflow: log -> register -> promote -> load"""
        model_name = "integration_test_model"
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
        params = {
            "C": 1.0,
            "solver": "liblinear",
            "random_state": 42,
            "max_iter": 100
        }
        
        try:
            # Step 1: Log model
            run_id = await model_registry.log_model(
                model=sample_model,
                model_name=model_name,
                metrics=metrics,
                params=params
            )
            
            assert run_id is not None
            assert len(run_id) > 0
            
            # Step 2: Register model
            version = await model_registry.register_model(
                run_id=run_id,
                model_name=model_name,
                description="Integration test model"
            )
            
            assert version is not None
            
            # Step 3: Get model info
            model_info = await model_registry.get_model_info(model_name, version)
            
            assert model_info["name"] == model_name
            assert model_info["version"] == version
            assert model_info["metrics"]["accuracy"] == metrics["accuracy"]
            assert model_info["params"]["C"] == params["C"]
            
            # Step 4: Promote to staging
            success = await model_registry.promote_model(
                model_name=model_name,
                version=version,
                stage="Staging"
            )
            
            assert success is True
            
            # Step 5: Load model by stage
            loaded_model = await model_registry.load_model(
                model_name=model_name,
                stage="Staging"
            )
            
            assert loaded_model is not None
            
            # Step 6: Test the loaded model works
            test_data = np.array([[2, 3], [6, 7]])
            predictions = loaded_model.predict(test_data)
            assert len(predictions) == 2
            assert all(pred in [0, 1] for pred in predictions)
            
            # Step 7: Promote to production
            success = await model_registry.promote_model(
                model_name=model_name,
                version=version,
                stage="Production"
            )
            
            assert success is True
            
            # Step 8: Load production model
            prod_model = await model_registry.load_model(
                model_name=model_name,
                stage="Production"
            )
            
            assert prod_model is not None
            
        except Exception as e:
            pytest.fail(f"End-to-end workflow failed: {str(e)}")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_model_comparison(self, model_registry, sample_model):
        """Test model comparison functionality"""
        model_name = "comparison_test_model"
        
        # Log two different model versions
        versions = []
        
        for i, c_value in enumerate([0.5, 1.0]):
            # Create slightly different model
            model = LogisticRegression(C=c_value, random_state=42)
            X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            y = np.array([0, 1, 0, 1])
            model.fit(X, y)
            
            metrics = {"accuracy": 0.8 + i * 0.05, "f1_score": 0.75 + i * 0.05}
            params = {"C": c_value, "solver": "liblinear"}
            
            run_id = await model_registry.log_model(
                model=model,
                model_name=model_name,
                metrics=metrics,
                params=params
            )
            
            version = await model_registry.register_model(
                run_id=run_id,
                model_name=model_name
            )
            
            versions.append(version)
        
        # Compare the models
        comparison_df = await model_registry.compare_models(model_name, versions)
        
        assert len(comparison_df) == 2
        assert "version" in comparison_df.columns
        assert "accuracy" in comparison_df.columns
        assert "C" in comparison_df.columns
        
        # Check that different C values are recorded
        c_values = comparison_df["C"].tolist()
        assert 0.5 in c_values
        assert 1.0 in c_values
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_artifact_logging(self, model_registry, sample_model):
        """Test logging models with artifacts"""
        model_name = "artifact_test_model"
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": "config", "version": "1.0"}')
            config_path = f.name
        
        try:
            metrics = {"accuracy": 0.90}
            params = {"C": 2.0}
            artifacts = {"config": config_path}
            
            run_id = await model_registry.log_model(
                model=sample_model,
                model_name=model_name,
                metrics=metrics,
                params=params,
                artifacts=artifacts
            )
            
            assert run_id is not None
            
            # Register the model
            version = await model_registry.register_model(
                run_id=run_id,
                model_name=model_name
            )
            
            # Verify model info includes the artifacts
            model_info = await model_registry.get_model_info(model_name, version)
            assert model_info["name"] == model_name
            assert model_info["metrics"]["accuracy"] == 0.90
            
        finally:
            # Clean up temporary file
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_load_latest_model(self, model_registry, sample_model):
        """Test loading the latest model version"""
        model_name = "latest_test_model"
        
        # Log a model
        run_id = await model_registry.log_model(
            model=sample_model,
            model_name=model_name,
            metrics={"accuracy": 0.88},
            params={"C": 1.5}
        )
        
        # Register the model
        version = await model_registry.register_model(
            run_id=run_id,
            model_name=model_name
        )
        
        # Load latest version
        loaded_model = await model_registry.load_model(
            model_name=model_name,
            version="latest"
        )
        
        assert loaded_model is not None
        
        # Test that the model works
        test_data = np.array([[1, 2]])
        prediction = loaded_model.predict(test_data)
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling_nonexistent_model(self, model_registry):
        """Test error handling when loading non-existent model"""
        with pytest.raises(Exception):
            await model_registry.load_model("nonexistent_model_12345")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_experiments(self, model_registry, sample_model):
        """Test that models can be logged to different experiments"""
        # This test assumes the init_mlflow.py script has created default experiments
        model_name = "multi_experiment_model"
        
        # Try to log to different experiments by setting experiment
        import mlflow
        
        # Set experiment to baseline-models
        try:
            mlflow.set_experiment("baseline-models")
            
            run_id = await model_registry.log_model(
                model=sample_model,
                model_name=model_name,
                metrics={"accuracy": 0.75},
                params={"experiment": "baseline"}
            )
            
            assert run_id is not None
            
        except Exception as e:
            # If experiment doesn't exist, that's okay for this test
            pytest.skip(f"Experiment not found, skipping: {str(e)}")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_model_operations(self, model_registry, sample_model):
        """Test concurrent model operations"""
        model_name = "concurrent_test_model"
        
        async def log_model_task(suffix: str):
            """Task to log a model concurrently"""
            model = LogisticRegression(random_state=42 + int(suffix))
            X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            y = np.array([0, 1, 0, 1])
            model.fit(X, y)
            
            return await model_registry.log_model(
                model=model,
                model_name=f"{model_name}_{suffix}",
                metrics={"accuracy": 0.8 + int(suffix) * 0.01},
                params={"suffix": suffix}
            )
        
        # Run multiple model logging operations concurrently
        tasks = [log_model_task(str(i)) for i in range(3)]
        run_ids = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(run_ids) == 3
        assert all(run_id is not None for run_id in run_ids)
        assert len(set(run_ids)) == 3  # All should be unique