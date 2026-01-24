"""MLflow model registry service for model versioning and tracking"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.models.model_version import ModelVersion
from backend.app.repositories.model_version_repository import ModelVersionRepository

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Service for managing ML models with MLflow integration"""
    
    def __init__(self):
        """Initialize MLflow client and repository"""
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        self.repository = ModelVersionRepository()
        
        logger.info(f"Initialized ModelRegistry with tracking URI: {settings.MLFLOW_TRACKING_URI}")
    
    async def log_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None,
        db_session: Optional[AsyncSession] = None
    ) -> str:
        """
        Log model to MLflow with metrics and parameters.
        
        Args:
            model: The trained model object
            model_name: Name of the model
            metrics: Dictionary of metrics (accuracy, f1_score, etc.)
            params: Dictionary of hyperparameters
            artifacts: Optional dictionary of artifact paths
            db_session: Database session for storing model version
            
        Returns:
            str: MLflow run ID
        """
        try:
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log artifacts if provided
                if artifacts:
                    for artifact_name, artifact_path in artifacts.items():
                        mlflow.log_artifact(artifact_path, artifact_name)
                
                # Log the model
                if hasattr(model, 'fit'):  # sklearn-like model
                    mlflow.sklearn.log_model(model, "model")
                else:
                    # For other model types, try to import pytorch dynamically
                    try:
                        import mlflow.pytorch
                        mlflow.pytorch.log_model(model, "model")
                    except ImportError:
                        # Fallback to generic logging
                        mlflow.log_artifact(str(model), "model")
                
                run_id = run.info.run_id
                
                # Store model version in database
                if db_session:
                    await self._store_model_version(
                        model_name=model_name,
                        version=run_id[:8],  # Use first 8 chars of run_id as version
                        mlflow_run_id=run_id,
                        metrics=metrics,
                        params=params,
                        db_session=db_session
                    )
                
                logger.info(f"Successfully logged model {model_name} with run_id: {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to log model {model_name}: {str(e)}")
            raise
    
    async def load_model(
        self,
        model_name: str,
        version: str = "latest",
        stage: Optional[str] = None
    ) -> Any:
        """
        Load model from MLflow registry.
        
        Args:
            model_name: Name of the model
            version: Version to load ("latest", specific version, or run_id)
            stage: Stage to load from ("Production", "Staging", "None")
            
        Returns:
            Loaded model object
        """
        try:
            if stage:
                # Load by stage
                model_uri = f"models:/{model_name}/{stage}"
            elif version == "latest":
                # Get latest version
                latest_version = self.client.get_latest_versions(
                    model_name, stages=["Production", "Staging", "None"]
                )[0]
                model_uri = f"models:/{model_name}/{latest_version.version}"
            else:
                # Load specific version
                model_uri = f"models:/{model_name}/{version}"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Successfully loaded model {model_name} version {version}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} version {version}: {str(e)}")
            raise
    
    async def register_model(
        self,
        run_id: str,
        model_name: str,
        description: Optional[str] = None
    ) -> str:
        """
        Register a model from an MLflow run.
        
        Args:
            run_id: MLflow run ID containing the model
            model_name: Name to register the model under
            description: Optional description
            
        Returns:
            str: Model version
        """
        try:
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                description=description
            )
            
            logger.info(f"Registered model {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {str(e)}")
            raise
    
    async def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production",
        archive_existing: bool = True
    ) -> bool:
        """
        Promote model version to specified stage.
        
        Args:
            model_name: Name of the model
            version: Version to promote
            stage: Target stage ("Production", "Staging", "Archived")
            archive_existing: Whether to archive existing models in target stage
            
        Returns:
            bool: Success status
        """
        try:
            # Archive existing models in the target stage if requested
            if archive_existing and stage in ["Production", "Staging"]:
                existing_models = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
                for model in existing_models:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model.version,
                        stage="Archived"
                    )
            
            # Promote the new model
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Promoted model {model_name} version {version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model {model_name} version {version}: {str(e)}")
            return False
    
    async def compare_models(
        self,
        model_name: str,
        versions: List[str]
    ) -> pd.DataFrame:
        """
        Compare metrics across model versions.
        
        Args:
            model_name: Name of the model
            versions: List of versions to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        try:
            comparison_data = []
            
            for version in versions:
                try:
                    model_version = self.client.get_model_version(model_name, version)
                    run = self.client.get_run(model_version.run_id)
                    
                    row = {
                        'version': version,
                        'run_id': model_version.run_id,
                        'stage': model_version.current_stage,
                        'created_at': model_version.creation_timestamp,
                        **run.data.metrics,
                        **run.data.params
                    }
                    comparison_data.append(row)
                    
                except Exception as e:
                    logger.warning(f"Could not get data for version {version}: {str(e)}")
                    continue
            
            df = pd.DataFrame(comparison_data)
            logger.info(f"Generated comparison for {len(comparison_data)} versions of {model_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to compare models for {model_name}: {str(e)}")
            return pd.DataFrame()
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = self.client.search_registered_models()
            model_list = []
            
            for model in models:
                latest_versions = self.client.get_latest_versions(
                    model.name, stages=["Production", "Staging", "None"]
                )
                
                model_info = {
                    'name': model.name,
                    'description': model.description,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'latest_versions': [
                        {
                            'version': v.version,
                            'stage': v.current_stage,
                            'run_id': v.run_id
                        }
                        for v in latest_versions
                    ]
                }
                model_list.append(model_info)
            
            logger.info(f"Listed {len(model_list)} registered models")
            return model_list
            
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
    
    async def get_model_info(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            version: Specific version (optional)
            
        Returns:
            Dictionary with model information
        """
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
                run = self.client.get_run(model_version.run_id)
                
                return {
                    'name': model_name,
                    'version': version,
                    'stage': model_version.current_stage,
                    'run_id': model_version.run_id,
                    'created_at': model_version.creation_timestamp,
                    'description': model_version.description,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                }
            else:
                # Get model info without specific version
                model = self.client.get_registered_model(model_name)
                latest_versions = self.client.get_latest_versions(
                    model_name, stages=["Production", "Staging", "None"]
                )
                
                return {
                    'name': model_name,
                    'description': model.description,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'latest_versions': [
                        {
                            'version': v.version,
                            'stage': v.current_stage,
                            'run_id': v.run_id
                        }
                        for v in latest_versions
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {str(e)}")
            return {}
    
    async def _store_model_version(
        self,
        model_name: str,
        version: str,
        mlflow_run_id: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        db_session: AsyncSession
    ) -> ModelVersion:
        """Store model version information in database"""
        try:
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                mlflow_run_id=mlflow_run_id,
                stage="None",
                metrics=metrics,
                params=params,
                artifact_path=f"{settings.MLFLOW_ARTIFACT_ROOT}/{mlflow_run_id}/artifacts"
            )
            
            return await self.repository.create(model_version, db_session)
            
        except Exception as e:
            logger.error(f"Failed to store model version in database: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check MLflow server health.
        
        Returns:
            Dictionary with health status
        """
        try:
            # Try to list experiments to check connectivity
            experiments = self.client.search_experiments(view_type=ViewType.ALL)
            
            return {
                'status': 'healthy',
                'tracking_uri': settings.MLFLOW_TRACKING_URI,
                'experiments_count': len(experiments),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"MLflow health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'tracking_uri': settings.MLFLOW_TRACKING_URI,
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }


# Global model registry instance
model_registry = ModelRegistry()