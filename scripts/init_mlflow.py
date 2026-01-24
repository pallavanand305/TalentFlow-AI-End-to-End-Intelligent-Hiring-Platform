#!/usr/bin/env python3
"""Initialize MLflow tracking server and create default experiments"""

import os
import sys
import logging
import mlflow
from mlflow.tracking import MlflowClient

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_mlflow():
    """Initialize MLflow with default experiments and settings"""
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        
        logger.info(f"Connecting to MLflow at: {settings.MLFLOW_TRACKING_URI}")
        
        # Create default experiments if they don't exist
        experiments = [
            {
                "name": "resume-parsing-models",
                "description": "Experiments for resume parsing and NER models"
            },
            {
                "name": "scoring-models", 
                "description": "Experiments for candidate-job similarity scoring models"
            },
            {
                "name": "baseline-models",
                "description": "Baseline TF-IDF and simple similarity models"
            },
            {
                "name": "semantic-models",
                "description": "Advanced semantic similarity models using transformers"
            }
        ]
        
        for exp in experiments:
            try:
                experiment = client.get_experiment_by_name(exp["name"])
                if experiment is None:
                    experiment_id = client.create_experiment(
                        name=exp["name"],
                        artifact_location=f"{settings.MLFLOW_ARTIFACT_ROOT}/{exp['name']}"
                    )
                    logger.info(f"Created experiment: {exp['name']} (ID: {experiment_id})")
                else:
                    logger.info(f"Experiment already exists: {exp['name']} (ID: {experiment.experiment_id})")
            except Exception as e:
                logger.warning(f"Could not create experiment {exp['name']}: {str(e)}")
        
        # Test basic functionality
        logger.info("Testing MLflow connectivity...")
        experiments = client.search_experiments()
        logger.info(f"Found {len(experiments)} experiments")
        
        # Create a test run to verify everything works
        with mlflow.start_run(experiment_id=client.get_experiment_by_name("baseline-models").experiment_id):
            mlflow.log_param("test_param", "initialization")
            mlflow.log_metric("test_metric", 1.0)
            logger.info("Successfully created test run")
        
        logger.info("MLflow initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {str(e)}")
        return False


def health_check():
    """Perform a health check on MLflow server"""
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        
        # Try to list experiments
        experiments = client.search_experiments()
        logger.info(f"MLflow health check passed. Found {len(experiments)} experiments.")
        
        # Print server info
        logger.info(f"Tracking URI: {settings.MLFLOW_TRACKING_URI}")
        logger.info(f"Artifact Root: {settings.MLFLOW_ARTIFACT_ROOT}")
        
        return True
        
    except Exception as e:
        logger.error(f"MLflow health check failed: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MLflow initialization and health check")
    parser.add_argument("--health-check", action="store_true", help="Perform health check only")
    parser.add_argument("--init", action="store_true", help="Initialize MLflow with default experiments")
    
    args = parser.parse_args()
    
    if args.health_check:
        success = health_check()
        sys.exit(0 if success else 1)
    elif args.init:
        success = init_mlflow()
        sys.exit(0 if success else 1)
    else:
        # Default: run both health check and initialization
        logger.info("Running MLflow health check...")
        if health_check():
            logger.info("Running MLflow initialization...")
            success = init_mlflow()
            sys.exit(0 if success else 1)
        else:
            sys.exit(1)