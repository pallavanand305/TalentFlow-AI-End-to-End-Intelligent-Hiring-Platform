"""Model management API endpoints"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user, require_any_role, require_admin
from backend.app.models.user import User
from backend.app.services.model_registry import model_registry
from backend.app.schemas.model import (
    ModelListResponse,
    ModelDetailResponse,
    ModelPromoteRequest,
    ModelPromoteResponse,
    ModelComparisonRequest,
    ModelComparisonResponse,
    ModelHealthResponse,
    ModelRegistrationRequest,
    ModelRegistrationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "",
    response_model=List[ModelListResponse],
    summary="List all models",
    description="Retrieve a list of all registered models with their latest versions",
    responses={
        200: {"description": "List of models retrieved successfully"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    }
)
async def list_models(
    current_user: User = Depends(require_any_role)
) -> List[ModelListResponse]:
    """
    List all registered models and their versions.
    
    Returns information about all models in the MLflow registry including:
    - Model name and description
    - Creation and last updated timestamps
    - Latest versions by stage (Production, Staging, None)
    
    **Requirements**: 5.2, 5.4, 5.5
    """
    try:
        logger.info(f"User {current_user.username} requesting model list")
        
        models_data = await model_registry.list_models()
        
        models = []
        for model_data in models_data:
            # Convert latest_versions to ModelVersionInfo objects
            latest_versions = [
                {
                    "version": v["version"],
                    "stage": v["stage"],
                    "run_id": v["run_id"],
                    "created_at": None  # MLflow doesn't provide this in list view
                }
                for v in model_data.get("latest_versions", [])
            ]
            
            model_response = ModelListResponse(
                name=model_data["name"],
                description=model_data.get("description"),
                creation_timestamp=model_data.get("creation_timestamp"),
                last_updated_timestamp=model_data.get("last_updated_timestamp"),
                latest_versions=latest_versions
            )
            models.append(model_response)
        
        logger.info(f"Retrieved {len(models)} models for user {current_user.username}")
        return models
        
    except Exception as e:
        logger.error(f"Failed to list models for user {current_user.username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models: {str(e)}"
        )


@router.get(
    "/{model_name}",
    response_model=ModelDetailResponse,
    summary="Get model details",
    description="Retrieve detailed information about a specific model",
    responses={
        200: {"description": "Model details retrieved successfully"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"},
    }
)
async def get_model_details(
    model_name: str,
    version: Optional[str] = Query(None, description="Specific version to retrieve (optional)"),
    current_user: User = Depends(require_any_role)
) -> ModelDetailResponse:
    """
    Get detailed information about a specific model.
    
    If version is provided, returns details for that specific version.
    If version is omitted, returns general model information with all latest versions.
    
    **Requirements**: 5.2, 5.4, 5.5
    """
    try:
        logger.info(f"User {current_user.username} requesting details for model {model_name}, version {version}")
        
        model_info = await model_registry.get_model_info(model_name, version)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        # Convert latest_versions if present (for model-level details)
        latest_versions = None
        if "latest_versions" in model_info:
            latest_versions = [
                {
                    "version": v["version"],
                    "stage": v["stage"],
                    "run_id": v["run_id"],
                    "created_at": None
                }
                for v in model_info["latest_versions"]
            ]
        
        model_detail = ModelDetailResponse(
            name=model_info["name"],
            version=model_info.get("version"),
            stage=model_info.get("stage"),
            run_id=model_info.get("run_id"),
            created_at=model_info.get("created_at"),
            description=model_info.get("description"),
            metrics=model_info.get("metrics", {}),
            params=model_info.get("params", {}),
            tags=model_info.get("tags", {}),
            latest_versions=latest_versions
        )
        
        logger.info(f"Retrieved details for model {model_name} for user {current_user.username}")
        return model_detail
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model details for {model_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model details: {str(e)}"
        )


@router.post(
    "/promote",
    response_model=ModelPromoteResponse,
    summary="Promote model to production",
    description="Promote a model version to a specific stage (typically Production)",
    responses={
        200: {"description": "Model promoted successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Model or version not found"},
        500: {"description": "Internal server error"},
    }
)
async def promote_model(
    request: ModelPromoteRequest,
    current_user: User = Depends(require_admin)  # Only admins can promote models
) -> ModelPromoteResponse:
    """
    Promote a model version to a specific stage.
    
    This endpoint allows promoting models between stages:
    - **Production**: Live models serving production traffic
    - **Staging**: Models ready for testing
    - **Archived**: Deprecated models
    
    By default, existing models in the target stage are archived.
    
    **Requirements**: 5.2, 5.4, 5.5
    """
    try:
        logger.info(
            f"User {current_user.username} promoting model {request.model_name} "
            f"version {request.version} to {request.stage}"
        )
        
        # Validate stage
        valid_stages = ["Production", "Staging", "Archived", "None"]
        if request.stage not in valid_stages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stage '{request.stage}'. Must be one of: {valid_stages}"
            )
        
        # Check if model and version exist
        model_info = await model_registry.get_model_info(request.model_name, request.version)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{request.model_name}' version '{request.version}' not found"
            )
        
        # Promote the model
        success = await model_registry.promote_model(
            model_name=request.model_name,
            version=request.version,
            stage=request.stage,
            archive_existing=request.archive_existing
        )
        
        if success:
            message = f"Successfully promoted {request.model_name} v{request.version} to {request.stage}"
            logger.info(f"Model promotion successful: {message}")
        else:
            message = f"Failed to promote {request.model_name} v{request.version} to {request.stage}"
            logger.error(f"Model promotion failed: {message}")
        
        return ModelPromoteResponse(
            model_name=request.model_name,
            version=request.version,
            stage=request.stage,
            success=success,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to promote model {request.model_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote model: {str(e)}"
        )


@router.post(
    "/compare",
    response_model=ModelComparisonResponse,
    summary="Compare model versions",
    description="Compare metrics and parameters across multiple model versions",
    responses={
        200: {"description": "Model comparison completed successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"},
    }
)
async def compare_models(
    request: ModelComparisonRequest,
    current_user: User = Depends(require_any_role)
) -> ModelComparisonResponse:
    """
    Compare metrics and parameters across multiple model versions.
    
    This endpoint allows comparing different versions of the same model
    to analyze performance improvements and parameter changes.
    
    **Requirements**: 5.4
    """
    try:
        logger.info(
            f"User {current_user.username} comparing model {request.model_name} "
            f"versions {request.versions}"
        )
        
        # Validate that we have at least 2 versions to compare
        if len(request.versions) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 versions are required for comparison"
            )
        
        # Get comparison data
        comparison_df = await model_registry.compare_models(
            model_name=request.model_name,
            versions=request.versions
        )
        
        if comparison_df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for model '{request.model_name}' with specified versions"
            )
        
        # Convert DataFrame to list of dictionaries
        comparison_data = comparison_df.to_dict('records')
        
        logger.info(f"Generated comparison for {len(comparison_data)} versions of {request.model_name}")
        
        return ModelComparisonResponse(
            model_name=request.model_name,
            comparison_data=comparison_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare models for {request.model_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare models: {str(e)}"
        )


@router.post(
    "/register",
    response_model=ModelRegistrationResponse,
    summary="Register a model from MLflow run",
    description="Register a model from an existing MLflow run",
    responses={
        200: {"description": "Model registered successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "MLflow run not found"},
        500: {"description": "Internal server error"},
    }
)
async def register_model(
    request: ModelRegistrationRequest,
    current_user: User = Depends(require_admin)  # Only admins can register models
) -> ModelRegistrationResponse:
    """
    Register a model from an existing MLflow run.
    
    This endpoint allows registering models that were logged in MLflow runs
    into the model registry for versioning and deployment.
    
    **Requirements**: 5.2
    """
    try:
        logger.info(
            f"User {current_user.username} registering model {request.model_name} "
            f"from run {request.run_id}"
        )
        
        # Register the model
        version = await model_registry.register_model(
            run_id=request.run_id,
            model_name=request.model_name,
            description=request.description
        )
        
        message = f"Successfully registered {request.model_name} version {version}"
        logger.info(f"Model registration successful: {message}")
        
        return ModelRegistrationResponse(
            model_name=request.model_name,
            version=version,
            run_id=request.run_id,
            success=True,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Failed to register model {request.model_name}: {str(e)}")
        
        # Check if it's a "not found" type error
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MLflow run '{request.run_id}' not found"
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register model: {str(e)}"
        )


@router.get(
    "/health",
    response_model=ModelHealthResponse,
    summary="Check MLflow health",
    description="Check the health and connectivity of the MLflow tracking server",
    responses={
        200: {"description": "Health check completed"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
    }
)
async def check_mlflow_health(
    current_user: User = Depends(require_any_role)
) -> ModelHealthResponse:
    """
    Check MLflow server health and connectivity.
    
    This endpoint verifies that the MLflow tracking server is accessible
    and returns basic statistics about the registry.
    """
    try:
        logger.info(f"User {current_user.username} checking MLflow health")
        
        health_data = await model_registry.health_check()
        
        return ModelHealthResponse(**health_data)
        
    except Exception as e:
        logger.error(f"MLflow health check failed: {str(e)}")
        return ModelHealthResponse(
            status="unhealthy",
            tracking_uri="unknown",
            error=str(e),
            timestamp=str(e)  # This will be overridden by the actual timestamp
        )