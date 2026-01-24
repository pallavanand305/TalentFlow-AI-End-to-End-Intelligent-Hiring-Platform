"""Pydantic schemas for model management"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class ModelVersionInfo(BaseModel):
    """Model version information"""
    version: str = Field(..., description="Model version identifier")
    stage: str = Field(..., description="Model stage (None, Staging, Production, Archived)")
    run_id: str = Field(..., description="MLflow run ID")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class ModelListResponse(BaseModel):
    """Response schema for listing models"""
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    creation_timestamp: Optional[datetime] = Field(None, description="Model creation timestamp")
    last_updated_timestamp: Optional[datetime] = Field(None, description="Last update timestamp")
    latest_versions: List[ModelVersionInfo] = Field(default_factory=list, description="Latest versions by stage")


class ModelDetailResponse(BaseModel):
    """Response schema for model details"""
    name: str = Field(..., description="Model name")
    version: Optional[str] = Field(None, description="Model version")
    stage: Optional[str] = Field(None, description="Model stage")
    run_id: Optional[str] = Field(None, description="MLflow run ID")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    description: Optional[str] = Field(None, description="Model description")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Model performance metrics")
    params: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    tags: Dict[str, str] = Field(default_factory=dict, description="Model tags")
    latest_versions: Optional[List[ModelVersionInfo]] = Field(None, description="Latest versions (for model-level details)")


class ModelPromoteRequest(BaseModel):
    """Request schema for promoting a model"""
    model_name: str = Field(..., description="Name of the model to promote")
    version: str = Field(..., description="Version to promote")
    stage: str = Field(default="Production", description="Target stage (Production, Staging, Archived)")
    archive_existing: bool = Field(default=True, description="Whether to archive existing models in target stage")


class ModelPromoteResponse(BaseModel):
    """Response schema for model promotion"""
    model_name: str = Field(..., description="Name of the promoted model")
    version: str = Field(..., description="Promoted version")
    stage: str = Field(..., description="New stage")
    success: bool = Field(..., description="Whether promotion was successful")
    message: str = Field(..., description="Status message")


class ModelComparisonRequest(BaseModel):
    """Request schema for comparing models"""
    model_name: str = Field(..., description="Name of the model")
    versions: List[str] = Field(..., min_length=2, description="List of versions to compare")


class ModelComparisonResponse(BaseModel):
    """Response schema for model comparison"""
    model_name: str = Field(..., description="Name of the model")
    comparison_data: List[Dict[str, Any]] = Field(..., description="Comparison data for each version")


class ModelHealthResponse(BaseModel):
    """Response schema for MLflow health check"""
    status: str = Field(..., description="Health status (healthy/unhealthy)")
    tracking_uri: str = Field(..., description="MLflow tracking URI")
    experiments_count: Optional[int] = Field(None, description="Number of experiments")
    error: Optional[str] = Field(None, description="Error message if unhealthy")
    timestamp: str = Field(..., description="Health check timestamp")


class ModelRegistrationRequest(BaseModel):
    """Request schema for registering a model"""
    run_id: str = Field(..., description="MLflow run ID containing the model")
    model_name: str = Field(..., description="Name to register the model under")
    description: Optional[str] = Field(None, description="Optional model description")


class ModelRegistrationResponse(BaseModel):
    """Response schema for model registration"""
    model_name: str = Field(..., description="Registered model name")
    version: str = Field(..., description="Registered model version")
    run_id: str = Field(..., description="MLflow run ID")
    success: bool = Field(..., description="Whether registration was successful")
    message: str = Field(..., description="Status message")