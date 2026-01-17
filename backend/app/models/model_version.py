"""Model version tracking"""

from sqlalchemy import Column, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from backend.app.core.database import Base
from backend.app.models.base import TimestampMixin
import uuid


class ModelVersion(Base, TimestampMixin):
    """Model version for MLflow tracking"""
    
    __tablename__ = "model_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    mlflow_run_id = Column(String(255), nullable=True)
    stage = Column(String(50), nullable=False, default="None")  # None, Staging, Production
    metrics = Column(JSONB, nullable=True)
    params = Column(JSONB, nullable=True)
    artifact_path = Column(String(500), nullable=False)  # S3 path
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('model_name', 'version', name='uq_model_name_version'),
    )
    
    def __repr__(self):
        return f"<ModelVersion(id={self.id}, model_name={self.model_name}, version={self.version}, stage={self.stage})>"
