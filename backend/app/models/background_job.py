"""Background job tracking"""

from sqlalchemy import Column, String, Text, DateTime, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from backend.app.core.database import Base
from backend.app.models.base import TimestampMixin
import uuid
import enum


class BackgroundJobStatus(str, enum.Enum):
    """Background job status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundJob(Base, TimestampMixin):
    """Background job for async task tracking"""
    
    __tablename__ = "background_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(100), nullable=False, index=True)  # resume_parsing, batch_scoring
    status = Column(SQLEnum(BackgroundJobStatus), nullable=False, default=BackgroundJobStatus.QUEUED, index=True)
    input_data = Column(JSONB, nullable=True)
    result_data = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<BackgroundJob(id={self.id}, job_type={self.job_type}, status={self.status})>"
