"""Job model"""

from sqlalchemy import Column, String, Text, ARRAY, Enum as SQLEnum, DECIMAL, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.models.base import TimestampMixin
import uuid
import enum


class ExperienceLevel(str, enum.Enum):
    """Experience level enumeration"""
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"


class JobStatus(str, enum.Enum):
    """Job status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CLOSED = "closed"


class Job(Base, TimestampMixin):
    """Job posting model"""
    
    __tablename__ = "jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    required_skills = Column(ARRAY(String), nullable=False)
    experience_level = Column(SQLEnum(ExperienceLevel), nullable=False)
    location = Column(String(255), nullable=True)
    salary_min = Column(DECIMAL(10, 2), nullable=True)
    salary_max = Column(DECIMAL(10, 2), nullable=True)
    status = Column(SQLEnum(JobStatus), nullable=False, default=JobStatus.ACTIVE, index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    creator = relationship("User", backref="jobs")
    history = relationship("JobHistory", back_populates="job", cascade="all, delete-orphan")
    scores = relationship("Score", back_populates="job", cascade="all, delete-orphan")
    prediction_logs = relationship("PredictionLog", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Job(id={self.id}, title={self.title}, status={self.status})>"


class JobHistory(Base):
    """Job history for tracking changes"""
    
    __tablename__ = "job_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, index=True)
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    required_skills = Column(ARRAY(String), nullable=True)
    changed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    changed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    job = relationship("Job", back_populates="history")
    changer = relationship("User")
    
    def __repr__(self):
        return f"<JobHistory(id={self.id}, job_id={self.job_id}, changed_at={self.changed_at})>"
