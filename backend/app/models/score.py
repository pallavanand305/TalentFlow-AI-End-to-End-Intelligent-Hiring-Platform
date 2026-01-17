"""Score model"""

from sqlalchemy import Column, String, DECIMAL, Boolean, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.models.base import TimestampMixin
import uuid


class Score(Base, TimestampMixin):
    """Score model for candidate-job matching"""
    
    __tablename__ = "scores"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=False, index=True)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, index=True)
    score = Column(DECIMAL(5, 4), nullable=False)  # 0.0000 to 1.0000
    model_version = Column(String(100), nullable=False)
    explanation = Column(Text, nullable=True)
    is_current = Column(Boolean, default=True, nullable=False, index=True)
    
    # Relationships
    candidate = relationship("Candidate", back_populates="scores")
    job = relationship("Job", back_populates="scores")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('candidate_id', 'job_id', 'model_version', name='uq_candidate_job_model'),
    )
    
    def __repr__(self):
        return f"<Score(id={self.id}, candidate_id={self.candidate_id}, job_id={self.job_id}, score={self.score})>"
