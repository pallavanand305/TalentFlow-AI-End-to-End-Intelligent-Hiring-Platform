"""Prediction log model for monitoring and drift detection"""

from sqlalchemy import Column, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from backend.app.models.base import Base


class PredictionLog(Base):
    """Model for storing prediction logs for monitoring and drift detection"""
    
    __tablename__ = "prediction_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Reference to the score that was created
    score_id = Column(UUID(as_uuid=True), ForeignKey("scores.id"), nullable=True)
    
    # Model information
    model_name = Column(String(255), nullable=False)
    model_version = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # "tfidf", "semantic", etc.
    
    # Input data (features used for prediction)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=False)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)
    
    # Input features (JSON containing extracted features)
    input_features = Column(JSON, nullable=False)
    
    # Prediction outputs
    overall_score = Column(Float, nullable=False)
    section_scores = Column(JSON, nullable=False)  # skills, experience, education scores
    
    # Metadata
    prediction_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    processing_time_ms = Column(Float, nullable=True)  # Time taken for prediction
    
    # Environment information
    environment = Column(String(50), default="production", nullable=False)  # dev, staging, production
    
    # Relationships
    candidate = relationship("Candidate", back_populates="prediction_logs")
    job = relationship("Job", back_populates="prediction_logs")
    score = relationship("Score", back_populates="prediction_log")
    
    def __repr__(self):
        return f"<PredictionLog(id={self.id}, candidate_id={self.candidate_id}, job_id={self.job_id}, score={self.overall_score:.3f})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': str(self.id),
            'score_id': str(self.score_id) if self.score_id else None,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_type': self.model_type,
            'candidate_id': str(self.candidate_id),
            'job_id': str(self.job_id),
            'input_features': self.input_features,
            'overall_score': self.overall_score,
            'section_scores': self.section_scores,
            'prediction_timestamp': self.prediction_timestamp,
            'processing_time_ms': self.processing_time_ms,
            'environment': self.environment
        }