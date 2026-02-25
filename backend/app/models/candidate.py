"""Candidate model"""

from sqlalchemy import Column, String, Integer, ARRAY, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.models.base import TimestampMixin
import uuid


class Candidate(Base, TimestampMixin):
    """Candidate model for storing resume data"""
    
    __tablename__ = "candidates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    resume_file_path = Column(String(500), nullable=False)  # S3 path
    parsed_data = Column(JSONB, nullable=True)  # Structured resume data
    skills = Column(ARRAY(String), nullable=True, index=True)
    experience_years = Column(Integer, nullable=True)
    education_level = Column(String(100), nullable=True)
    
    # Relationships
    scores = relationship("Score", back_populates="candidate", cascade="all, delete-orphan")
    prediction_logs = relationship("PredictionLog", back_populates="candidate", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Candidate(id={self.id}, name={self.name}, email={self.email})>"
