"""Score schemas for API requests and responses"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from decimal import Decimal

from pydantic import BaseModel, Field, validator


class ScoringRequest(BaseModel):
    """Request schema for scoring a candidate against a job"""
    candidate_id: UUID = Field(..., description="UUID of the candidate to score")
    job_id: UUID = Field(..., description="UUID of the job to score against")
    generate_explanation: bool = Field(
        default=False, 
        description="Whether to generate natural language explanation for the score"
    )
    detailed_explanation: bool = Field(
        default=False,
        description="Whether to generate detailed explanation with section analysis"
    )
    force_rescore: bool = Field(
        default=False,
        description="Whether to force rescoring even if a current score exists"
    )


class BatchScoringRequest(BaseModel):
    """Request schema for batch scoring multiple candidates"""
    candidate_ids: List[UUID] = Field(..., description="List of candidate UUIDs to score")
    job_id: UUID = Field(..., description="UUID of the job to score against")
    generate_explanations: bool = Field(
        default=False,
        description="Whether to generate explanations for all scores"
    )
    force_rescore: bool = Field(
        default=False,
        description="Whether to force rescoring even if current scores exist"
    )


class ScoringResponse(BaseModel):
    """Response schema for scoring operations"""
    score_id: UUID = Field(..., description="Unique identifier for this score")
    candidate_id: UUID = Field(..., description="UUID of the scored candidate")
    job_id: UUID = Field(..., description="UUID of the job scored against")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score between 0.0 and 1.0")
    model_version: str = Field(..., description="Version of the ML model used for scoring")
    explanation: Optional[str] = Field(None, description="Natural language explanation of the score")
    is_current: bool = Field(..., description="Whether this is the current/active score")
    created_at: datetime = Field(..., description="Timestamp when the score was computed")
    
    @classmethod
    def from_score(cls, score) -> "ScoringResponse":
        """Create response from Score model"""
        return cls(
            score_id=score.id,
            candidate_id=score.candidate_id,
            job_id=score.job_id,
            score=float(score.score),
            model_version=score.model_version,
            explanation=score.explanation,
            is_current=score.is_current,
            created_at=score.created_at
        )


class ScoreDetailResponse(BaseModel):
    """Detailed score response with additional metadata"""
    score_id: UUID = Field(..., description="Unique identifier for this score")
    candidate_id: UUID = Field(..., description="UUID of the scored candidate")
    job_id: UUID = Field(..., description="UUID of the job scored against")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score between 0.0 and 1.0")
    model_version: str = Field(..., description="Version of the ML model used for scoring")
    explanation: Optional[str] = Field(None, description="Natural language explanation of the score")
    is_current: bool = Field(..., description="Whether this is the current/active score")
    created_at: datetime = Field(..., description="Timestamp when the score was computed")
    updated_at: datetime = Field(..., description="Timestamp when the score was last updated")
    
    # Additional metadata
    candidate_name: Optional[str] = Field(None, description="Name of the candidate")
    job_title: Optional[str] = Field(None, description="Title of the job")
    
    @classmethod
    def from_score(cls, score) -> "ScoreDetailResponse":
        """Create detailed response from Score model"""
        return cls(
            score_id=score.id,
            candidate_id=score.candidate_id,
            job_id=score.job_id,
            score=float(score.score),
            model_version=score.model_version,
            explanation=score.explanation,
            is_current=score.is_current,
            created_at=score.created_at,
            updated_at=score.updated_at,
            candidate_name=score.candidate.name if hasattr(score, 'candidate') and score.candidate else None,
            job_title=score.job.title if hasattr(score, 'job') and score.job else None
        )


class CandidateRankingResponse(BaseModel):
    """Response schema for candidate rankings"""
    candidate_id: UUID = Field(..., description="UUID of the candidate")
    candidate_name: str = Field(..., description="Name of the candidate")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Rank position (1 = highest score)")
    explanation: Optional[str] = Field(None, description="Score explanation if available")
    created_at: datetime = Field(..., description="When the score was computed")
    
    # Additional candidate metadata
    skills: Optional[List[str]] = Field(None, description="Candidate skills")
    experience_years: Optional[int] = Field(None, description="Years of experience")
    education_level: Optional[str] = Field(None, description="Education level")


class BatchScoringResponse(BaseModel):
    """Response schema for batch scoring operations"""
    job_id: UUID = Field(..., description="UUID of the job scored against")
    total_candidates: int = Field(..., description="Total number of candidates requested")
    successful_scores: int = Field(..., description="Number of successfully computed scores")
    failed_scores: int = Field(..., description="Number of failed scoring attempts")
    scores: List[ScoringResponse] = Field(..., description="List of computed scores")
    errors: Optional[List[str]] = Field(None, description="List of error messages for failed scores")


class JobScoringStatsResponse(BaseModel):
    """Response schema for job scoring statistics"""
    job_id: UUID = Field(..., description="UUID of the job")
    job_title: str = Field(..., description="Title of the job")
    total_candidates: int = Field(..., description="Total number of candidates scored")
    average_score: Optional[float] = Field(None, description="Average score across all candidates")
    median_score: Optional[float] = Field(None, description="Median score across all candidates")
    min_score: Optional[float] = Field(None, description="Minimum score")
    max_score: Optional[float] = Field(None, description="Maximum score")
    score_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Score distribution in ranges (e.g., '0.0-0.2': 5)"
    )
    top_candidates_count: int = Field(..., description="Number of candidates with score >= 0.7")
    last_updated: Optional[datetime] = Field(None, description="When statistics were last computed")


class CandidateComparisonResponse(BaseModel):
    """Response schema for comparing candidates"""
    job_id: UUID = Field(..., description="UUID of the job being compared against")
    job_title: str = Field(..., description="Title of the job")
    candidates: List[CandidateRankingResponse] = Field(..., description="List of compared candidates")
    comparison_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics for the comparison"
    )


class ScoreExplanationRequest(BaseModel):
    """Request schema for generating score explanations"""
    score_id: UUID = Field(..., description="UUID of the score to explain")
    detailed: bool = Field(
        default=False,
        description="Whether to generate detailed explanation with section breakdowns"
    )


class ScoreExplanationResponse(BaseModel):
    """Response schema for score explanations"""
    score_id: UUID = Field(..., description="UUID of the score")
    score: float = Field(..., ge=0.0, le=1.0, description="The score being explained")
    explanation: str = Field(..., description="Natural language explanation")
    section_scores: Optional[Dict[str, float]] = Field(
        None,
        description="Breakdown of scores by resume section"
    )
    key_matches: Optional[List[str]] = Field(
        None,
        description="Key matching elements between candidate and job"
    )
    improvement_suggestions: Optional[List[str]] = Field(
        None,
        description="Suggestions for improving the match"
    )


class TopCandidatesRequest(BaseModel):
    """Request schema for getting top candidates"""
    job_id: UUID = Field(..., description="UUID of the job")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of candidates to return")
    min_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold"
    )
    include_explanations: bool = Field(
        default=False,
        description="Whether to include score explanations"
    )


class CandidateJobMatchRequest(BaseModel):
    """Request schema for finding job matches for a candidate"""
    candidate_id: UUID = Field(..., description="UUID of the candidate")
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum score threshold")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of jobs to return")


class CandidateJobMatchResponse(BaseModel):
    """Response schema for candidate job matches"""
    candidate_id: UUID = Field(..., description="UUID of the candidate")
    candidate_name: str = Field(..., description="Name of the candidate")
    matches: List[Dict[str, Any]] = Field(..., description="List of job matches with scores")
    total_matches: int = Field(..., description="Total number of matching jobs")


class RescoreJobRequest(BaseModel):
    """Request schema for rescoring all candidates for a job"""
    job_id: UUID = Field(..., description="UUID of the job to rescore")
    generate_explanations: bool = Field(
        default=False,
        description="Whether to generate explanations for all new scores"
    )


class RescoreJobResponse(BaseModel):
    """Response schema for job rescoring operations"""
    job_id: UUID = Field(..., description="UUID of the job")
    job_title: str = Field(..., description="Title of the job")
    candidates_rescored: int = Field(..., description="Number of candidates successfully rescored")
    candidates_failed: int = Field(..., description="Number of candidates that failed rescoring")
    message: str = Field(..., description="Summary message")
    started_at: datetime = Field(..., description="When rescoring started")
    completed_at: Optional[datetime] = Field(None, description="When rescoring completed")


# Validation helpers
class ScoreValidators:
    """Common validators for score-related schemas"""
    
    @staticmethod
    def validate_score_range(v: float) -> float:
        """Validate score is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v
    
    @staticmethod
    def validate_candidate_list(v: List[UUID]) -> List[UUID]:
        """Validate candidate list is not empty"""
        if not v:
            raise ValueError("Candidate list cannot be empty")
        if len(v) > 100:
            raise ValueError("Cannot score more than 100 candidates at once")
        return v