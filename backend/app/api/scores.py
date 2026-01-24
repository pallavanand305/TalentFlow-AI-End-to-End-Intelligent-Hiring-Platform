"""Scoring API endpoints"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user, require_any_role, require_hiring_manager
from backend.app.models.user import User
from backend.app.repositories.score_repository import ScoreRepository
from backend.app.repositories.candidate_repository import CandidateRepository
from backend.app.repositories.job_repository import JobRepository
from backend.app.services.scoring_service import ScoringService
from backend.app.schemas.score import (
    ScoringRequest, ScoringResponse, ScoreDetailResponse,
    CandidateRankingResponse, BatchScoringRequest, BatchScoringResponse,
    JobScoringStatsResponse, CandidateComparisonResponse,
    RescoreJobRequest, RescoreJobResponse, CandidateJobMatchResponse,
    ScoreExplanationRequest, ScoreExplanationResponse
)
from backend.app.core.exceptions import ValidationException, NotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


async def get_scoring_service(db: AsyncSession = Depends(get_db)) -> ScoringService:
    """Dependency to get scoring service"""
    score_repo = ScoreRepository(db)
    candidate_repo = CandidateRepository(db)
    job_repo = JobRepository(db)
    return ScoringService(score_repo, candidate_repo, job_repo)


@router.post("/compute", response_model=ScoringResponse, status_code=status.HTTP_201_CREATED)
async def compute_score(
    request: ScoringRequest,
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Compute candidate-job similarity score
    
    **Requirements:**
    - User must be authenticated
    - Candidate and job must exist
    - Candidate must have parsed resume data
    
    **Process:**
    1. Validates candidate and job exist
    2. Checks if score already exists (returns existing unless force_rescore=true)
    3. Uses ML scoring engine to compute similarity
    4. Optionally generates natural language explanation
    5. Persists score to database
    
    **Returns:**
    - Unique score ID for future reference
    - Overall similarity score (0.0 to 1.0)
    - Section-wise scores (skills, experience, education)
    - Optional explanation text
    - Timestamp and model version used
    
    **Scoring Algorithm:**
    - TF-IDF baseline: Computes cosine similarity between resume and job text
    - Semantic model: Uses sentence transformers for semantic embeddings
    - Weighted combination: Skills (40%), Experience (40%), Education (20%)
    """
    logger.info(f"Score computation request from user {current_user.id}")
    
    try:
        score = await scoring_service.score_candidate_for_job(
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            generate_explanation=request.generate_explanation,
            detailed_explanation=getattr(request, 'detailed_explanation', False),
            force_rescore=request.force_rescore
        )
        
        return ScoringResponse.from_score(score)
        
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Score computation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Score computation failed"
        )


@router.get("/{score_id}", response_model=ScoreDetailResponse)
async def get_score_details(
    score_id: UUID,
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Get detailed score information
    
    **Returns:**
    - Complete score details including metadata
    - Candidate and job information
    - Explanation text (if available)
    - Model version and timestamp
    
    **Use Cases:**
    - Reviewing specific scoring decisions
    - Auditing ML model performance
    - Understanding candidate-job matches
    """
    logger.info(f"Score details request from user {current_user.id} for score {score_id}")
    
    try:
        score = await scoring_service.get_score_by_id(score_id)
        return ScoreDetailResponse.from_score(score)
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Score not found"
        )
    except Exception as e:
        logger.error(f"Failed to get score details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve score details"
        )


@router.post("/batch", response_model=BatchScoringResponse, status_code=status.HTTP_201_CREATED)
async def batch_score_candidates(
    request: BatchScoringRequest,
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Score multiple candidates for a job in batch
    
    **Efficiently processes multiple candidates at once**
    
    **Request Body:**
    - candidate_ids: List of candidate UUIDs (max 100)
    - job_id: Job to score against
    - generate_explanations: Whether to generate explanations
    - force_rescore: Whether to force rescoring existing scores
    
    **Use Cases:**
    - Bulk candidate evaluation
    - Rescoring after job description updates
    - Initial screening of large candidate pools
    
    **Performance:**
    - Processes candidates in parallel where possible
    - Continues processing even if individual candidates fail
    - Returns partial results with error details
    """
    logger.info(f"Batch scoring request from user {current_user.id} for {len(request.candidate_ids)} candidates")
    
    try:
        scores = await scoring_service.batch_score_candidates_for_job(
            candidate_ids=request.candidate_ids,
            job_id=request.job_id,
            generate_explanations=request.generate_explanations,
            force_rescore=request.force_rescore
        )
        
        # Collect any errors (candidates that failed to score)
        successful_ids = {score.candidate_id for score in scores}
        failed_ids = set(request.candidate_ids) - successful_ids
        errors = [f"Failed to score candidate {cid}" for cid in failed_ids] if failed_ids else None
        
        return BatchScoringResponse(
            job_id=request.job_id,
            total_candidates=len(request.candidate_ids),
            successful_scores=len(scores),
            failed_scores=len(failed_ids),
            scores=[ScoringResponse.from_score(score) for score in scores],
            errors=errors
        )
        
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch scoring failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch scoring failed"
        )


@router.get("/jobs/{job_id}/stats", response_model=JobScoringStatsResponse)
async def get_job_scoring_statistics(
    job_id: UUID,
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Get scoring statistics for a job
    
    **Provides comprehensive analytics on candidate scoring**
    
    **Returns:**
    - Total candidates scored
    - Score distribution statistics (mean, median, min, max)
    - Score range distribution
    - High-potential candidate count
    
    **Use Cases:**
    - Job market analysis
    - Recruitment strategy optimization
    - Understanding candidate pool quality
    - Identifying scoring patterns
    """
    logger.info(f"Job statistics request from user {current_user.id} for job {job_id}")
    
    try:
        stats = await scoring_service.get_job_scoring_statistics(job_id)
        
        return JobScoringStatsResponse(
            job_id=UUID(stats['job_id']),
            job_title=stats['job_title'],
            total_candidates=stats.get('total_candidates', 0),
            average_score=stats.get('average_score'),
            median_score=stats.get('median_score'),
            min_score=stats.get('min_score'),
            max_score=stats.get('max_score'),
            score_distribution=stats.get('score_distribution', {}),
            top_candidates_count=stats.get('top_candidates_count', 0),
            last_updated=stats.get('last_updated')
        )
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Failed to get job statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job statistics"
        )


@router.get("/candidates/{candidate_id}/scores", response_model=List[ScoringResponse])
async def get_candidate_scores(
    candidate_id: UUID,
    current_only: bool = Query(True, description="Only return current/active scores"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of scores to return"),
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Get all scores for a specific candidate
    
    **Returns candidate's scores across all jobs**
    
    **Query Parameters:**
    - current_only: Filter to only active scores (default: true)
    - limit: Maximum number of scores to return
    
    **Use Cases:**
    - Candidate profile review
    - Understanding candidate's job market fit
    - Tracking candidate performance over time
    - Identifying best job matches for candidate
    """
    logger.info(f"Candidate scores request from user {current_user.id} for candidate {candidate_id}")
    
    try:
        scores = await scoring_service.get_candidate_scores(
            candidate_id=candidate_id,
            current_only=current_only,
            limit=limit
        )
        
        return [ScoringResponse.from_score(score) for score in scores]
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate not found"
        )
    except Exception as e:
        logger.error(f"Failed to get candidate scores: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve candidate scores"
        )


@router.get("/candidates/{candidate_id}/best-matches", response_model=CandidateJobMatchResponse)
async def get_best_job_matches_for_candidate(
    candidate_id: UUID,
    min_score: float = Query(0.5, ge=0.0, le=1.0, description="Minimum score threshold"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of job matches to return"),
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Find best job matches for a candidate
    
    **Returns jobs where candidate scores highest**
    
    **Query Parameters:**
    - min_score: Only return jobs above this score threshold
    - limit: Maximum number of job matches to return
    
    **Use Cases:**
    - Career guidance for candidates
    - Job recommendation systems
    - Understanding candidate market positioning
    - Identifying mutual best fits
    """
    logger.info(f"Best matches request from user {current_user.id} for candidate {candidate_id}")
    
    try:
        scores = await scoring_service.find_best_jobs_for_candidate(
            candidate_id=candidate_id,
            min_score=min_score,
            limit=limit
        )
        
        # Get candidate info
        candidate = await scoring_service.candidate_repo.get_by_id(candidate_id)
        
        matches = []
        for score in scores:
            match = {
                'job_id': str(score.job_id),
                'job_title': score.job.title if hasattr(score, 'job') and score.job else 'Unknown',
                'score': float(score.score),
                'explanation': score.explanation,
                'created_at': score.created_at
            }
            matches.append(match)
        
        return CandidateJobMatchResponse(
            candidate_id=candidate_id,
            candidate_name=candidate.name if candidate else 'Unknown',
            matches=matches,
            total_matches=len(matches)
        )
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate not found"
        )
    except Exception as e:
        logger.error(f"Failed to get best matches: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve best job matches"
        )


@router.post("/jobs/{job_id}/rescore", response_model=RescoreJobResponse)
async def rescore_all_candidates_for_job(
    job_id: UUID,
    request: RescoreJobRequest,
    current_user: User = Depends(require_hiring_manager),  # Requires higher permissions
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Rescore all candidates for a job
    
    **Recomputes scores for all previously scored candidates**
    
    **Use Cases:**
    - After job description updates
    - After ML model updates
    - Periodic score refresh
    - Data quality improvements
    
    **Process:**
    1. Identifies all candidates previously scored for this job
    2. Invalidates existing scores
    3. Recomputes scores using current ML model
    4. Updates database with new scores
    
    **Note:** This is a potentially expensive operation for jobs with many candidates
    """
    logger.info(f"Rescore request from user {current_user.id} for job {job_id}")
    
    try:
        started_at = datetime.utcnow()
        
        result = await scoring_service.rescore_all_candidates_for_job(
            job_id=job_id,
            generate_explanations=request.generate_explanations
        )
        
        completed_at = datetime.utcnow()
        
        return RescoreJobResponse(
            job_id=job_id,
            job_title=result.get('message', '').split(' for job ')[-1] if 'for job' in result.get('message', '') else 'Unknown',
            candidates_rescored=result['candidates_rescored'],
            candidates_failed=result['candidates_failed'],
            message=result['message'],
            started_at=started_at,
            completed_at=completed_at
        )
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Rescoring failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Rescoring operation failed"
        )


@router.post("/jobs/{job_id}/compare", response_model=CandidateComparisonResponse)
async def compare_candidates_for_job(
    job_id: UUID,
    candidate_ids: List[UUID],
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Compare specific candidates for a job
    
    **Provides side-by-side comparison of selected candidates**
    
    **Request Body:**
    - List of candidate UUIDs to compare (max 10)
    
    **Returns:**
    - Ranked comparison of candidates
    - Individual scores and explanations
    - Summary statistics for the comparison group
    
    **Use Cases:**
    - Final candidate selection
    - Interview panel preparation
    - Detailed candidate assessment
    - Hiring decision support
    """
    logger.info(f"Candidate comparison request from user {current_user.id} for job {job_id}")
    
    if len(candidate_ids) > 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot compare more than 10 candidates at once"
        )
    
    try:
        comparisons = await scoring_service.compare_candidates_for_job(
            job_id=job_id,
            candidate_ids=candidate_ids
        )
        
        # Convert to ranking responses
        candidates = []
        for i, comp in enumerate(comparisons, 1):
            candidates.append(CandidateRankingResponse(
                candidate_id=UUID(comp['candidate_id']),
                candidate_name=comp['candidate_name'],
                score=comp['score'],
                rank=i,
                explanation=comp['explanation'],
                created_at=comp['created_at']
            ))
        
        # Calculate summary statistics
        scores = [c.score for c in candidates]
        summary = {
            'total_candidates': len(candidates),
            'average_score': sum(scores) / len(scores) if scores else 0,
            'score_range': max(scores) - min(scores) if scores else 0,
            'top_candidate': candidates[0].candidate_name if candidates else None
        }
        
        # Get job title
        job = await scoring_service.job_repo.get_by_id(job_id)
        
        return CandidateComparisonResponse(
            job_id=job_id,
            job_title=job.title if job else 'Unknown',
            candidates=candidates,
            comparison_summary=summary
        )
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Candidate comparison failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Candidate comparison failed"
        )


@router.get("/engine/info")
async def get_scoring_engine_info(
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Get information about the current scoring engine
    
    **Returns technical details about the ML model in use**
    
    **Use Cases:**
    - Model version tracking
    - Debugging scoring issues
    - Understanding model capabilities
    - Audit trail for scoring decisions
    """
    logger.info(f"Scoring engine info request from user {current_user.id}")
    
    try:
        info = scoring_service.get_scoring_engine_info()
        return info
        
    except Exception as e:
        logger.error(f"Failed to get scoring engine info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve scoring engine information"
        )


@router.post("/{score_id}/explain", response_model=ScoreExplanationResponse)
async def generate_score_explanation(
    score_id: UUID,
    request: ScoreExplanationRequest,
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Generate detailed explanation for a specific score
    
    **Enhanced explanation generation with section-wise analysis**
    
    **Features:**
    - Template-based explanation generation
    - Section-wise contribution analysis
    - Key matching elements identification
    - Improvement suggestions
    - Optional detailed breakdown
    
    **Use Cases:**
    - Understanding scoring decisions
    - Providing feedback to candidates
    - Explaining hiring decisions
    - Model interpretability
    """
    logger.info(f"Score explanation request from user {current_user.id} for score {score_id}")
    
    try:
        # Get the score first to extract candidate and job IDs
        score = await scoring_service.get_score_by_id(score_id)
        
        # Generate detailed explanation
        detailed_explanation = await scoring_service.generate_detailed_explanation(
            candidate_id=score.candidate_id,
            job_id=score.job_id,
            score_id=score_id
        )
        
        # Extract section scores if available
        section_scores = {}
        if 'section_analysis' in detailed_explanation:
            for section in detailed_explanation['section_analysis']:
                section_scores[section['section']] = section['score']
        
        # Extract key matches
        key_matches = []
        if 'section_analysis' in detailed_explanation:
            for section in detailed_explanation['section_analysis']:
                key_matches.extend(section.get('key_matches', []))
        
        return ScoreExplanationResponse(
            score_id=score_id,
            score=float(score.score),
            explanation=detailed_explanation.get('explanation', ''),
            section_scores=section_scores if request.detailed else None,
            key_matches=key_matches[:10] if request.detailed else None,  # Limit to top 10
            improvement_suggestions=detailed_explanation.get('improvement_suggestions') if request.detailed else None
        )
        
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Score not found"
        )
    except Exception as e:
        logger.error(f"Score explanation generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate score explanation"
        )


@router.post("/candidates/{candidate_id}/jobs/{job_id}/explain", response_model=ScoreExplanationResponse)
async def generate_candidate_job_explanation(
    candidate_id: UUID,
    job_id: UUID,
    detailed: bool = Query(False, description="Whether to include detailed section analysis"),
    current_user: User = Depends(require_any_role),
    scoring_service: ScoringService = Depends(get_scoring_service)
):
    """
    Generate explanation for candidate-job match
    
    **Generates explanation for any candidate-job pair**
    
    **Features:**
    - Works with existing scores or generates new ones
    - Template-based explanation with section analysis
    - Identifies key matching elements and gaps
    - Provides improvement suggestions
    - Supports both basic and detailed explanations
    
    **Query Parameters:**
    - detailed: Include section-wise breakdown and suggestions
    
    **Use Cases:**
    - Candidate feedback and coaching
    - Hiring manager decision support
    - Understanding match quality
    - Identifying development areas
    """
    logger.info(f"Candidate-job explanation request from user {current_user.id}")
    
    try:
        # Generate detailed explanation (will create score if needed)
        detailed_explanation = await scoring_service.generate_detailed_explanation(
            candidate_id=candidate_id,
            job_id=job_id
        )
        
        # Extract section scores if available
        section_scores = {}
        if 'section_analysis' in detailed_explanation:
            for section in detailed_explanation['section_analysis']:
                section_scores[section['section']] = section['score']
        
        # Extract key matches
        key_matches = []
        if 'section_analysis' in detailed_explanation:
            for section in detailed_explanation['section_analysis']:
                key_matches.extend(section.get('key_matches', []))
        
        return ScoreExplanationResponse(
            score_id=UUID(detailed_explanation.get('score_id', '00000000-0000-0000-0000-000000000000')),
            score=detailed_explanation.get('overall_score', 0.0),
            explanation=detailed_explanation.get('explanation', ''),
            section_scores=section_scores if detailed else None,
            key_matches=key_matches[:10] if detailed else None,  # Limit to top 10
            improvement_suggestions=detailed_explanation.get('improvement_suggestions') if detailed else None
        )
        
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Candidate-job explanation generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate candidate-job explanation"
        )