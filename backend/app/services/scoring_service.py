"""Scoring service for orchestrating candidate-job scoring operations"""

from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
import asyncio
from datetime import datetime

from backend.app.repositories.score_repository import ScoreRepository
from backend.app.repositories.candidate_repository import CandidateRepository
from backend.app.repositories.job_repository import JobRepository
from backend.app.models.score import Score
from backend.app.models.candidate import Candidate
from backend.app.models.job import Job
from ml.inference.scoring_engine import ScoringEngine, ScoringResult
from backend.app.core.logging import get_logger
from backend.app.core.exceptions import ValidationException, NotFoundException

logger = get_logger(__name__)


class ScoringService:
    """Service for candidate-job scoring operations"""
    
    def __init__(
        self,
        score_repository: ScoreRepository,
        candidate_repository: CandidateRepository,
        job_repository: JobRepository,
        scoring_engine: Optional[ScoringEngine] = None
    ):
        """
        Initialize scoring service
        
        Args:
            score_repository: Score repository
            candidate_repository: Candidate repository
            job_repository: Job repository
            scoring_engine: Optional scoring engine (creates default if None)
        """
        self.score_repo = score_repository
        self.candidate_repo = candidate_repository
        self.job_repo = job_repository
        self.scoring_engine = scoring_engine or ScoringEngine(model_type="tfidf")
    
    async def score_candidate_for_job(
        self,
        candidate_id: UUID,
        job_id: UUID,
        generate_explanation: bool = False,
        detailed_explanation: bool = False,
        force_rescore: bool = False
    ) -> Score:
        """
        Score a candidate for a job
        
        Args:
            candidate_id: Candidate UUID
            job_id: Job UUID
            generate_explanation: Whether to generate explanation
            detailed_explanation: Whether to generate detailed explanation
            force_rescore: Whether to force rescoring even if score exists
        
        Returns:
            Score object
        
        Raises:
            NotFoundException: If candidate or job not found
            ValidationException: If scoring fails
        """
        logger.info(f"Scoring candidate {candidate_id} for job {job_id}")
        
        # Check if score already exists and is current
        if not force_rescore:
            existing_score = await self.score_repo.get_by_candidate_and_job(
                candidate_id, job_id, current_only=True
            )
            if existing_score:
                logger.info(f"Using existing score: {existing_score.id}")
                return existing_score
        
        # Get candidate and job
        candidate = await self.candidate_repo.get_by_id(candidate_id)
        if not candidate:
            raise NotFoundException(f"Candidate not found: {candidate_id}")
        
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        # Validate that candidate has parsed data
        if not candidate.parsed_data:
            raise ValidationException(f"Candidate {candidate_id} has no parsed resume data")
        
        try:
            # Convert parsed data to ParsedResume object
            from backend.app.schemas.resume import ParsedResume
            parsed_resume = ParsedResume(**candidate.parsed_data)
            
            # Compute score using ML engine
            score_result = self.scoring_engine.score_candidate(parsed_resume, job)
            
            # Generate explanation if requested
            explanation = None
            if generate_explanation:
                explanation = self.scoring_engine.explain_score(
                    parsed_resume, job, score_result, detailed=detailed_explanation
                )
            
            # Create score record
            score_data = {
                'candidate_id': candidate_id,
                'job_id': job_id,
                'score': score_result['overall_score'],
                'model_version': self.scoring_engine.get_model_info()['version'],
                'explanation': explanation,
                'is_current': True
            }
            
            # If force_rescore, invalidate existing scores first
            if force_rescore:
                await self._invalidate_existing_score(candidate_id, job_id)
            
            score = await self.score_repo.create(score_data)
            
            logger.info(f"Successfully scored candidate {candidate_id} for job {job_id}: {score.score:.3f}")
            return score
            
        except Exception as e:
            logger.error(f"Scoring failed for candidate {candidate_id} and job {job_id}: {str(e)}")
            raise ValidationException(f"Scoring failed: {str(e)}")
    
    async def generate_detailed_explanation(
        self,
        candidate_id: UUID,
        job_id: UUID,
        score_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation for a candidate-job score
        
        Args:
            candidate_id: Candidate UUID
            job_id: Job UUID
            score_id: Optional specific score ID to explain
        
        Returns:
            Dictionary containing detailed explanation and analysis
        
        Raises:
            NotFoundException: If candidate, job, or score not found
        """
        logger.info(f"Generating detailed explanation for candidate {candidate_id} and job {job_id}")
        
        # Get candidate and job
        candidate = await self.candidate_repo.get_by_id(candidate_id)
        if not candidate:
            raise NotFoundException(f"Candidate not found: {candidate_id}")
        
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        # Get score if specific score_id provided, otherwise get current score
        if score_id:
            score = await self.score_repo.get_by_id(score_id)
            if not score:
                raise NotFoundException(f"Score not found: {score_id}")
        else:
            score = await self.score_repo.get_by_candidate_and_job(
                candidate_id, job_id, current_only=True
            )
            if not score:
                # Generate new score if none exists
                score = await self.score_candidate_for_job(
                    candidate_id=candidate_id,
                    job_id=job_id,
                    generate_explanation=False
                )
        
        # Validate that candidate has parsed data
        if not candidate.parsed_data:
            raise ValidationException(f"Candidate {candidate_id} has no parsed resume data")
        
        try:
            # Convert parsed data to ParsedResume object
            from backend.app.schemas.resume import ParsedResume
            parsed_resume = ParsedResume(**candidate.parsed_data)
            
            # Recompute scores for detailed analysis
            score_result = self.scoring_engine.score_candidate(parsed_resume, job)
            
            # Generate detailed explanation
            detailed_explanation = self.scoring_engine.generate_detailed_explanation(
                parsed_resume, job, score_result
            )
            
            # Add score metadata
            detailed_explanation['score_id'] = str(score.id)
            detailed_explanation['candidate_id'] = str(candidate_id)
            detailed_explanation['job_id'] = str(job_id)
            detailed_explanation['candidate_name'] = candidate.name
            detailed_explanation['job_title'] = job.title
            detailed_explanation['created_at'] = score.created_at
            
            return detailed_explanation
            
        except Exception as e:
            logger.error(f"Detailed explanation generation failed: {str(e)}")
            raise ValidationException(f"Explanation generation failed: {str(e)}")
    
    async def batch_score_candidates_for_job(
        self,
        candidate_ids: List[UUID],
        job_id: UUID,
        generate_explanations: bool = False,
        force_rescore: bool = False
    ) -> List[Score]:
        """
        Score multiple candidates for a job
        
        Args:
            candidate_ids: List of candidate UUIDs
            job_id: Job UUID
            generate_explanations: Whether to generate explanations
            force_rescore: Whether to force rescoring
        
        Returns:
            List of Score objects
        """
        logger.info(f"Batch scoring {len(candidate_ids)} candidates for job {job_id}")
        
        scores = []
        for candidate_id in candidate_ids:
            try:
                score = await self.score_candidate_for_job(
                    candidate_id=candidate_id,
                    job_id=job_id,
                    generate_explanation=generate_explanations,
                    force_rescore=force_rescore
                )
                scores.append(score)
            except Exception as e:
                logger.error(f"Failed to score candidate {candidate_id}: {str(e)}")
                # Continue with other candidates
                continue
        
        logger.info(f"Successfully scored {len(scores)} out of {len(candidate_ids)} candidates")
        return scores
    
    async def get_ranked_candidates_for_job(
        self,
        job_id: UUID,
        min_score: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get ranked candidates for a job
        
        Args:
            job_id: Job UUID
            min_score: Minimum score threshold
            limit: Maximum number of candidates to return
        
        Returns:
            List of candidate ranking dictionaries
        """
        logger.info(f"Getting ranked candidates for job {job_id}")
        
        # Verify job exists
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        # Get candidate rankings
        rankings = await self.score_repo.get_candidate_rankings_for_job(
            job_id=job_id,
            limit=limit
        )
        
        # Filter by minimum score if specified
        if min_score is not None:
            rankings = [r for r in rankings if r['score'] >= min_score]
        
        return rankings
    
    async def get_top_candidates_for_job(
        self,
        job_id: UUID,
        limit: int = 10,
        min_score: Optional[float] = None
    ) -> List[Score]:
        """
        Get top candidates for a job
        
        Args:
            job_id: Job UUID
            limit: Maximum number of candidates to return
            min_score: Minimum score threshold
        
        Returns:
            List of top scores
        """
        logger.info(f"Getting top {limit} candidates for job {job_id}")
        
        # Verify job exists
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        scores = await self.score_repo.get_top_candidates_for_job(
            job_id=job_id,
            limit=limit,
            min_score=min_score
        )
        
        return scores
    
    async def get_candidate_scores(
        self,
        candidate_id: UUID,
        current_only: bool = True,
        limit: int = 100
    ) -> List[Score]:
        """
        Get all scores for a candidate
        
        Args:
            candidate_id: Candidate UUID
            current_only: Whether to only return current scores
            limit: Maximum number of scores to return
        
        Returns:
            List of scores for the candidate
        """
        logger.info(f"Getting scores for candidate {candidate_id}")
        
        # Verify candidate exists
        candidate = await self.candidate_repo.get_by_id(candidate_id)
        if not candidate:
            raise NotFoundException(f"Candidate not found: {candidate_id}")
        
        scores = await self.score_repo.get_scores_for_candidate(
            candidate_id=candidate_id,
            current_only=current_only,
            limit=limit
        )
        
        return scores
    
    async def get_score_by_id(self, score_id: UUID) -> Score:
        """
        Get score by ID
        
        Args:
            score_id: Score UUID
        
        Returns:
            Score object
        
        Raises:
            NotFoundException: If score not found
        """
        score = await self.score_repo.get_by_id(score_id)
        if not score:
            raise NotFoundException(f"Score not found: {score_id}")
        
        return score
    
    async def invalidate_scores_for_job(self, job_id: UUID) -> int:
        """
        Invalidate all scores for a job (mark as not current)
        
        Args:
            job_id: Job UUID
        
        Returns:
            Number of scores invalidated
        """
        logger.info(f"Invalidating scores for job {job_id}")
        
        count = await self.score_repo.invalidate_scores_for_job(job_id)
        logger.info(f"Invalidated {count} scores for job {job_id}")
        return count
    
    async def delete_scores_for_candidate(self, candidate_id: UUID) -> int:
        """
        Delete all scores for a candidate
        
        Args:
            candidate_id: Candidate UUID
        
        Returns:
            Number of scores deleted
        """
        logger.info(f"Deleting scores for candidate {candidate_id}")
        
        count = await self.score_repo.delete_scores_for_candidate(candidate_id)
        logger.info(f"Deleted {count} scores for candidate {candidate_id}")
        return count
    
    async def get_job_scoring_statistics(self, job_id: UUID) -> Dict[str, Any]:
        """
        Get scoring statistics for a job
        
        Args:
            job_id: Job UUID
        
        Returns:
            Dictionary with scoring statistics
        """
        logger.info(f"Getting scoring statistics for job {job_id}")
        
        # Verify job exists
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        stats = await self.score_repo.get_score_statistics(job_id=job_id)
        
        # Add additional job-specific stats
        stats['job_id'] = str(job_id)
        stats['job_title'] = job.title
        
        return stats
    
    async def rescore_all_candidates_for_job(
        self,
        job_id: UUID,
        generate_explanations: bool = False
    ) -> Dict[str, Any]:
        """
        Rescore all candidates for a job (useful after job updates)
        
        Args:
            job_id: Job UUID
            generate_explanations: Whether to generate explanations
        
        Returns:
            Dictionary with rescoring results
        """
        logger.info(f"Rescoring all candidates for job {job_id}")
        
        # Verify job exists
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        # Get all candidates who have been scored for this job
        existing_scores = await self.score_repo.get_scores_for_job(
            job_id=job_id,
            current_only=True,
            limit=1000  # Large limit to get all candidates
        )
        
        candidate_ids = [score.candidate_id for score in existing_scores]
        
        if not candidate_ids:
            return {
                'job_id': str(job_id),
                'candidates_rescored': 0,
                'candidates_failed': 0,
                'message': 'No candidates found to rescore'
            }
        
        # Invalidate existing scores
        await self.invalidate_scores_for_job(job_id)
        
        # Rescore all candidates
        new_scores = await self.batch_score_candidates_for_job(
            candidate_ids=candidate_ids,
            job_id=job_id,
            generate_explanations=generate_explanations,
            force_rescore=True
        )
        
        result = {
            'job_id': str(job_id),
            'candidates_rescored': len(new_scores),
            'candidates_failed': len(candidate_ids) - len(new_scores),
            'message': f'Rescored {len(new_scores)} candidates for job {job.title}'
        }
        
        logger.info(f"Rescoring completed: {result}")
        return result
    
    async def find_best_jobs_for_candidate(
        self,
        candidate_id: UUID,
        min_score: float = 0.5,
        limit: int = 10
    ) -> List[Score]:
        """
        Find best job matches for a candidate
        
        Args:
            candidate_id: Candidate UUID
            min_score: Minimum score threshold
            limit: Maximum number of jobs to return
        
        Returns:
            List of scores for best matching jobs
        """
        logger.info(f"Finding best jobs for candidate {candidate_id}")
        
        # Verify candidate exists
        candidate = await self.candidate_repo.get_by_id(candidate_id)
        if not candidate:
            raise NotFoundException(f"Candidate not found: {candidate_id}")
        
        # Get candidate's scores, filtered by minimum score
        scores = await self.score_repo.get_scores_for_candidate(
            candidate_id=candidate_id,
            current_only=True,
            limit=limit * 2  # Get more to filter
        )
        
        # Filter by minimum score and limit
        filtered_scores = [s for s in scores if s.score >= min_score][:limit]
        
        return filtered_scores
    
    async def compare_candidates_for_job(
        self,
        job_id: UUID,
        candidate_ids: List[UUID]
    ) -> List[Dict[str, Any]]:
        """
        Compare specific candidates for a job
        
        Args:
            job_id: Job UUID
            candidate_ids: List of candidate UUIDs to compare
        
        Returns:
            List of candidate comparison data
        """
        logger.info(f"Comparing {len(candidate_ids)} candidates for job {job_id}")
        
        # Verify job exists
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise NotFoundException(f"Job not found: {job_id}")
        
        comparisons = []
        for candidate_id in candidate_ids:
            try:
                # Get or create score
                score = await self.score_candidate_for_job(
                    candidate_id=candidate_id,
                    job_id=job_id,
                    generate_explanation=True
                )
                
                # Get candidate details
                candidate = await self.candidate_repo.get_by_id(candidate_id)
                
                comparison = {
                    'candidate_id': str(candidate_id),
                    'candidate_name': candidate.name if candidate else 'Unknown',
                    'score': float(score.score),
                    'explanation': score.explanation,
                    'created_at': score.created_at
                }
                comparisons.append(comparison)
                
            except Exception as e:
                logger.error(f"Failed to compare candidate {candidate_id}: {str(e)}")
                continue
        
        # Sort by score descending
        comparisons.sort(key=lambda x: x['score'], reverse=True)
        
        return comparisons
    
    async def _invalidate_existing_score(self, candidate_id: UUID, job_id: UUID) -> None:
        """
        Invalidate existing score for candidate-job pair
        
        Args:
            candidate_id: Candidate UUID
            job_id: Job UUID
        """
        existing_score = await self.score_repo.get_by_candidate_and_job(
            candidate_id, job_id, current_only=True
        )
        
        if existing_score:
            await self.score_repo.update_score(
                existing_score.id,
                {'is_current': False}
            )
            logger.info(f"Invalidated existing score: {existing_score.id}")
    
    def get_scoring_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the current scoring engine
        
        Returns:
            Dictionary with scoring engine information
        """
        return self.scoring_engine.get_model_info()