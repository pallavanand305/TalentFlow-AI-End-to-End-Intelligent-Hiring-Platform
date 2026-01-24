"""Score repository for database operations"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from backend.app.models.score import Score
from backend.app.models.candidate import Candidate
from backend.app.models.job import Job
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class ScoreRepository:
    """Repository for score-related database operations"""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize score repository
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def create(self, score_data: Dict[str, Any]) -> Score:
        """
        Create a new score
        
        Args:
            score_data: Score data dictionary
        
        Returns:
            Created score
        """
        score = Score(**score_data)
        self.db.add(score)
        await self.db.commit()
        await self.db.refresh(score)
        
        logger.info(f"Created score: {score.id}")
        return score
    
    async def get_by_id(self, score_id: UUID) -> Optional[Score]:
        """
        Get score by ID
        
        Args:
            score_id: Score UUID
        
        Returns:
            Score if found, None otherwise
        """
        stmt = select(Score).where(Score.id == score_id).options(
            selectinload(Score.candidate),
            selectinload(Score.job)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_candidate_and_job(
        self, 
        candidate_id: UUID, 
        job_id: UUID,
        current_only: bool = True
    ) -> Optional[Score]:
        """
        Get score by candidate and job
        
        Args:
            candidate_id: Candidate UUID
            job_id: Job UUID
            current_only: Whether to only return current scores
        
        Returns:
            Score if found, None otherwise
        """
        stmt = select(Score).where(
            and_(
                Score.candidate_id == candidate_id,
                Score.job_id == job_id
            )
        ).options(
            selectinload(Score.candidate),
            selectinload(Score.job)
        )
        
        if current_only:
            stmt = stmt.where(Score.is_current == True)
        
        # Get the most recent score
        stmt = stmt.order_by(desc(Score.created_at))
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_scores_for_job(
        self, 
        job_id: UUID,
        current_only: bool = True,
        min_score: Optional[float] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Score]:
        """
        Get all scores for a job, ordered by score descending
        
        Args:
            job_id: Job UUID
            current_only: Whether to only return current scores
            min_score: Minimum score threshold
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of scores ordered by score descending
        """
        stmt = select(Score).where(Score.job_id == job_id).options(
            selectinload(Score.candidate),
            selectinload(Score.job)
        )
        
        if current_only:
            stmt = stmt.where(Score.is_current == True)
        
        if min_score is not None:
            stmt = stmt.where(Score.score >= min_score)
        
        # Order by score descending (highest first)
        stmt = stmt.order_by(desc(Score.score), desc(Score.created_at))
        
        # Apply pagination
        stmt = stmt.offset(skip).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_scores_for_candidate(
        self, 
        candidate_id: UUID,
        current_only: bool = True,
        skip: int = 0,
        limit: int = 100
    ) -> List[Score]:
        """
        Get all scores for a candidate
        
        Args:
            candidate_id: Candidate UUID
            current_only: Whether to only return current scores
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of scores for the candidate
        """
        stmt = select(Score).where(Score.candidate_id == candidate_id).options(
            selectinload(Score.candidate),
            selectinload(Score.job)
        )
        
        if current_only:
            stmt = stmt.where(Score.is_current == True)
        
        # Order by score descending
        stmt = stmt.order_by(desc(Score.score), desc(Score.created_at))
        
        # Apply pagination
        stmt = stmt.offset(skip).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_top_candidates_for_job(
        self, 
        job_id: UUID, 
        limit: int = 10,
        min_score: Optional[float] = None
    ) -> List[Score]:
        """
        Get top candidates for a job based on scores
        
        Args:
            job_id: Job UUID
            limit: Maximum number of candidates to return
            min_score: Minimum score threshold
        
        Returns:
            List of top scores for the job
        """
        stmt = select(Score).where(
            and_(
                Score.job_id == job_id,
                Score.is_current == True
            )
        ).options(
            selectinload(Score.candidate),
            selectinload(Score.job)
        )
        
        if min_score is not None:
            stmt = stmt.where(Score.score >= min_score)
        
        # Order by score descending and limit
        stmt = stmt.order_by(desc(Score.score), desc(Score.created_at)).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def invalidate_scores_for_job(self, job_id: UUID) -> int:
        """
        Mark all scores for a job as not current (invalidate them)
        
        Args:
            job_id: Job UUID
        
        Returns:
            Number of scores invalidated
        """
        stmt = update(Score).where(
            and_(
                Score.job_id == job_id,
                Score.is_current == True
            )
        ).values(is_current=False)
        
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        count = result.rowcount
        logger.info(f"Invalidated {count} scores for job: {job_id}")
        return count
    
    async def delete_scores_for_candidate(self, candidate_id: UUID) -> int:
        """
        Delete all scores for a candidate
        
        Args:
            candidate_id: Candidate UUID
        
        Returns:
            Number of scores deleted
        """
        stmt = delete(Score).where(Score.candidate_id == candidate_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        count = result.rowcount
        logger.info(f"Deleted {count} scores for candidate: {candidate_id}")
        return count
    
    async def delete_scores_for_job(self, job_id: UUID) -> int:
        """
        Delete all scores for a job
        
        Args:
            job_id: Job UUID
        
        Returns:
            Number of scores deleted
        """
        stmt = delete(Score).where(Score.job_id == job_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        count = result.rowcount
        logger.info(f"Deleted {count} scores for job: {job_id}")
        return count
    
    async def get_score_statistics(self, job_id: Optional[UUID] = None) -> Dict[str, Any]:
        """
        Get score statistics
        
        Args:
            job_id: Optional job ID to filter by
        
        Returns:
            Dictionary with score statistics
        """
        stmt = select(
            func.count(Score.id).label('total_scores'),
            func.avg(Score.score).label('avg_score'),
            func.min(Score.score).label('min_score'),
            func.max(Score.score).label('max_score')
        ).where(Score.is_current == True)
        
        if job_id:
            stmt = stmt.where(Score.job_id == job_id)
        
        result = await self.db.execute(stmt)
        row = result.first()
        
        return {
            'total_scores': row.total_scores or 0,
            'avg_score': float(row.avg_score) if row.avg_score else 0.0,
            'min_score': float(row.min_score) if row.min_score else 0.0,
            'max_score': float(row.max_score) if row.max_score else 0.0
        }
    
    async def get_scores_by_model_version(
        self, 
        model_version: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Score]:
        """
        Get scores by model version
        
        Args:
            model_version: Model version string
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of scores for the model version
        """
        stmt = select(Score).where(Score.model_version == model_version).options(
            selectinload(Score.candidate),
            selectinload(Score.job)
        ).order_by(desc(Score.created_at)).offset(skip).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def count_scores_for_job(self, job_id: UUID, current_only: bool = True) -> int:
        """
        Count scores for a job
        
        Args:
            job_id: Job UUID
            current_only: Whether to only count current scores
        
        Returns:
            Number of scores
        """
        stmt = select(func.count(Score.id)).where(Score.job_id == job_id)
        
        if current_only:
            stmt = stmt.where(Score.is_current == True)
        
        result = await self.db.execute(stmt)
        return result.scalar()
    
    async def get_candidate_rankings_for_job(
        self, 
        job_id: UUID,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get candidate rankings for a job with additional metadata
        
        Args:
            job_id: Job UUID
            limit: Maximum number of candidates to return
        
        Returns:
            List of candidate ranking dictionaries
        """
        stmt = select(
            Score,
            Candidate.name.label('candidate_name'),
            Candidate.email.label('candidate_email'),
            func.row_number().over(order_by=desc(Score.score)).label('rank')
        ).join(
            Candidate, Score.candidate_id == Candidate.id
        ).where(
            and_(
                Score.job_id == job_id,
                Score.is_current == True
            )
        ).order_by(desc(Score.score)).limit(limit)
        
        result = await self.db.execute(stmt)
        rows = result.all()
        
        rankings = []
        for row in rows:
            rankings.append({
                'rank': row.rank,
                'score_id': str(row.Score.id),
                'candidate_id': str(row.Score.candidate_id),
                'candidate_name': row.candidate_name,
                'candidate_email': row.candidate_email,
                'score': float(row.Score.score),
                'explanation': row.Score.explanation,
                'created_at': row.Score.created_at
            })
        
        return rankings
    
    async def update_score(self, score_id: UUID, updates: Dict[str, Any]) -> Optional[Score]:
        """
        Update a score
        
        Args:
            score_id: Score UUID
            updates: Fields to update
        
        Returns:
            Updated score if found, None otherwise
        """
        stmt = update(Score).where(Score.id == score_id).values(**updates)
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        if result.rowcount > 0:
            return await self.get_by_id(score_id)
        return None
    
    async def bulk_create_scores(self, scores_data: List[Dict[str, Any]]) -> List[Score]:
        """
        Create multiple scores in bulk
        
        Args:
            scores_data: List of score data dictionaries
        
        Returns:
            List of created scores
        """
        scores = [Score(**data) for data in scores_data]
        self.db.add_all(scores)
        await self.db.commit()
        
        # Refresh all scores to get their IDs
        for score in scores:
            await self.db.refresh(score)
        
        logger.info(f"Created {len(scores)} scores in bulk")
        return scores