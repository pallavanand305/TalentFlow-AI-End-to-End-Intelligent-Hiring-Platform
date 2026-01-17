"""Candidate repository for database operations"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy import select, or_, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.candidate import Candidate
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class CandidateRepository:
    """Repository for candidate database operations"""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository
        
        Args:
            session: Database session
        """
        self.session = session
    
    async def create(self, candidate_data: Dict[str, Any]) -> Candidate:
        """
        Create a new candidate
        
        Args:
            candidate_data: Dictionary with candidate data
        
        Returns:
            Created candidate
        """
        candidate = Candidate(**candidate_data)
        self.session.add(candidate)
        await self.session.commit()
        await self.session.refresh(candidate)
        
        logger.info(f"Created candidate: {candidate.id}")
        return candidate
    
    async def get_by_id(self, candidate_id: UUID) -> Optional[Candidate]:
        """
        Get candidate by ID
        
        Args:
            candidate_id: Candidate UUID
        
        Returns:
            Candidate if found, None otherwise
        """
        result = await self.session.execute(
            select(Candidate).where(Candidate.id == candidate_id)
        )
        candidate = result.scalar_one_or_none()
        
        if candidate:
            logger.debug(f"Found candidate: {candidate_id}")
        else:
            logger.debug(f"Candidate not found: {candidate_id}")
        
        return candidate
    
    async def get_by_email(self, email: str) -> Optional[Candidate]:
        """
        Get candidate by email
        
        Args:
            email: Candidate email
        
        Returns:
            Candidate if found, None otherwise
        """
        result = await self.session.execute(
            select(Candidate).where(Candidate.email == email)
        )
        return result.scalar_one_or_none()
    
    async def update(self, candidate_id: UUID, update_data: Dict[str, Any]) -> Optional[Candidate]:
        """
        Update candidate
        
        Args:
            candidate_id: Candidate UUID
            update_data: Dictionary with fields to update
        
        Returns:
            Updated candidate if found, None otherwise
        """
        candidate = await self.get_by_id(candidate_id)
        
        if not candidate:
            return None
        
        for key, value in update_data.items():
            if hasattr(candidate, key):
                setattr(candidate, key, value)
        
        await self.session.commit()
        await self.session.refresh(candidate)
        
        logger.info(f"Updated candidate: {candidate_id}")
        return candidate
    
    async def delete(self, candidate_id: UUID) -> bool:
        """
        Delete candidate (hard delete)
        
        Args:
            candidate_id: Candidate UUID
        
        Returns:
            True if deleted, False if not found
        """
        candidate = await self.get_by_id(candidate_id)
        
        if not candidate:
            return False
        
        await self.session.delete(candidate)
        await self.session.commit()
        
        logger.info(f"Deleted candidate: {candidate_id}")
        return True
    
    async def list_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "created_at"
    ) -> List[Candidate]:
        """
        List all candidates with pagination
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: Field to order by
        
        Returns:
            List of candidates
        """
        query = select(Candidate)
        
        # Apply ordering
        if hasattr(Candidate, order_by):
            query = query.order_by(getattr(Candidate, order_by).desc())
        
        query = query.offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        candidates = result.scalars().all()
        
        logger.debug(f"Listed {len(candidates)} candidates")
        return list(candidates)
    
    async def search(
        self,
        query: Optional[str] = None,
        skills: Optional[List[str]] = None,
        min_experience_years: Optional[int] = None,
        education_level: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Candidate]:
        """
        Search candidates with filters
        
        Args:
            query: Text search query (searches name and email)
            skills: List of required skills
            min_experience_years: Minimum years of experience
            education_level: Required education level
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of matching candidates
        """
        stmt = select(Candidate)
        
        conditions = []
        
        # Text search
        if query:
            search_pattern = f"%{query}%"
            conditions.append(
                or_(
                    Candidate.name.ilike(search_pattern),
                    Candidate.email.ilike(search_pattern)
                )
            )
        
        # Skills filter
        if skills:
            for skill in skills:
                conditions.append(Candidate.skills.contains([skill]))
        
        # Experience filter
        if min_experience_years is not None:
            conditions.append(Candidate.experience_years >= min_experience_years)
        
        # Education filter
        if education_level:
            conditions.append(Candidate.education_level == education_level)
        
        # Apply all conditions
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        # Apply pagination
        stmt = stmt.order_by(Candidate.created_at.desc()).offset(skip).limit(limit)
        
        result = await self.session.execute(stmt)
        candidates = result.scalars().all()
        
        logger.info(f"Search returned {len(candidates)} candidates")
        return list(candidates)
    
    async def count(self) -> int:
        """
        Count total number of candidates
        
        Returns:
            Total count
        """
        result = await self.session.execute(
            select(func.count()).select_from(Candidate)
        )
        count = result.scalar()
        return count or 0
    
    async def get_by_skills(self, skills: List[str], limit: int = 100) -> List[Candidate]:
        """
        Get candidates by skills
        
        Args:
            skills: List of skills to match
            limit: Maximum number of results
        
        Returns:
            List of candidates with matching skills
        """
        stmt = select(Candidate)
        
        # Match candidates that have any of the specified skills
        for skill in skills:
            stmt = stmt.where(Candidate.skills.contains([skill]))
        
        stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def bulk_create(self, candidates_data: List[Dict[str, Any]]) -> List[Candidate]:
        """
        Create multiple candidates in bulk
        
        Args:
            candidates_data: List of candidate data dictionaries
        
        Returns:
            List of created candidates
        """
        candidates = [Candidate(**data) for data in candidates_data]
        self.session.add_all(candidates)
        await self.session.commit()
        
        for candidate in candidates:
            await self.session.refresh(candidate)
        
        logger.info(f"Bulk created {len(candidates)} candidates")
        return candidates
