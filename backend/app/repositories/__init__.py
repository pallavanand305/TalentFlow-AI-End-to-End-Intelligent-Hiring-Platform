"""Data access layer"""

from backend.app.repositories.user_repository import UserRepository
from backend.app.repositories.candidate_repository import CandidateRepository

__all__ = ['UserRepository', 'CandidateRepository']
