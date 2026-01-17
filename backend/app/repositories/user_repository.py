"""User repository for database operations"""

from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from passlib.context import CryptContext

from backend.app.models.user import User, UserRole
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRepository:
    """Repository for User CRUD operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    async def create(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.RECRUITER
    ) -> User:
        """Create a new user"""
        hashed_password = self.hash_password(password)
        
        user = User(
            username=username,
            email=email,
            password_hash=hashed_password,
            role=role
        )
        
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        
        logger.info(f"Created user: {user.username} with role {user.role}")
        return user
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def list_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List all users with pagination"""
        result = await self.session.execute(
            select(User).offset(skip).limit(limit)
        )
        return list(result.scalars().all())
    
    async def update(self, user: User) -> User:
        """Update user"""
        await self.session.flush()
        await self.session.refresh(user)
        logger.info(f"Updated user: {user.username}")
        return user
    
    async def delete(self, user: User) -> None:
        """Delete user"""
        await self.session.delete(user)
        await self.session.flush()
        logger.info(f"Deleted user: {user.username}")
    
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user by username and password"""
        user = await self.get_by_username(username)
        
        if not user:
            logger.warning(f"Authentication failed: user {username} not found")
            return None
        
        if not self.verify_password(password, user.password_hash):
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None
        
        logger.info(f"User authenticated: {username}")
        return user
