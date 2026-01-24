"""Authentication service for JWT token management"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.models.user import User, UserRole

logger = get_logger(__name__)


class AuthService:
    """Service for authentication and JWT token management"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.REFRESH_TOKEN_EXPIRE_DAYS
    
    def create_access_token(
        self,
        user_id: str,
        username: str,
        role: UserRole,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Generate JWT access token
        
        Args:
            user_id: User ID
            username: Username
            role: User role
            expires_delta: Optional custom expiration time
        
        Returns:
            JWT token string
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {
            "sub": str(user_id),
            "username": username,
            "role": role.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created access token for user: {username}")
        return encoded_jwt
    
    def create_refresh_token(
        self,
        user_id: str,
        username: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Generate JWT refresh token
        
        Args:
            user_id: User ID
            username: Username
            expires_delta: Optional custom expiration time
        
        Returns:
            JWT refresh token string
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = {
            "sub": str(user_id),
            "username": username,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created refresh token for user: {username}")
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
        
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
                logger.warning("Token has expired")
                return None
            
            return payload
        
        except JWTError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            return None
    
    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify access token specifically
        
        Args:
            token: JWT token string
        
        Returns:
            Token payload if valid access token, None otherwise
        """
        payload = self.verify_token(token)
        
        if not payload:
            return None
        
        if payload.get("type") != "access":
            logger.warning("Token is not an access token")
            return None
        
        return payload
    
    def verify_refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify refresh token specifically
        
        Args:
            token: JWT token string
        
        Returns:
            Token payload if valid refresh token, None otherwise
        """
        payload = self.verify_token(token)
        
        if not payload:
            return None
        
        if payload.get("type") != "refresh":
            logger.warning("Token is not a refresh token")
            return None
        
        return payload
    
    def get_user_id_from_token(self, token: str) -> Optional[str]:
        """Extract user ID from token"""
        payload = self.verify_access_token(token)
        if payload:
            return payload.get("sub")
        return None
    
    def get_user_role_from_token(self, token: str) -> Optional[UserRole]:
        """Extract user role from token"""
        payload = self.verify_access_token(token)
        if payload:
            role_str = payload.get("role")
            try:
                return UserRole(role_str)
            except ValueError:
                return None
        return None


# Global auth service instance
auth_service = AuthService()
