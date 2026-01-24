"""Pydantic schemas for authentication"""

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from backend.app.models.user import UserRole


class UserCreate(BaseModel):
    """Schema for user registration"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    role: UserRole = UserRole.RECRUITER


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Schema for token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Schema for token refresh request"""
    refresh_token: str


class UserResponse(BaseModel):
    """Schema for user response"""
    id: str
    username: str
    email: str
    role: UserRole
    
    model_config = ConfigDict(from_attributes=True)
