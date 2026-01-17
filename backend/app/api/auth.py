"""Authentication API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user
from backend.app.repositories.user_repository import UserRepository
from backend.app.services.auth_service import auth_service
from backend.app.schemas.auth import (
    UserCreate, UserLogin, TokenResponse, TokenRefresh, UserResponse
)
from backend.app.models.user import User
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user
    
    Creates a new user account with the specified role.
    
    **Parameters:**
    - **username**: Unique username (3-50 characters, alphanumeric)
    - **email**: Valid email address
    - **password**: Password (minimum 8 characters)
    - **role**: User role (admin, recruiter, hiring_manager) - default: recruiter
    
    **Returns:**
    - User object with ID, username, email, and role
    
    **Errors:**
    - 400: Username or email already exists
    - 422: Validation error (invalid format)
    """
    user_repo = UserRepository(db)
    
    # Check if username already exists
    existing_user = await user_repo.get_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = await user_repo.get_by_email(user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = await user_repo.create(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        role=user_data.role
    )
    
    await db.commit()
    
    logger.info(f"User registered: {user.username}")
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Login and get access token
    
    Authenticates user credentials and returns JWT tokens.
    
    **Parameters:**
    - **username**: Username
    - **password**: Password
    
    **Returns:**
    - **access_token**: JWT access token (expires in 30 minutes)
    - **refresh_token**: JWT refresh token (expires in 7 days)
    - **token_type**: Token type (always "bearer")
    
    **Usage:**
    Include the access token in subsequent requests:
    ```
    Authorization: Bearer <access_token>
    ```
    
    **Errors:**
    - 401: Invalid username or password
    """
    user_repo = UserRepository(db)
    
    # Authenticate user
    user = await user_repo.authenticate(credentials.username, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = auth_service.create_access_token(
        user_id=str(user.id),
        username=user.username,
        role=user.role
    )
    
    refresh_token = auth_service.create_refresh_token(
        user_id=str(user.id),
        username=user.username
    )
    
    logger.info(f"User logged in: {user.username}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token
    
    - **refresh_token**: Valid refresh token
    
    Returns new access token and refresh token
    """
    # Verify refresh token
    payload = auth_service.verify_refresh_token(token_data.refresh_token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user
    user_id = payload.get("sub")
    username = payload.get("username")
    
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new tokens
    access_token = auth_service.create_access_token(
        user_id=str(user.id),
        username=user.username,
        role=user.role
    )
    
    refresh_token = auth_service.create_refresh_token(
        user_id=str(user.id),
        username=user.username
    )
    
    logger.info(f"Token refreshed for user: {user.username}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information
    
    Requires authentication
    """
    return current_user
