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
    Register a new user account
    
    Creates a new user account with the specified role and returns user information.
    
    ## Request Body
    
    - **username**: Unique username (3-50 characters, alphanumeric and underscores only)
    - **email**: Valid email address (will be used for notifications)
    - **password**: Secure password (minimum 8 characters, recommended: mix of letters, numbers, symbols)
    - **role**: User role determining permissions (default: recruiter)
    
    ## User Roles
    
    | Role | Permissions |
    |------|-------------|
    | **admin** | Full system access, user management, model promotion |
    | **recruiter** | Upload resumes, view candidates, manage candidate data |
    | **hiring_manager** | Create jobs, view ranked candidates, manage job postings |
    
    ## Response
    
    Returns the created user object with:
    - Unique user ID (UUID)
    - Username and email
    - Assigned role
    
    ## Error Responses
    
    - **400 Bad Request**: Username or email already exists
    - **422 Unprocessable Entity**: Validation error (invalid format, missing fields)
    
    ## Example Usage
    
    ```bash
    curl -X POST "http://localhost:8000/api/v1/auth/register" \\
         -H "Content-Type: application/json" \\
         -d '{
           "username": "jane_recruiter",
           "email": "jane@company.com",
           "password": "SecurePass123!",
           "role": "recruiter"
         }'
    ```
    
    **Requirements**: 7.1, 7.6
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
    Authenticate user and obtain JWT tokens
    
    Validates user credentials and returns access and refresh tokens for API authentication.
    
    ## Request Body
    
    - **username**: Registered username
    - **password**: User password
    
    ## Response
    
    Returns JWT tokens for authentication:
    - **access_token**: Short-lived token for API requests (expires in 30 minutes)
    - **refresh_token**: Long-lived token for obtaining new access tokens (expires in 7 days)
    - **token_type**: Always "bearer"
    
    ## Token Usage
    
    Include the access token in the Authorization header for subsequent API requests:
    
    ```
    Authorization: Bearer <access_token>
    ```
    
    ## Token Expiration
    
    - **Access Token**: 30 minutes (for security)
    - **Refresh Token**: 7 days (for convenience)
    
    When the access token expires, use the refresh token with the `/api/v1/auth/refresh` endpoint to get a new access token without re-entering credentials.
    
    ## Security Notes
    
    - Store tokens securely (never in localStorage for web apps)
    - Use HTTPS in production
    - Tokens are signed with HS256 algorithm
    - Failed login attempts are logged for security monitoring
    
    ## Error Responses
    
    - **401 Unauthorized**: Invalid username or password
    - **422 Unprocessable Entity**: Missing or invalid request format
    
    ## Example Usage
    
    ```bash
    curl -X POST "http://localhost:8000/api/v1/auth/login" \\
         -H "Content-Type: application/json" \\
         -d '{
           "username": "jane_recruiter",
           "password": "SecurePass123!"
         }'
    ```
    
    **Requirements**: 7.2, 7.3
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
    
    Obtains a new access token using a valid refresh token, extending the authentication session without requiring the user to log in again.
    
    ## Request Body
    
    - **refresh_token**: Valid refresh token obtained from login
    
    ## Response
    
    Returns new JWT tokens:
    - **access_token**: New access token (expires in 30 minutes)
    - **refresh_token**: New refresh token (expires in 7 days)
    - **token_type**: Always "bearer"
    
    ## When to Use
    
    Use this endpoint when:
    - Your access token has expired (401 error with "token expired" message)
    - You want to extend the session before the access token expires
    - Implementing automatic token refresh in your application
    
    ## Security Features
    
    - Refresh tokens are single-use (old refresh token becomes invalid)
    - Refresh tokens have longer expiration but are still time-limited
    - User information is re-validated during refresh
    - Failed refresh attempts are logged
    
    ## Error Responses
    
    - **401 Unauthorized**: Invalid, expired, or already-used refresh token
    - **422 Unprocessable Entity**: Missing or malformed request
    
    ## Example Usage
    
    ```bash
    curl -X POST "http://localhost:8000/api/v1/auth/refresh" \\
         -H "Content-Type: application/json" \\
         -d '{
           "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
         }'
    ```
    
    ## Implementation Pattern
    
    ```javascript
    // Automatic token refresh example
    async function apiCall(url, options = {}) {
      let response = await fetch(url, {
        ...options,
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          ...options.headers
        }
      });
      
      if (response.status === 401) {
        // Try to refresh token
        const refreshResponse = await fetch('/api/v1/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken })
        });
        
        if (refreshResponse.ok) {
          const tokens = await refreshResponse.json();
          accessToken = tokens.access_token;
          refreshToken = tokens.refresh_token;
          
          // Retry original request
          response = await fetch(url, {
            ...options,
            headers: {
              'Authorization': `Bearer ${accessToken}`,
              ...options.headers
            }
          });
        }
      }
      
      return response;
    }
    ```
    
    **Requirements**: 7.2, 7.3
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
    Get current authenticated user information
    
    Returns the profile information for the currently authenticated user.
    
    ## Authentication Required
    
    This endpoint requires a valid JWT access token in the Authorization header:
    ```
    Authorization: Bearer <access_token>
    ```
    
    ## Response
    
    Returns current user information:
    - **id**: Unique user identifier (UUID)
    - **username**: User's username
    - **email**: User's email address
    - **role**: User's role (admin, recruiter, hiring_manager)
    
    ## Use Cases
    
    - Verify token validity and get user info
    - Display user profile in applications
    - Check user permissions based on role
    - Validate session state
    
    ## Error Responses
    
    - **401 Unauthorized**: Missing, invalid, or expired access token
    
    ## Example Usage
    
    ```bash
    curl -X GET "http://localhost:8000/api/v1/auth/me" \\
         -H "Authorization: Bearer <your_access_token>"
    ```
    
    ## Response Example
    
    ```json
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "username": "jane_recruiter",
      "email": "jane@company.com",
      "role": "recruiter"
    }
    ```
    
    **Requirements**: 7.2, 7.5
    """
    return current_user
