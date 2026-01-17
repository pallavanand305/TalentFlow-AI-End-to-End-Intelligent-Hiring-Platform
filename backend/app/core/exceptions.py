"""Custom exception classes"""

from typing import Any, Optional


class TalentFlowException(Exception):
    """Base exception for TalentFlow AI"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(TalentFlowException):
    """Exception for validation errors"""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class AuthenticationException(TalentFlowException):
    """Exception for authentication errors"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class AuthorizationException(TalentFlowException):
    """Exception for authorization errors"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, status_code=403)


class NotFoundException(TalentFlowException):
    """Exception for resource not found errors"""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class ConflictException(TalentFlowException):
    """Exception for resource conflict errors"""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=409, details=details)


class RateLimitException(TalentFlowException):
    """Exception for rate limit errors"""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


class ExternalServiceException(TalentFlowException):
    """Exception for external service errors"""
    
    def __init__(self, service: str, message: str):
        full_message = f"External service error ({service}): {message}"
        super().__init__(full_message, status_code=502)


class BackgroundJobException(TalentFlowException):
    """Exception for background job errors"""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)
