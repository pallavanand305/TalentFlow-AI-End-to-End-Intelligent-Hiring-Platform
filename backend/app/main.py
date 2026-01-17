"""FastAPI application entry point"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging, get_logger
from backend.app.core.middleware import (
    RequestIDMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware
)
from backend.app.core.exceptions import TalentFlowException

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## TalentFlow AI - Intelligent Hiring Platform

An end-to-end AI-powered backend system that automates candidate screening and ranking.

### Features

* **Resume Parsing**: Extract structured data from PDF/DOCX resumes using NLP
* **Semantic Matching**: ML-powered candidate-job similarity scoring
* **Background Processing**: Async task execution for long-running operations
* **MLOps**: Model versioning and tracking with MLflow
* **Authentication**: JWT-based auth with role-based access control

### Authentication

Most endpoints require authentication. To authenticate:

1. Register a new user at `/api/v1/auth/register`
2. Login at `/api/v1/auth/login` to get an access token
3. Include the token in the `Authorization` header: `Bearer <token>`

### Rate Limiting

API requests are rate-limited to {rate_limit} requests per minute per IP address.

### Roles

* **admin**: Full system access
* **recruiter**: Manage resumes and view candidates
* **hiring_manager**: Create jobs and view ranked candidates
    """.format(rate_limit=settings.RATE_LIMIT_PER_MINUTE),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "User registration, login, and token management"
        },
        {
            "name": "Resumes",
            "description": "Resume upload, parsing, and candidate management"
        },
        {
            "name": "Jobs",
            "description": "Job posting creation and management"
        },
        {
            "name": "Scoring",
            "description": "Candidate-job matching and ranking"
        },
        {
            "name": "Models",
            "description": "ML model management and versioning"
        },
        {
            "name": "Background Jobs",
            "description": "Async task status tracking"
        },
    ],
    contact={
        "name": "TalentFlow AI Support",
        "email": "support@talentflow.ai",
    },
    license_info={
        "name": "MIT License",
    },
)

# Add custom middleware (order matters - first added is outermost)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(TalentFlowException)
async def talentflow_exception_handler(request: Request, exc: TalentFlowException):
    """Handle custom TalentFlow exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"TalentFlow exception: {exc.message}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "details": exc.details,
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "request_id": request_id,
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        f"Validation error: {exc.errors()}",
        extra={"request_id": request_id}
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "request_id": request_id,
        }
    )


@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    """Handle database integrity errors"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Database integrity error: {str(exc)}",
        extra={"request_id": request_id},
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "error": "Database constraint violation",
            "details": {"message": "The operation violates a database constraint"},
            "request_id": request_id,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={"request_id": request_id},
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "details": {"message": "An unexpected error occurred"},
            "request_id": request_id,
        }
    )


@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down application")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# API routers
from backend.app.api import auth

app.include_router(auth.router, prefix=f"{settings.API_V1_PREFIX}/auth", tags=["Authentication"])

# Additional routers will be added as we build them
# from backend.app.api import resumes, jobs, scores
# app.include_router(resumes.router, prefix=f"{settings.API_V1_PREFIX}/resumes", tags=["resumes"])
# app.include_router(jobs.router, prefix=f"{settings.API_V1_PREFIX}/jobs", tags=["jobs"])
# app.include_router(scores.router, prefix=f"{settings.API_V1_PREFIX}/scores", tags=["scores"])
