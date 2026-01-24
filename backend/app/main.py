"""FastAPI application entry point"""

from contextlib import asynccontextmanager
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
from backend.app.core.documentation import get_custom_openapi

# Setup logging
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize task queue connection
    try:
        from backend.app.core.task_queue import task_queue
        await task_queue.connect()
        logger.info("Connected to Redis task queue")
    except Exception as e:
        logger.error(f"Failed to connect to Redis task queue: {e}")
        # Don't fail startup if Redis is not available in development
        if settings.ENVIRONMENT == "production":
            raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "User registration, login, and token management",
            "externalDocs": {
                "description": "Authentication Guide",
                "url": "https://docs.talentflow.ai/auth"
            }
        },
        {
            "name": "Resumes",
            "description": "Resume upload, parsing, and candidate management",
            "externalDocs": {
                "description": "Resume Processing Guide",
                "url": "https://docs.talentflow.ai/resumes"
            }
        },
        {
            "name": "Jobs",
            "description": "Job posting creation and management",
            "externalDocs": {
                "description": "Job Management Guide", 
                "url": "https://docs.talentflow.ai/jobs"
            }
        },
        {
            "name": "Scoring",
            "description": "Candidate-job matching and ranking",
            "externalDocs": {
                "description": "Scoring Algorithm Guide",
                "url": "https://docs.talentflow.ai/scoring"
            }
        },
        {
            "name": "Models",
            "description": "ML model management and versioning",
            "externalDocs": {
                "description": "MLOps Guide",
                "url": "https://docs.talentflow.ai/mlops"
            }
        },
        {
            "name": "Background Jobs",
            "description": "Async task status tracking",
            "externalDocs": {
                "description": "Background Processing Guide",
                "url": "https://docs.talentflow.ai/background-jobs"
            }
        },
    ],
    contact={
        "name": "TalentFlow AI Support",
        "email": "support@talentflow.ai",
        "url": "https://support.talentflow.ai"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    terms_of_service="https://talentflow.ai/terms"
)

# Set custom OpenAPI schema
app.openapi = lambda: get_custom_openapi(app)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize task queue connection
    try:
        from backend.app.core.task_queue import task_queue
        await task_queue.connect()
        logger.info("Connected to Redis task queue")
    except Exception as e:
        logger.error(f"Failed to connect to Redis task queue: {e}")
        # Don't fail startup if Redis is not available in development
        if settings.ENVIRONMENT == "production":
            raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Disconnect from task queue
    try:
        from backend.app.core.task_queue import task_queue
        await task_queue.disconnect()
        logger.info("Disconnected from Redis task queue")
    except Exception as e:
        logger.error(f"Error disconnecting from Redis task queue: {e}")


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
from backend.app.api import auth, resumes, jobs, background_jobs, models

app.include_router(auth.router, prefix=f"{settings.API_V1_PREFIX}/auth", tags=["Authentication"])
app.include_router(resumes.router, prefix=f"{settings.API_V1_PREFIX}/resumes", tags=["Resumes"])
app.include_router(jobs.router, prefix=f"{settings.API_V1_PREFIX}/jobs", tags=["Jobs"])
# app.include_router(scores.router, prefix=f"{settings.API_V1_PREFIX}/scores", tags=["Scoring"])
app.include_router(models.router, prefix=f"{settings.API_V1_PREFIX}/models", tags=["Models"])
app.include_router(background_jobs.router, prefix=f"{settings.API_V1_PREFIX}", tags=["Background Jobs"])
