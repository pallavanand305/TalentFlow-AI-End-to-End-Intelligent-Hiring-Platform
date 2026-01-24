"""Enhanced OpenAPI documentation configuration"""

from typing import Dict, Any, List
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI

from backend.app.core.config import settings


def get_custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=get_api_description(),
        routes=app.routes,
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.talentflow.ai",
                "description": "Production server"
            }
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /api/v1/auth/login endpoint"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    # Add custom examples and enhanced descriptions
    enhance_endpoint_documentation(openapi_schema)
    
    # Add error response schemas
    add_error_schemas(openapi_schema)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def get_api_description() -> str:
    """Get comprehensive API description"""
    return """
# TalentFlow AI - Intelligent Hiring Platform API

An end-to-end AI-powered backend system that automates candidate screening and ranking using natural language processing and machine learning.

## 🚀 Key Features

* **Resume Parsing**: Extract structured data from PDF/DOCX resumes using advanced NLP
* **Semantic Matching**: ML-powered candidate-job similarity scoring with explainable AI
* **Background Processing**: Asynchronous task execution for long-running operations
* **MLOps Integration**: Model versioning and tracking with MLflow
* **Role-Based Access**: JWT authentication with granular permissions
* **Real-time Monitoring**: Comprehensive logging and performance metrics

## 🔐 Authentication

This API uses JWT (JSON Web Tokens) for authentication. To access protected endpoints:

### 1. Register a New User
```bash
POST /api/v1/auth/register
{
  "username": "john_doe",
  "email": "john@example.com", 
  "password": "secure_password123",
  "role": "recruiter"
}
```

### 2. Login to Get Tokens
```bash
POST /api/v1/auth/login
{
  "username": "john_doe",
  "password": "secure_password123"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

### 3. Use Token in Requests
Include the access token in the Authorization header:
```
Authorization: Bearer <access_token>
```

### 4. Refresh Expired Tokens
```bash
POST /api/v1/auth/refresh
{
  "refresh_token": "your_refresh_token_here"
}
```

## 👥 User Roles

| Role | Permissions |
|------|-------------|
| **admin** | Full system access, user management, model promotion |
| **recruiter** | Upload resumes, view candidates, manage candidate data |
| **hiring_manager** | Create jobs, view ranked candidates, manage job postings |

## 📊 Rate Limiting

API requests are rate-limited to **{rate_limit} requests per minute** per IP address.

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## 🔄 Asynchronous Operations

Long-running operations (resume parsing, batch scoring) are processed asynchronously:

1. **Submit Request**: Returns immediately with a `job_id`
2. **Track Progress**: Use `/api/v1/jobs/status/{{job_id}}` to monitor status
3. **Get Results**: Retrieve results when status is "completed"

### Job Status Values
- `queued`: Job is waiting to be processed
- `processing`: Job is currently being executed
- `completed`: Job finished successfully
- `failed`: Job encountered an error

## 📝 Request/Response Format

### Success Response Format
```json
{
  "data": { ... },
  "message": "Operation completed successfully",
  "request_id": "uuid-v4"
}
```

### Error Response Format
```json
{
  "error": "Error description",
  "details": {
    "field": "specific_field",
    "reason": "detailed explanation"
  },
  "request_id": "uuid-v4",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🎯 Common Use Cases

### Upload and Parse Resume
```bash
# 1. Upload resume file
POST /api/v1/resumes/upload
Content-Type: multipart/form-data

# 2. Check processing status
GET /api/v1/jobs/status/{{job_id}}

# 3. Get parsed resume data
GET /api/v1/resumes/{{resume_id}}
```

### Create Job and Find Candidates
```bash
# 1. Create job posting
POST /api/v1/jobs
{
  "title": "Senior Python Developer",
  "description": "We are looking for...",
  "required_skills": ["Python", "FastAPI", "PostgreSQL"],
  "experience_level": "senior"
}

# 2. Get ranked candidates
GET /api/v1/jobs/{{job_id}}/candidates?min_score=0.7

# 3. Get top candidates
GET /api/v1/jobs/{{job_id}}/top-candidates?limit=10
```

### Model Management
```bash
# 1. List available models
GET /api/v1/models

# 2. Get model details
GET /api/v1/models/{{model_name}}?version=1

# 3. Promote model to production
POST /api/v1/models/promote
{
  "model_name": "candidate_scorer_v2",
  "version": "3",
  "stage": "Production"
}
```

## 🔍 Search and Filtering

Most list endpoints support advanced filtering:

### Resume Search
```bash
GET /api/v1/resumes?query=python&skills=Python,FastAPI&min_experience_years=3
```

### Job Search  
```bash
GET /api/v1/jobs?query=developer&experience_level=senior&location=remote
```

## 📈 Monitoring and Health

### Health Check
```bash
GET /health
```

### MLflow Health
```bash
GET /api/v1/models/health
```

### Queue Statistics (Admin only)
```bash
GET /api/v1/jobs/stats
```

## 🚨 Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input data |
| 401 | Unauthorized - Missing or invalid token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 409 | Conflict - Resource already exists |
| 413 | Payload Too Large - File size exceeds limit |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Unexpected error |

## 📚 Additional Resources

- **GitHub Repository**: [https://github.com/your-org/talentflow-ai](https://github.com/your-org/talentflow-ai)
- **Documentation**: [https://docs.talentflow.ai](https://docs.talentflow.ai)
- **Support**: support@talentflow.ai

---

*Built with FastAPI, SQLAlchemy, and MLflow. Deployed on AWS with full CI/CD automation.*
    """.format(rate_limit=settings.RATE_LIMIT_PER_MINUTE)


def enhance_endpoint_documentation(openapi_schema: Dict[str, Any]) -> None:
    """Add enhanced examples and descriptions to endpoints"""
    
    # Authentication examples
    if "/api/v1/auth/register" in openapi_schema["paths"]:
        register_path = openapi_schema["paths"]["/api/v1/auth/register"]["post"]
        register_path["requestBody"]["content"]["application/json"]["examples"] = {
            "recruiter": {
                "summary": "Register as Recruiter",
                "description": "Register a new recruiter account",
                "value": {
                    "username": "jane_recruiter",
                    "email": "jane@company.com",
                    "password": "SecurePass123!",
                    "role": "recruiter"
                }
            },
            "hiring_manager": {
                "summary": "Register as Hiring Manager", 
                "description": "Register a new hiring manager account",
                "value": {
                    "username": "john_manager",
                    "email": "john@company.com",
                    "password": "SecurePass123!",
                    "role": "hiring_manager"
                }
            }
        }
    
    if "/api/v1/auth/login" in openapi_schema["paths"]:
        login_path = openapi_schema["paths"]["/api/v1/auth/login"]["post"]
        login_path["requestBody"]["content"]["application/json"]["examples"] = {
            "standard_login": {
                "summary": "Standard Login",
                "description": "Login with username and password",
                "value": {
                    "username": "jane_recruiter",
                    "password": "SecurePass123!"
                }
            }
        }
    
    # Resume upload examples
    if "/api/v1/resumes/upload" in openapi_schema["paths"]:
        upload_path = openapi_schema["paths"]["/api/v1/resumes/upload"]["post"]
        upload_path["requestBody"]["content"]["multipart/form-data"]["examples"] = {
            "pdf_resume": {
                "summary": "Upload PDF Resume",
                "description": "Upload a PDF resume with candidate information",
                "value": {
                    "file": "john_doe_resume.pdf",
                    "candidate_name": "John Doe",
                    "candidate_email": "john.doe@email.com"
                }
            },
            "docx_resume": {
                "summary": "Upload DOCX Resume", 
                "description": "Upload a Word document resume",
                "value": {
                    "file": "jane_smith_resume.docx",
                    "candidate_name": "Jane Smith",
                    "candidate_email": "jane.smith@email.com"
                }
            }
        }
    
    # Job creation examples
    if "/api/v1/jobs" in openapi_schema["paths"] and "post" in openapi_schema["paths"]["/api/v1/jobs"]:
        job_create_path = openapi_schema["paths"]["/api/v1/jobs"]["post"]
        job_create_path["requestBody"]["content"]["application/json"]["examples"] = {
            "senior_developer": {
                "summary": "Senior Developer Position",
                "description": "Example senior software developer job posting",
                "value": {
                    "title": "Senior Python Developer",
                    "description": "We are seeking an experienced Python developer to join our growing team. The ideal candidate will have strong experience with web frameworks, databases, and cloud technologies.",
                    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
                    "experience_level": "senior",
                    "location": "San Francisco, CA (Remote OK)",
                    "salary_min": 120000,
                    "salary_max": 180000
                }
            },
            "entry_level": {
                "summary": "Entry Level Position",
                "description": "Example entry-level job posting",
                "value": {
                    "title": "Junior Data Analyst",
                    "description": "Great opportunity for a recent graduate to start their career in data analysis. We provide mentorship and training.",
                    "required_skills": ["Python", "SQL", "Excel", "Statistics"],
                    "experience_level": "entry",
                    "location": "New York, NY",
                    "salary_min": 60000,
                    "salary_max": 80000
                }
            }
        }
    
    # Model promotion examples
    if "/api/v1/models/promote" in openapi_schema["paths"]:
        promote_path = openapi_schema["paths"]["/api/v1/models/promote"]["post"]
        promote_path["requestBody"]["content"]["application/json"]["examples"] = {
            "promote_to_production": {
                "summary": "Promote to Production",
                "description": "Promote a model version to production stage",
                "value": {
                    "model_name": "candidate_scorer_v2",
                    "version": "3",
                    "stage": "Production",
                    "archive_existing": True
                }
            },
            "promote_to_staging": {
                "summary": "Promote to Staging",
                "description": "Promote a model version to staging for testing",
                "value": {
                    "model_name": "resume_parser_v1",
                    "version": "2", 
                    "stage": "Staging",
                    "archive_existing": False
                }
            }
        }


def add_error_schemas(openapi_schema: Dict[str, Any]) -> None:
    """Add common error response schemas"""
    
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
    
    # Add error schemas
    openapi_schema["components"]["schemas"].update({
        "ErrorResponse": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "Human-readable error message"
                },
                "details": {
                    "type": "object",
                    "description": "Additional error details",
                    "additionalProperties": True
                },
                "request_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Unique request identifier for tracking"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Error timestamp"
                }
            },
            "required": ["error", "request_id"],
            "example": {
                "error": "Validation error",
                "details": {
                    "field": "email",
                    "reason": "Invalid email format"
                },
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        },
        "ValidationError": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "example": "Validation error"
                },
                "details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "loc": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Field location"
                            },
                            "msg": {
                                "type": "string",
                                "description": "Error message"
                            },
                            "type": {
                                "type": "string",
                                "description": "Error type"
                            }
                        }
                    }
                },
                "request_id": {
                    "type": "string",
                    "format": "uuid"
                }
            },
            "example": {
                "error": "Validation error",
                "details": [
                    {
                        "loc": ["body", "email"],
                        "msg": "field required",
                        "type": "value_error.missing"
                    }
                ],
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        },
        "AuthenticationError": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "example": "Authentication required"
                },
                "details": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "example": "Missing or invalid authorization token"
                        }
                    }
                },
                "request_id": {
                    "type": "string",
                    "format": "uuid"
                }
            }
        },
        "RateLimitError": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "example": "Rate limit exceeded"
                },
                "details": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "example": 100
                        },
                        "window": {
                            "type": "string",
                            "example": "1 minute"
                        },
                        "retry_after": {
                            "type": "integer",
                            "example": 60,
                            "description": "Seconds until rate limit resets"
                        }
                    }
                },
                "request_id": {
                    "type": "string",
                    "format": "uuid"
                }
            }
        }
    })


def get_response_examples() -> Dict[str, Any]:
    """Get common response examples"""
    return {
        "resume_upload_success": {
            "summary": "Resume Upload Success",
            "description": "Successful resume upload response",
            "value": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "Resume upload successful. Processing in background.",
                "status": "processing"
            }
        },
        "job_creation_success": {
            "summary": "Job Creation Success", 
            "description": "Successful job creation response",
            "value": {
                "id": "456e7890-e89b-12d3-a456-426614174000",
                "title": "Senior Python Developer",
                "description": "We are seeking an experienced Python developer...",
                "required_skills": ["Python", "FastAPI", "PostgreSQL"],
                "experience_level": "senior",
                "status": "active",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        },
        "candidate_ranking": {
            "summary": "Candidate Ranking",
            "description": "Ranked candidates for a job",
            "value": [
                {
                    "candidate_id": "789e0123-e89b-12d3-a456-426614174000",
                    "candidate_name": "John Doe",
                    "score": 0.92,
                    "rank": 1,
                    "explanation": "Strong match in Python, FastAPI, and database skills. 5+ years experience aligns well with senior requirements.",
                    "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
                    "experience_years": 6,
                    "education_level": "Bachelor's Degree"
                },
                {
                    "candidate_id": "012e3456-e89b-12d3-a456-426614174000", 
                    "candidate_name": "Jane Smith",
                    "score": 0.87,
                    "rank": 2,
                    "explanation": "Excellent technical skills match. Slightly less experience but strong educational background.",
                    "skills": ["Python", "Django", "PostgreSQL", "Redis"],
                    "experience_years": 4,
                    "education_level": "Master's Degree"
                }
            ]
        }
    }