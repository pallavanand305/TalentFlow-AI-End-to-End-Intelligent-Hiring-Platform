"""OpenAPI examples for request/response schemas"""

from typing import Dict, Any


class AuthExamples:
    """Authentication endpoint examples"""
    
    REGISTER_REQUEST = {
        "recruiter": {
            "summary": "Register as Recruiter",
            "description": "Register a new recruiter account with standard permissions",
            "value": {
                "username": "jane_recruiter",
                "email": "jane@company.com",
                "password": "SecurePass123!",
                "role": "recruiter"
            }
        },
        "hiring_manager": {
            "summary": "Register as Hiring Manager",
            "description": "Register a new hiring manager account with job creation permissions",
            "value": {
                "username": "john_manager",
                "email": "john@company.com",
                "password": "SecurePass123!",
                "role": "hiring_manager"
            }
        },
        "admin": {
            "summary": "Register as Admin",
            "description": "Register a new admin account with full system access",
            "value": {
                "username": "admin_user",
                "email": "admin@company.com",
                "password": "SuperSecurePass123!",
                "role": "admin"
            }
        }
    }
    
    LOGIN_REQUEST = {
        "standard_login": {
            "summary": "Standard Login",
            "description": "Login with username and password to get JWT tokens",
            "value": {
                "username": "jane_recruiter",
                "password": "SecurePass123!"
            }
        }
    }
    
    TOKEN_RESPONSE = {
        "successful_login": {
            "summary": "Successful Login Response",
            "description": "JWT tokens returned after successful authentication",
            "value": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ",
                "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ",
                "token_type": "bearer"
            }
        }
    }
    
    USER_RESPONSE = {
        "user_profile": {
            "summary": "User Profile",
            "description": "User information returned after registration or profile query",
            "value": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "username": "jane_recruiter",
                "email": "jane@company.com",
                "role": "recruiter"
            }
        }
    }


class ResumeExamples:
    """Resume endpoint examples"""
    
    UPLOAD_RESPONSE = {
        "pdf_upload": {
            "summary": "PDF Resume Upload Success",
            "description": "Response after successfully uploading a PDF resume",
            "value": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "Resume upload successful. Processing in background.",
                "status": "processing"
            }
        },
        "docx_upload": {
            "summary": "DOCX Resume Upload Success",
            "description": "Response after successfully uploading a Word document resume",
            "value": {
                "job_id": "456e7890-e89b-12d3-a456-426614174000",
                "message": "Resume upload successful. Processing in background.",
                "status": "processing"
            }
        }
    }
    
    RESUME_DETAIL = {
        "parsed_resume": {
            "summary": "Parsed Resume Details",
            "description": "Complete parsed resume data with structured information",
            "value": {
                "candidate_id": "789e0123-e89b-12d3-a456-426614174000",
                "parsed_resume": {
                    "candidate_name": "John Doe",
                    "email": "john.doe@email.com",
                    "phone": "+1-555-123-4567",
                    "work_experience": [
                        {
                            "company": "Tech Corp",
                            "title": "Senior Software Engineer",
                            "start_date": "2020-01",
                            "end_date": "2024-01",
                            "description": "Led development of microservices architecture using Python and FastAPI",
                            "confidence": 0.95
                        },
                        {
                            "company": "StartupXYZ",
                            "title": "Full Stack Developer",
                            "start_date": "2018-06",
                            "end_date": "2019-12",
                            "description": "Built web applications using React and Node.js",
                            "confidence": 0.88
                        }
                    ],
                    "education": [
                        {
                            "institution": "University of Technology",
                            "degree": "Bachelor of Science",
                            "field_of_study": "Computer Science",
                            "graduation_date": "2018-05",
                            "confidence": 0.92
                        }
                    ],
                    "skills": ["Python", "FastAPI", "React", "PostgreSQL", "Docker", "AWS"],
                    "certifications": ["AWS Certified Developer", "Python Institute PCAP"],
                    "confidence_scores": {
                        "name": 0.98,
                        "contact": 0.95,
                        "experience": 0.91,
                        "education": 0.92,
                        "skills": 0.89
                    }
                },
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }
    }
    
    RESUME_LIST = {
        "candidate_list": {
            "summary": "Candidate List",
            "description": "List of candidates with basic information",
            "value": [
                {
                    "candidate_id": "789e0123-e89b-12d3-a456-426614174000",
                    "name": "John Doe",
                    "email": "john.doe@email.com",
                    "skills": ["Python", "FastAPI", "PostgreSQL"],
                    "experience_years": 6,
                    "education_level": "Bachelor's Degree",
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-15T10:30:00Z"
                },
                {
                    "candidate_id": "012e3456-e89b-12d3-a456-426614174000",
                    "name": "Jane Smith",
                    "email": "jane.smith@email.com",
                    "skills": ["Python", "Django", "React"],
                    "experience_years": 4,
                    "education_level": "Master's Degree",
                    "created_at": "2024-01-14T15:20:00Z",
                    "updated_at": "2024-01-14T15:20:00Z"
                }
            ]
        }
    }


class JobExamples:
    """Job endpoint examples"""
    
    CREATE_REQUEST = {
        "senior_developer": {
            "summary": "Senior Developer Position",
            "description": "Example senior software developer job posting with competitive salary",
            "value": {
                "title": "Senior Python Developer",
                "description": "We are seeking an experienced Python developer to join our growing team. The ideal candidate will have strong experience with web frameworks, databases, and cloud technologies. You'll be working on high-scale applications serving millions of users.",
                "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
                "experience_level": "senior",
                "location": "San Francisco, CA (Remote OK)",
                "salary_min": 120000,
                "salary_max": 180000
            }
        },
        "entry_level": {
            "summary": "Entry Level Position",
            "description": "Example entry-level job posting with growth opportunities",
            "value": {
                "title": "Junior Data Analyst",
                "description": "Great opportunity for a recent graduate to start their career in data analysis. We provide mentorship, training, and a clear path for career advancement. You'll work with our data science team to analyze user behavior and business metrics.",
                "required_skills": ["Python", "SQL", "Excel", "Statistics"],
                "experience_level": "entry",
                "location": "New York, NY",
                "salary_min": 60000,
                "salary_max": 80000
            }
        },
        "remote_position": {
            "summary": "Remote Position",
            "description": "Example fully remote job posting",
            "value": {
                "title": "Full Stack Engineer (Remote)",
                "description": "Join our distributed team building the next generation of SaaS tools. We're looking for a versatile engineer comfortable with both frontend and backend development.",
                "required_skills": ["JavaScript", "React", "Node.js", "MongoDB"],
                "experience_level": "mid",
                "location": "Remote (US timezone)",
                "salary_min": 90000,
                "salary_max": 130000
            }
        }
    }
    
    JOB_RESPONSE = {
        "created_job": {
            "summary": "Created Job Response",
            "description": "Response after successfully creating a job posting",
            "value": {
                "id": "456e7890-e89b-12d3-a456-426614174000",
                "title": "Senior Python Developer",
                "description": "We are seeking an experienced Python developer...",
                "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
                "experience_level": "senior",
                "location": "San Francisco, CA (Remote OK)",
                "salary_min": 120000,
                "salary_max": 180000,
                "status": "active",
                "created_by": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }
    }
    
    CANDIDATE_RANKING = {
        "top_candidates": {
            "summary": "Top Candidates for Job",
            "description": "Ranked list of candidates for a specific job position",
            "value": [
                {
                    "candidate_id": "789e0123-e89b-12d3-a456-426614174000",
                    "candidate_name": "John Doe",
                    "score": 0.92,
                    "rank": 1,
                    "explanation": "Strong match in Python, FastAPI, and database skills. 6+ years experience aligns well with senior requirements. AWS certification is a plus.",
                    "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
                    "experience_years": 6,
                    "education_level": "Bachelor's Degree",
                    "created_at": "2024-01-15T11:00:00Z"
                },
                {
                    "candidate_id": "012e3456-e89b-12d3-a456-426614174000",
                    "candidate_name": "Jane Smith",
                    "score": 0.87,
                    "rank": 2,
                    "explanation": "Excellent technical skills match with Python and databases. Slightly less experience but strong educational background and proven track record.",
                    "skills": ["Python", "Django", "PostgreSQL", "Redis"],
                    "experience_years": 4,
                    "education_level": "Master's Degree",
                    "created_at": "2024-01-15T11:05:00Z"
                },
                {
                    "candidate_id": "345e6789-e89b-12d3-a456-426614174000",
                    "candidate_name": "Mike Johnson",
                    "score": 0.81,
                    "rank": 3,
                    "explanation": "Good technical foundation with Python and web frameworks. Some experience with cloud technologies. Room for growth in senior-level responsibilities.",
                    "skills": ["Python", "Flask", "MySQL", "Docker"],
                    "experience_years": 3,
                    "education_level": "Bachelor's Degree",
                    "created_at": "2024-01-15T11:10:00Z"
                }
            ]
        }
    }


class ModelExamples:
    """Model management endpoint examples"""
    
    MODEL_LIST = {
        "available_models": {
            "summary": "Available Models",
            "description": "List of registered models in the MLflow registry",
            "value": [
                {
                    "name": "candidate_scorer_v2",
                    "description": "Advanced semantic similarity model for candidate-job matching",
                    "creation_timestamp": "2024-01-10T09:00:00Z",
                    "last_updated_timestamp": "2024-01-15T14:30:00Z",
                    "latest_versions": [
                        {
                            "version": "3",
                            "stage": "Production",
                            "run_id": "abc123def456",
                            "created_at": "2024-01-15T14:30:00Z"
                        },
                        {
                            "version": "4",
                            "stage": "Staging",
                            "run_id": "def456ghi789",
                            "created_at": "2024-01-15T16:00:00Z"
                        }
                    ]
                },
                {
                    "name": "resume_parser_v1",
                    "description": "NLP model for extracting structured data from resumes",
                    "creation_timestamp": "2024-01-05T10:00:00Z",
                    "last_updated_timestamp": "2024-01-12T11:15:00Z",
                    "latest_versions": [
                        {
                            "version": "2",
                            "stage": "Production",
                            "run_id": "ghi789jkl012",
                            "created_at": "2024-01-12T11:15:00Z"
                        }
                    ]
                }
            ]
        }
    }
    
    PROMOTE_REQUEST = {
        "promote_to_production": {
            "summary": "Promote to Production",
            "description": "Promote a model version to production stage",
            "value": {
                "model_name": "candidate_scorer_v2",
                "version": "4",
                "stage": "Production",
                "archive_existing": True
            }
        },
        "promote_to_staging": {
            "summary": "Promote to Staging",
            "description": "Promote a model version to staging for testing",
            "value": {
                "model_name": "resume_parser_v1",
                "version": "3",
                "stage": "Staging",
                "archive_existing": False
            }
        }
    }
    
    MODEL_COMPARISON = {
        "version_comparison": {
            "summary": "Model Version Comparison",
            "description": "Comparison of metrics across different model versions",
            "value": {
                "model_name": "candidate_scorer_v2",
                "comparison_data": [
                    {
                        "version": "3",
                        "stage": "Production",
                        "accuracy": 0.89,
                        "precision": 0.87,
                        "recall": 0.91,
                        "f1_score": 0.89,
                        "training_time": 3600,
                        "model_size_mb": 45.2
                    },
                    {
                        "version": "4",
                        "stage": "Staging",
                        "accuracy": 0.92,
                        "precision": 0.90,
                        "recall": 0.94,
                        "f1_score": 0.92,
                        "training_time": 4200,
                        "model_size_mb": 52.1
                    }
                ]
            }
        }
    }


class BackgroundJobExamples:
    """Background job endpoint examples"""
    
    JOB_STATUS = {
        "processing": {
            "summary": "Job in Progress",
            "description": "Status of a background job that is currently processing",
            "value": {
                "success": True,
                "data": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "job_type": "resume_parsing",
                    "status": "processing",
                    "input_data": {
                        "filename": "john_doe_resume.pdf",
                        "candidate_name": "John Doe",
                        "uploaded_by": "456e7890-e89b-12d3-a456-426614174000"
                    },
                    "created_at": "2024-01-15T10:30:00Z",
                    "started_at": "2024-01-15T10:30:15Z",
                    "progress": 65
                }
            }
        },
        "completed": {
            "summary": "Job Completed",
            "description": "Status of a successfully completed background job",
            "value": {
                "success": True,
                "data": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "job_type": "resume_parsing",
                    "status": "completed",
                    "input_data": {
                        "filename": "john_doe_resume.pdf",
                        "candidate_name": "John Doe",
                        "uploaded_by": "456e7890-e89b-12d3-a456-426614174000"
                    },
                    "result_data": {
                        "candidate_id": "789e0123-e89b-12d3-a456-426614174000",
                        "parsing_success": True,
                        "sections_found": 4,
                        "skills_extracted": 12,
                        "experience_entries": 3,
                        "education_entries": 1
                    },
                    "created_at": "2024-01-15T10:30:00Z",
                    "started_at": "2024-01-15T10:30:15Z",
                    "completed_at": "2024-01-15T10:32:30Z"
                }
            }
        },
        "failed": {
            "summary": "Job Failed",
            "description": "Status of a background job that encountered an error",
            "value": {
                "success": True,
                "data": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "job_type": "resume_parsing",
                    "status": "failed",
                    "input_data": {
                        "filename": "corrupted_resume.pdf",
                        "candidate_name": "Jane Doe",
                        "uploaded_by": "456e7890-e89b-12d3-a456-426614174000"
                    },
                    "error_message": "Unable to extract text from PDF: File appears to be corrupted",
                    "created_at": "2024-01-15T10:30:00Z",
                    "started_at": "2024-01-15T10:30:15Z",
                    "completed_at": "2024-01-15T10:30:45Z"
                }
            }
        }
    }


class ErrorExamples:
    """Common error response examples"""
    
    VALIDATION_ERROR = {
        "field_validation": {
            "summary": "Field Validation Error",
            "description": "Error when request data fails validation",
            "value": {
                "error": "Validation error",
                "details": [
                    {
                        "loc": ["body", "email"],
                        "msg": "field required",
                        "type": "value_error.missing"
                    },
                    {
                        "loc": ["body", "password"],
                        "msg": "ensure this value has at least 8 characters",
                        "type": "value_error.any_str.min_length"
                    }
                ],
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    }
    
    AUTHENTICATION_ERROR = {
        "missing_token": {
            "summary": "Missing Authentication Token",
            "description": "Error when no authentication token is provided",
            "value": {
                "error": "Authentication required",
                "details": {
                    "message": "Missing authorization header"
                },
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        },
        "invalid_token": {
            "summary": "Invalid Authentication Token",
            "description": "Error when authentication token is invalid or expired",
            "value": {
                "error": "Invalid or expired token",
                "details": {
                    "message": "Token signature verification failed"
                },
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    }
    
    RATE_LIMIT_ERROR = {
        "rate_exceeded": {
            "summary": "Rate Limit Exceeded",
            "description": "Error when API rate limit is exceeded",
            "value": {
                "error": "Rate limit exceeded",
                "details": {
                    "limit": 100,
                    "window": "1 minute",
                    "retry_after": 45
                },
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    }
    
    NOT_FOUND_ERROR = {
        "resource_not_found": {
            "summary": "Resource Not Found",
            "description": "Error when requested resource doesn't exist",
            "value": {
                "error": "Resource not found",
                "details": {
                    "message": "Job with ID '123e4567-e89b-12d3-a456-426614174000' not found"
                },
                "request_id": "456e7890-e89b-12d3-a456-426614174000"
            }
        }
    }