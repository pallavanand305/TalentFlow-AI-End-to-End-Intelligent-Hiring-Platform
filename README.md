# TalentFlow AI - Intelligent Hiring Platform

An end-to-end AI-powered backend system that ingests resumes, scores candidates against job descriptions, deploys ML models on AWS, and runs fully automated CI/CD + MLOps pipelines.

## 🚀 Features

- **Resume Parsing**: Extract structured data from PDF/DOCX resumes using NLP
- **Semantic Matching**: ML-powered candidate-job similarity scoring
- **REST APIs**: FastAPI-based async APIs with OpenAPI documentation
- **MLOps**: Model versioning, tracking, and deployment with MLflow
- **Background Processing**: Async task execution with Celery/Redis
- **Authentication**: JWT-based auth with role-based access control
- **Cloud-Native**: AWS deployment with Terraform IaC
- **CI/CD**: Automated testing, building, and deployment with GitHub Actions

## 🏗️ Architecture

```
TalentFlow-AI/
├── backend/              # FastAPI application
│   └── app/
│       ├── api/          # API routers
│       ├── core/         # Configuration & utilities
│       ├── models/       # Database models
│       ├── schemas/      # Pydantic schemas
│       ├── services/     # Business logic
│       └── repositories/ # Data access layer
├── ml/                   # ML pipeline
│   ├── training/         # Model training scripts
│   ├── inference/        # Model inference
│   └── drift_detection/  # Data drift monitoring
├── infra/                # Infrastructure as code
│   └── terraform/        # AWS Terraform configs
├── docker/               # Dockerfiles
├── tests/                # Test suite
│   ├── unit/
│   ├── integration/
│   └── property/         # Property-based tests
└── .github/workflows/    # CI/CD pipelines
```

## 🛠️ Tech Stack

**Backend**: FastAPI, SQLAlchemy, Pydantic, Redis

**ML/AI**: scikit-learn, Sentence Transformers, spaCy, PyPDF2, Hypothesis

**MLOps**: MLflow, Model Registry, Experiment Tracking

**Cloud**: AWS (ECS, RDS, S3, ECR, CloudWatch)

**DevOps**: Docker, Terraform, GitHub Actions

**Database**: PostgreSQL

## 🚦 Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL (or use Docker)

### Local Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/pallavanand305/TalentFlow-AI-End-to-End-Intelligent-Hiring-Platform.git
cd TalentFlow-AI-End-to-End-Intelligent-Hiring-Platform

```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start services with Docker Compose**
```bash
docker-compose up -d
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- MLflow (port 5000)

6. **Run database setup script**
```bash
chmod +x scripts/setup_db.sh
./scripts/setup_db.sh
```

Or manually:
```bash
# Start services
docker-compose up -d

# Run migrations
alembic upgrade head
```

7. **Start the development server**
```bash
uvicorn backend.app.main:app --reload
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

### Running Background Workers

For background processing (resume parsing, batch scoring):

```bash
# Start background workers
python scripts/worker.py --workers 2

# Or with Docker
docker-compose up worker
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov=ml --cov-report=html

# Run specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/property/
```

### Code Quality

```bash
# Format code
black .

# Lint code
pylint backend/ ml/

# Type checking
mypy backend/ ml/

# Run pre-commit hooks
pre-commit run --all-files
```

## 📊 MLflow Tracking

Access MLflow UI at `http://localhost:5000` to:
- Track experiments and metrics
- Compare model versions
- Manage model registry
- View artifacts

## 🔐 Authentication

The API uses JWT tokens for authentication. Default roles:
- `admin`: Full system access
- `recruiter`: Manage resumes and view candidates
- `hiring_manager`: Create jobs and view ranked candidates

## 📝 API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration

### Resumes
- `POST /api/v1/resumes/upload` - Upload resume
- `GET /api/v1/resumes/{id}` - Get resume details
- `GET /api/v1/resumes` - List resumes

### Jobs
- `POST /api/v1/jobs` - Create job
- `GET /api/v1/jobs/{id}` - Get job details
- `PUT /api/v1/jobs/{id}` - Update job
- `GET /api/v1/jobs` - List jobs

### Scoring
- `POST /api/v1/scores/compute` - Score candidate for job
- `GET /api/v1/scores/{id}` - Get score details
- `POST /api/v1/scores/{id}/explain` - Generate score explanation
- `GET /api/v1/jobs/{id}/candidates` - Get ranked candidates
- `GET /api/v1/jobs/{id}/top-candidates` - Get top N candidates

### Models
- `GET /api/v1/models` - List all models
- `GET /api/v1/models/{name}` - Get model details
- `POST /api/v1/models/promote` - Promote model to production
- `POST /api/v1/models/compare` - Compare model versions
- `GET /api/v1/models/health` - Check MLflow health

### Background Jobs
- `GET /api/v1/jobs/status/{id}` - Get job status
- `GET /api/v1/jobs/stats` - Get queue statistics (admin)
- `POST /api/v1/jobs/cleanup` - Clean up old jobs (admin)

## 🚀 Deployment

### AWS Deployment

1. **Configure AWS credentials**
```bash
aws configure
```

2. **Deploy infrastructure with Terraform**
```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

3. **Build and push Docker images**
```bash
docker build -f docker/Dockerfile.backend -t talentflow-backend .
docker tag talentflow-backend:latest <ecr-repo>/talentflow-backend:latest
docker push <ecr-repo>/talentflow-backend:latest
```

4. **Deploy to ECS**
CI/CD pipeline automatically deploys on merge to main branch.

## 🧪 Testing Strategy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test API endpoints and workflows
- **Property Tests**: Use Hypothesis for property-based testing
- **Coverage Target**: 80%+

## 📈 Monitoring

- **CloudWatch**: Logs and metrics
- **MLflow**: Model performance tracking
- **Drift Detection**: Automated data drift monitoring

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## 📄 License

MIT License

## 🎯 Status

### ✅ Completed Features
- **Core MVP**: Resume parsing, Job management, Candidate scoring
- **Authentication**: JWT-based auth with role-based access control
- **MLOps Integration**: MLflow model versioning and tracking
- **Background Processing**: Redis-based async task queue
- **API Documentation**: Comprehensive OpenAPI/Swagger docs
- **Testing**: Unit, integration, and property-based tests
- **Score Explanations**: Template-based explanation generation

### 🚧 In Progress
- AWS deployment infrastructure
- CI/CD pipelines
- Advanced monitoring and alerting

### 📋 Planned
- LLM-powered explanations
- Advanced drift detection
- Performance optimization
- Real-time monitoring dashboards

## 📧 Contact

For questions or support, please open an issue.

---

Built with ❤️ for intelligent hiring
