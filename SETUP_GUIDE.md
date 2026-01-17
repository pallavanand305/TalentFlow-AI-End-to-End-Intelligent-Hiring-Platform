# TalentFlow AI - Setup and Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git
- GitHub account

## 📦 Initial Setup

### 1. Initialize Git Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: TalentFlow AI - Intelligent Hiring Platform

- Complete project structure with FastAPI backend
- Database schema with Alembic migrations
- JWT authentication with role-based access control
- Custom middleware (logging, rate limiting, request tracking)
- Resume parsing foundation (text extraction, section identification)
- Comprehensive test suite (property-based + unit tests)
- Docker Compose setup with PostgreSQL, Redis, MLflow
- OpenAPI documentation
- CI/CD ready structure"
```

### 2. Create GitHub Repository

**Option A: Via GitHub CLI**
```bash
# Install GitHub CLI if needed
# Windows: winget install GitHub.cli
# Mac: brew install gh

# Login to GitHub
gh auth login

# Create repository
gh repo create TalentFlow-AI --public --description "AI-powered intelligent hiring platform with ML-based candidate screening"

# Push code
git branch -M main
git push -u origin main
```

**Option B: Via GitHub Web Interface**
1. Go to https://github.com/new
2. Repository name: `TalentFlow-AI`
3. Description: `AI-powered intelligent hiring platform with ML-based candidate screening`
4. Choose Public or Private
5. Don't initialize with README (we already have one)
6. Click "Create repository"

Then push:
```bash
git remote add origin https://github.com/YOUR_USERNAME/TalentFlow-AI.git
git branch -M main
git push -u origin main
```

### 3. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# IMPORTANT: Change SECRET_KEY in production!
# Generate a secure key: python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Start Services

```bash
# Start Docker services
docker-compose up -d

# Wait for services to be ready (about 10 seconds)
# Check status
docker-compose ps

# You should see:
# - talentflow-postgres (port 5432)
# - talentflow-redis (port 6379)
# - talentflow-mlflow (port 5000)
```

### 5. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 6. Run Database Migrations

```bash
# Run migrations
alembic upgrade head

# Verify tables were created
docker-compose exec postgres psql -U postgres -d talentflow -c "\dt"
```

### 7. Start the Application

```bash
# Start FastAPI server
uvicorn backend.app.main:app --reload

# Server will start at http://localhost:8000
# API docs at http://localhost:8000/docs
# MLflow UI at http://localhost:5000
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov=ml --cov-report=html

# Run specific test types
pytest tests/unit/
pytest tests/property/
pytest tests/integration/

# View coverage report
# Open htmlcov/index.html in browser
```

## 🔧 Development Workflow

### Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes...

# Run tests
pytest

# Run linting
black .
pylint backend/ ml/
mypy backend/ ml/

# Commit changes
git add .
git commit -m "feat: your feature description"

# Push to GitHub
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

### Code Quality

```bash
# Format code
black .

# Lint code
pylint backend/ ml/

# Type checking
mypy backend/ ml/

# Run all quality checks
black . && pylint backend/ ml/ && mypy backend/ ml/ && pytest
```

## 📊 Accessing Services

- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **MLflow UI**: http://localhost:5000
- **PostgreSQL**: localhost:5432 (user: postgres, password: postgres, db: talentflow)
- **Redis**: localhost:6379

## 🔐 First User Setup

```bash
# Using curl (Windows: use Git Bash or WSL)
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@talentflow.ai",
    "password": "SecurePassword123!",
    "role": "admin"
  }'

# Login to get token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "SecurePassword123!"
  }'
```

Or use the interactive API docs at http://localhost:8000/docs

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### Port Already in Use
```bash
# Check what's using the port
# Windows:
netstat -ano | findstr :8000
# Mac/Linux:
lsof -i :8000

# Kill the process or change port in .env
```

### Migration Issues
```bash
# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
alembic upgrade head
```

## 📝 Next Steps

1. ✅ Push code to GitHub
2. ✅ Set up local development environment
3. ✅ Run tests to verify everything works
4. 🔄 Continue implementing remaining features:
   - Resume parsing (entity extraction)
   - Candidate management
   - Job management
   - ML scoring engine
   - Background processing
   - MLOps integration
   - AWS deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

**Built with ❤️ for intelligent hiring**
