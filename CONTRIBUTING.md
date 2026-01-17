# Contributing to TalentFlow AI

Thank you for your interest in contributing to TalentFlow AI! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/YOUR_USERNAME/TalentFlow-AI/issues)
2. If not, create a new issue using the Bug Report template
3. Provide as much detail as possible:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
   - Relevant logs or screenshots

### Suggesting Features

1. Check if the feature has already been requested
2. Create a new issue using the Feature Request template
3. Clearly describe:
   - The problem you're trying to solve
   - Your proposed solution
   - Why this would be valuable

### Pull Requests

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/TalentFlow-AI.git
   cd TalentFlow-AI
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   
   # Start services
   docker-compose up -d
   
   # Run migrations
   alembic upgrade head
   ```

4. **Make your changes**
   - Write clean, readable code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

5. **Run tests and linting**
   ```bash
   # Format code
   black .
   
   # Lint code
   pylint backend/ ml/
   
   # Type checking
   mypy backend/ ml/
   
   # Run tests
   pytest
   
   # Check coverage
   pytest --cov=backend --cov=ml --cov-report=html
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```
   
   Use conventional commit messages:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `style:` - Code style changes (formatting, etc.)
   - `refactor:` - Code refactoring
   - `test:` - Adding or updating tests
   - `chore:` - Maintenance tasks

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template
   - Submit the PR

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Use property-based testing (Hypothesis) for universal properties
- Use unit tests for specific examples and edge cases
- Use integration tests for API endpoints

### Documentation

- Update README.md if adding new features
- Add docstrings to all public APIs
- Update OpenAPI documentation for new endpoints
- Add comments for complex logic

### Commit Messages

Good commit messages:
```
feat: add resume parsing for PDF files
fix: resolve authentication token expiration issue
docs: update API documentation for scoring endpoints
test: add property tests for job management
```

Bad commit messages:
```
update
fix bug
changes
wip
```

## Project Structure

```
TalentFlow-AI/
├── backend/              # FastAPI application
│   └── app/
│       ├── api/          # API endpoints
│       ├── core/         # Core utilities
│       ├── models/       # Database models
│       ├── repositories/ # Data access layer
│       ├── schemas/      # Pydantic schemas
│       └── services/     # Business logic
├── ml/                   # ML pipeline
│   ├── parsing/          # Resume parsing
│   ├── training/         # Model training
│   └── inference/        # Model inference
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   ├── property/        # Property-based tests
│   └── integration/     # Integration tests
└── infra/               # Infrastructure code
```

## Getting Help

- Check the [README](README.md) for setup instructions
- Review existing [Issues](https://github.com/YOUR_USERNAME/TalentFlow-AI/issues)
- Ask questions in issue comments
- Join discussions in Pull Requests

## Recognition

Contributors will be recognized in:
- GitHub contributors page
- Release notes for significant contributions
- Project documentation

Thank you for contributing to TalentFlow AI! 🚀
