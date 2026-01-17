#!/bin/bash
# Test runner script

set -e

echo "🧪 Running TalentFlow AI tests..."

# Create test database if it doesn't exist
echo "📊 Setting up test database..."
docker-compose exec -T postgres psql -U postgres -c "CREATE DATABASE talentflow_test;" 2>/dev/null || true

# Run migrations on test database
echo "🔄 Running test database migrations..."
DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/talentflow_test" alembic upgrade head

# Run tests
echo "🏃 Running tests..."
pytest tests/ -v --cov=backend --cov=ml --cov-report=html --cov-report=term-missing

echo "✅ Tests complete!"
echo "📊 Coverage report: htmlcov/index.html"
