#!/bin/bash
# Database setup script

set -e

echo "🚀 Setting up TalentFlow AI database..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
fi

# Start Docker services
echo "🐳 Starting Docker services (PostgreSQL, Redis, MLflow)..."
docker-compose up -d postgres redis mlflow

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 5

# Run migrations
echo "📊 Running database migrations..."
alembic upgrade head

echo "✅ Database setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start the backend: uvicorn backend.app.main:app --reload"
echo "  2. Visit API docs: http://localhost:8000/docs"
echo "  3. Visit MLflow: http://localhost:5000"
