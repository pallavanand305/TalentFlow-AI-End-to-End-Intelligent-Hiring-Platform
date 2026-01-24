# MLflow Setup Guide

This document describes the MLflow tracking server setup for TalentFlow AI.

## Overview

MLflow is configured to provide:
- **Experiment Tracking**: Log model training runs with parameters, metrics, and artifacts
- **Model Registry**: Version and manage trained models
- **Model Deployment**: Promote models through stages (None → Staging → Production)
- **Artifact Storage**: Store model files and training artifacts

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │   MLflow        │    │   Model         │
│   Scripts       │───▶│   Tracking      │───▶│   Registry      │
│                 │    │   Server        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Artifact      │
                       │   Storage       │
                       │   (Local/S3)    │
                       └─────────────────┘
```

## Configuration

### Environment Variables

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=./mlruns
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
```

### Docker Compose

The MLflow server is configured in `docker-compose.yml`:

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.10.0
  container_name: talentflow-mlflow
  ports:
    - "5000:5000"
  volumes:
    - mlflow_data:/mlflow
    - ./mlruns:/mlruns
  environment:
    - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
    - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
  command: >
    mlflow server
    --backend-store-uri sqlite:///mlflow/mlflow.db
    --default-artifact-root /mlflow/artifacts
    --host 0.0.0.0
    --port 5000
    --serve-artifacts
```

## Usage

### Starting MLflow

```bash
# Start MLflow server with Docker Compose
docker compose up mlflow

# Or start all services
docker compose up
```

### Initializing MLflow

```bash
# Initialize with default experiments
python scripts/init_mlflow.py --init

# Health check
python scripts/init_mlflow.py --health-check
```

### Using the Model Registry Service

```python
from backend.app.services.model_registry import model_registry

# Log a model
run_id = await model_registry.log_model(
    model=trained_model,
    model_name="candidate_scoring_model",
    metrics={"accuracy": 0.85, "f1_score": 0.82},
    params={"C": 1.0, "solver": "liblinear"}
)

# Register the model
version = await model_registry.register_model(
    run_id=run_id,
    model_name="candidate_scoring_model"
)

# Promote to production
await model_registry.promote_model(
    model_name="candidate_scoring_model",
    version=version,
    stage="Production"
)

# Load production model
model = await model_registry.load_model(
    model_name="candidate_scoring_model",
    stage="Production"
)
```

## Default Experiments

The initialization script creates these default experiments:

1. **resume-parsing-models**: For resume parsing and NER models
2. **scoring-models**: For candidate-job similarity scoring models  
3. **baseline-models**: For TF-IDF and simple similarity models
4. **semantic-models**: For transformer-based semantic models

## Model Lifecycle

### 1. Training Phase
```python
# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({"learning_rate": 0.01, "epochs": 100})
    
    # Train model
    model = train_model(data, params)
    
    # Log metrics
    mlflow.log_metrics({"accuracy": 0.85, "loss": 0.23})
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 2. Registration Phase
```python
# Register model from run
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="my_model"
)
```

### 3. Promotion Phase
```python
# Promote to staging
client.transition_model_version_stage(
    name="my_model",
    version="1",
    stage="Staging"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="my_model", 
    version="1",
    stage="Production"
)
```

### 4. Deployment Phase
```python
# Load production model
model = mlflow.sklearn.load_model("models:/my_model/Production")

# Use for inference
predictions = model.predict(new_data)
```

## Web UI

Access the MLflow UI at: http://localhost:5000

Features:
- **Experiments**: View all training runs with metrics and parameters
- **Models**: Browse registered models and their versions
- **Model Comparison**: Compare metrics across different model versions
- **Artifacts**: Download model files and training artifacts

## Storage Options

### Local Storage (Development)
- Backend Store: SQLite database
- Artifact Store: Local filesystem (`./mlruns`)

### Production Storage (AWS)
- Backend Store: RDS PostgreSQL
- Artifact Store: S3 bucket

```bash
# Production configuration
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@rds-endpoint/mlflow
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://my-mlflow-bucket/artifacts
```

## API Integration

The `ModelRegistry` service provides async methods for:

- `log_model()`: Log trained models with metrics and parameters
- `load_model()`: Load models by name, version, or stage
- `register_model()`: Register models in the registry
- `promote_model()`: Promote models between stages
- `compare_models()`: Compare metrics across versions
- `list_models()`: List all registered models
- `health_check()`: Check MLflow server connectivity

## Monitoring

### Health Checks
```python
# Check MLflow server health
health = await model_registry.health_check()
print(health["status"])  # "healthy" or "unhealthy"
```

### Model Metrics
- Track model performance over time
- Monitor data drift
- Compare model versions
- Set up alerts for model degradation

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if MLflow server is running
   docker compose ps mlflow
   
   # Check logs
   docker compose logs mlflow
   ```

2. **Permission Errors**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER ./mlruns
   ```

3. **Database Locked**
   ```bash
   # Stop all services and restart
   docker compose down
   docker compose up
   ```

### Logs

```bash
# View MLflow server logs
docker compose logs -f mlflow

# View application logs
docker compose logs -f backend
```

## Security

### Authentication
- MLflow server runs without authentication in development
- For production, configure authentication with:
  - Basic auth
  - OAuth integration
  - Reverse proxy with auth

### Network Security
- MLflow server exposed only on localhost in development
- Use VPC and security groups in production
- Enable HTTPS for production deployments

## Backup and Recovery

### Database Backup
```bash
# Backup SQLite database
cp mlflow_data/mlflow.db mlflow_backup_$(date +%Y%m%d).db
```

### Artifact Backup
```bash
# Backup artifacts
tar -czf mlruns_backup_$(date +%Y%m%d).tar.gz mlruns/
```

### Restore
```bash
# Restore database
cp mlflow_backup_20240115.db mlflow_data/mlflow.db

# Restore artifacts  
tar -xzf mlruns_backup_20240115.tar.gz
```

## Performance Tuning

### Database Optimization
- Use PostgreSQL for production
- Configure connection pooling
- Regular database maintenance

### Artifact Storage
- Use S3 for production artifact storage
- Configure lifecycle policies for old artifacts
- Enable compression for large models

### Caching
- Enable model caching for frequently accessed models
- Use Redis for caching model metadata
- Implement artifact caching strategies