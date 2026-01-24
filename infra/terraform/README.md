# TalentFlow AI - Infrastructure Deployment Guide

This directory contains the Terraform configuration for deploying TalentFlow AI infrastructure on AWS.

## Prerequisites

Before deploying the infrastructure, ensure you have the following tools installed:

1. **AWS CLI** (v2.0 or later)
2. **Terraform** (v1.6.0 or later)
3. **Docker** (for building and pushing images)
4. **Git** (for version control)

### Installation Commands

```bash
# Install AWS CLI (Linux/macOS)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install Terraform (Linux/macOS)
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Install Docker (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo usermod -aG docker $USER
```

## AWS Setup

### 1. Configure AWS Credentials

```bash
# Configure AWS CLI with your credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 2. Create Required Secrets

Set the following environment variables with secure values:

```bash
# Database password (minimum 8 characters)
export DB_PASSWORD="your-secure-db-password"

# JWT secret key (minimum 32 characters)
export SECRET_KEY="your-jwt-secret-key-32-chars-minimum"

# Application AWS credentials (separate from deployment credentials)
export APP_AWS_ACCESS_KEY_ID="your-app-access-key"
export APP_AWS_SECRET_ACCESS_KEY="your-app-secret-key"

# Redis authentication token (optional but recommended)
export REDIS_AUTH_TOKEN="your-redis-auth-token"

# SSL Certificate ARN (optional, for HTTPS)
export SSL_CERTIFICATE_ARN="arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"

# Domain name (optional)
export DOMAIN_NAME="api.talentflow.com"
```

## Deployment Steps

### Step 1: Initialize Terraform Backend

First, we need to create the S3 bucket and DynamoDB table for Terraform state management:

```bash
cd infra/terraform

# Initialize Terraform (first time only)
terraform init

# Create the backend resources
terraform apply -target=aws_s3_bucket.terraform_state -target=aws_dynamodb_table.terraform_locks

# Update the backend configuration in main.tf with the actual bucket name
# Then re-initialize with the backend
terraform init -migrate-state
```

### Step 2: Plan and Apply Infrastructure

```bash
# Select environment (prod or staging)
export ENVIRONMENT="prod"  # or "staging"

# Create workspace
terraform workspace new $ENVIRONMENT || terraform workspace select $ENVIRONMENT

# Plan the deployment
terraform plan \
  -var-file="environments/${ENVIRONMENT}.tfvars" \
  -var="db_password=$DB_PASSWORD" \
  -var="secret_key=$SECRET_KEY" \
  -var="aws_access_key_id=$APP_AWS_ACCESS_KEY_ID" \
  -var="aws_secret_access_key=$APP_AWS_SECRET_ACCESS_KEY" \
  -var="redis_auth_token=$REDIS_AUTH_TOKEN" \
  -var="certificate_arn=$SSL_CERTIFICATE_ARN" \
  -var="domain_name=$DOMAIN_NAME" \
  -out=tfplan

# Apply the infrastructure
terraform apply tfplan

# Save outputs for later use
terraform output -json > terraform-outputs.json
```

### Step 3: Build and Push Docker Images

```bash
cd ../../  # Back to project root

# Get ECR login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com

# Get repository URLs from Terraform outputs
BACKEND_REPO=$(jq -r '.ecr_backend_url.value' infra/terraform/terraform-outputs.json)
ML_REPO=$(jq -r '.ecr_ml_url.value' infra/terraform/terraform-outputs.json)

# Build and push backend image
docker build -f docker/Dockerfile.backend -t $BACKEND_REPO:latest -t $BACKEND_REPO:$(git rev-parse --short HEAD) .
docker push $BACKEND_REPO:latest
docker push $BACKEND_REPO:$(git rev-parse --short HEAD)

# Build and push ML image
docker build -f docker/Dockerfile.ml -t $ML_REPO:latest -t $ML_REPO:$(git rev-parse --short HEAD) .
docker push $ML_REPO:latest
docker push $ML_REPO:$(git rev-parse --short HEAD)
```

### Step 4: Deploy ECS Services

The ECS services are automatically created by Terraform, but you may need to trigger a deployment to pull the latest images:

```bash
# Get cluster name
CLUSTER_NAME=$(jq -r '.ecs_cluster_name.value' infra/terraform/terraform-outputs.json)

# Update services to pull latest images
aws ecs update-service --cluster $CLUSTER_NAME --service talentflow-ai-prod-backend-service --force-new-deployment
aws ecs update-service --cluster $CLUSTER_NAME --service talentflow-ai-prod-worker-service --force-new-deployment
aws ecs update-service --cluster $CLUSTER_NAME --service talentflow-ai-prod-mlflow-service --force-new-deployment

# Wait for services to stabilize
aws ecs wait services-stable --cluster $CLUSTER_NAME --services talentflow-ai-prod-backend-service talentflow-ai-prod-worker-service talentflow-ai-prod-mlflow-service
```

### Step 5: Run Database Migrations

```bash
# Get database endpoint
DB_ENDPOINT=$(jq -r '.rds_endpoint.value' infra/terraform/terraform-outputs.json)

# Set database URL
export DATABASE_URL="postgresql+asyncpg://postgres:$DB_PASSWORD@$DB_ENDPOINT:5432/talentflow"

# Run migrations
alembic upgrade head
```

### Step 6: Verify Deployment

```bash
# Get application URL
APP_URL=$(jq -r '.application_url.value' infra/terraform/terraform-outputs.json)

# Test health endpoint
curl -f "$APP_URL/health"

# Test API documentation
curl -f "$APP_URL/docs"

# Get MLflow URL
MLFLOW_URL=$(jq -r '.mlflow_url.value' infra/terraform/terraform-outputs.json)
echo "MLflow UI: $MLFLOW_URL"
```

## Environment-Specific Configurations

### Production Environment

- Uses `prod.tfvars` configuration
- Higher instance classes and storage
- Deletion protection enabled
- Multiple AZs for high availability
- Auto-scaling enabled

### Staging Environment

- Uses `staging.tfvars` configuration
- Smaller instance classes for cost optimization
- Deletion protection disabled
- Single AZ deployment
- Minimal auto-scaling

## Monitoring and Maintenance

### CloudWatch Logs

All services log to CloudWatch:
- Backend: `/aws/ecs/talentflow-ai-prod/backend`
- Worker: `/aws/ecs/talentflow-ai-prod/worker`
- MLflow: `/aws/ecs/talentflow-ai-prod/mlflow`

### Scaling

Auto-scaling is configured for:
- Backend service: CPU and memory-based scaling
- Worker service: CPU-based scaling
- Database: Storage auto-scaling enabled

### Backups

- RDS automated backups (7-14 days retention)
- S3 versioning enabled for all buckets
- Lifecycle policies for cost optimization

## Troubleshooting

### Common Issues

1. **ECS Service Failed to Start**
   - Check CloudWatch logs for container errors
   - Verify environment variables and secrets
   - Ensure security groups allow required traffic

2. **Database Connection Issues**
   - Verify security group rules
   - Check database endpoint and credentials
   - Ensure VPC configuration is correct

3. **S3 Access Issues**
   - Verify IAM roles and policies
   - Check bucket policies
   - Ensure correct AWS credentials

### Useful Commands

```bash
# Check ECS service status
aws ecs describe-services --cluster $CLUSTER_NAME --services talentflow-ai-prod-backend-service

# View CloudWatch logs
aws logs tail /aws/ecs/talentflow-ai-prod/backend --follow

# Check RDS status
aws rds describe-db-instances --db-instance-identifier talentflow-ai-prod-db

# List S3 buckets
aws s3 ls | grep talentflow-ai
```

## Cleanup

To destroy the infrastructure:

```bash
cd infra/terraform

# Destroy infrastructure (be careful!)
terraform destroy \
  -var-file="environments/${ENVIRONMENT}.tfvars" \
  -var="db_password=$DB_PASSWORD" \
  -var="secret_key=$SECRET_KEY" \
  -var="aws_access_key_id=$APP_AWS_ACCESS_KEY_ID" \
  -var="aws_secret_access_key=$APP_AWS_SECRET_ACCESS_KEY" \
  -var="redis_auth_token=$REDIS_AUTH_TOKEN" \
  -var="certificate_arn=$SSL_CERTIFICATE_ARN" \
  -var="domain_name=$DOMAIN_NAME"
```

## Security Considerations

1. **Secrets Management**: All sensitive data is stored in AWS Secrets Manager
2. **Network Security**: Private subnets for databases and application services
3. **Encryption**: Encryption at rest for RDS, S3, and ElastiCache
4. **Access Control**: IAM roles with least privilege principles
5. **Monitoring**: CloudWatch logging and monitoring enabled

## Cost Optimization

1. **Instance Sizing**: Right-sized instances for each environment
2. **Storage Lifecycle**: S3 lifecycle policies for cost-effective storage
3. **Auto Scaling**: Automatic scaling based on demand
4. **Spot Instances**: Consider using Fargate Spot for non-critical workloads

For more information, see the [deployment guide](../../docs/deployment-guide.md).