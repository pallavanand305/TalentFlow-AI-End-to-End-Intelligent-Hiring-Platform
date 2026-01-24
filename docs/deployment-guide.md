# TalentFlow AI - Deployment Guide

This comprehensive guide covers the complete deployment process for TalentFlow AI, including AWS infrastructure setup, CI/CD pipeline configuration, and production deployment procedures.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [AWS Infrastructure Setup](#aws-infrastructure-setup)
4. [CI/CD Pipeline Configuration](#cicd-pipeline-configuration)
5. [Production Deployment Process](#production-deployment-process)
6. [Environment Configuration](#environment-configuration)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Security Configuration](#security-configuration)
9. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
10. [Troubleshooting](#troubleshooting)

## Overview

TalentFlow AI is deployed using a cloud-native architecture on AWS with the following components:

- **Compute**: ECS Fargate for containerized services
- **Database**: RDS PostgreSQL for relational data
- **Cache**: ElastiCache Redis for session storage and task queues
- **Storage**: S3 for resume files and ML model artifacts
- **Container Registry**: ECR for Docker images
- **Load Balancing**: Application Load Balancer (ALB)
- **Monitoring**: CloudWatch for logs and metrics
- **Secrets**: AWS Secrets Manager for sensitive configuration
- **Infrastructure**: Terraform for Infrastructure as Code (IaC)

### Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Internet      │    │   CloudFront    │    │   Route 53      │
│   Gateway       │───▶│   (CDN)         │───▶│   (DNS)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Application   │
                       │   Load Balancer │
                       └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ECS Fargate   │    │   ECS Fargate   │    │   ECS Fargate   │
│   (Backend API) │    │   (ML Workers)  │    │   (MLflow)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RDS           │    │   ElastiCache   │    │   S3 Buckets    │
│   PostgreSQL    │    │   Redis         │    │   (Files/Models)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### Required Tools

1. **AWS CLI** (v2.0+)
   ```bash
   # Install AWS CLI
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   
   # Configure credentials
   aws configure
   ```

2. **Terraform** (v1.0+)
   ```bash
   # Install Terraform
   wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
   unzip terraform_1.6.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/
   ```

3. **Docker** (v20.0+)
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   ```

4. **kubectl** (for EKS if using Kubernetes)
   ```bash
   # Install kubectl
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
   ```

### AWS Account Setup

1. **Create AWS Account** with appropriate permissions
2. **Configure IAM User** with programmatic access
3. **Required IAM Permissions**:
   - EC2FullAccess
   - ECSFullAccess
   - RDSFullAccess
   - S3FullAccess
   - ElastiCacheFullAccess
   - CloudWatchFullAccess
   - SecretsManagerFullAccess
   - IAMFullAccess (for role creation)

### Domain and SSL

1. **Register Domain** (optional, can use ALB DNS)
2. **Request SSL Certificate** in AWS Certificate Manager
3. **Configure Route 53** hosted zone (if using custom domain)

## AWS Infrastructure Setup

### 1. Terraform Configuration

Create the Terraform infrastructure configuration:

```bash
# Create terraform directory structure
mkdir -p infra/terraform/{modules,environments}
cd infra/terraform
```

#### Main Terraform Configuration

Create `infra/terraform/main.tf`:

```hcl
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "talentflow-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "TalentFlow-AI"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  azs         = slice(data.aws_availability_zones.available.names, 0, 2)
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  name_prefix = local.name_prefix
  cidr_block  = var.vpc_cidr
  azs         = local.azs
  
  tags = local.common_tags
}

# Security Groups Module
module "security_groups" {
  source = "./modules/security_groups"
  
  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  
  tags = local.common_tags
}

# RDS Module
module "rds" {
  source = "./modules/rds"
  
  name_prefix           = local.name_prefix
  vpc_id               = module.vpc.vpc_id
  private_subnet_ids   = module.vpc.private_subnet_ids
  security_group_ids   = [module.security_groups.rds_security_group_id]
  
  db_name     = var.db_name
  db_username = var.db_username
  db_password = var.db_password
  
  tags = local.common_tags
}

# ElastiCache Module
module "elasticache" {
  source = "./modules/elasticache"
  
  name_prefix           = local.name_prefix
  vpc_id               = module.vpc.vpc_id
  private_subnet_ids   = module.vpc.private_subnet_ids
  security_group_ids   = [module.security_groups.redis_security_group_id]
  
  tags = local.common_tags
}

# S3 Module
module "s3" {
  source = "./modules/s3"
  
  name_prefix = local.name_prefix
  
  tags = local.common_tags
}

# ECR Module
module "ecr" {
  source = "./modules/ecr"
  
  name_prefix = local.name_prefix
  
  tags = local.common_tags
}

# ECS Module
module "ecs" {
  source = "./modules/ecs"
  
  name_prefix            = local.name_prefix
  vpc_id                = module.vpc.vpc_id
  public_subnet_ids     = module.vpc.public_subnet_ids
  private_subnet_ids    = module.vpc.private_subnet_ids
  
  # Security Groups
  alb_security_group_id     = module.security_groups.alb_security_group_id
  ecs_security_group_id     = module.security_groups.ecs_security_group_id
  
  # ECR Repositories
  backend_ecr_url = module.ecr.backend_repository_url
  ml_ecr_url      = module.ecr.ml_repository_url
  
  # Database
  db_host     = module.rds.db_endpoint
  db_name     = var.db_name
  db_username = var.db_username
  db_password = var.db_password
  
  # Redis
  redis_host = module.elasticache.redis_endpoint
  
  # S3
  s3_bucket_resumes = module.s3.resumes_bucket_name
  s3_bucket_models  = module.s3.models_bucket_name
  
  tags = local.common_tags
}

# CloudWatch Module
module "cloudwatch" {
  source = "./modules/cloudwatch"
  
  name_prefix = local.name_prefix
  
  tags = local.common_tags
}

# Secrets Manager
resource "aws_secretsmanager_secret" "app_secrets" {
  name = "${local.name_prefix}-app-secrets"
  
  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    SECRET_KEY    = var.secret_key
    DB_PASSWORD   = var.db_password
    AWS_ACCESS_KEY_ID     = var.aws_access_key_id
    AWS_SECRET_ACCESS_KEY = var.aws_secret_access_key
  })
}
```

#### Variables Configuration

Create `infra/terraform/variables.tf`:

```hcl
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "talentflow-ai"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "talentflow"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "postgres"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "secret_key" {
  description = "Application secret key"
  type        = string
  sensitive   = true
}

variable "aws_access_key_id" {
  description = "AWS access key ID for application"
  type        = string
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "AWS secret access key for application"
  type        = string
  sensitive   = true
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "SSL certificate ARN"
  type        = string
  default     = ""
}
```

#### Outputs Configuration

Create `infra/terraform/outputs.tf`:

```hcl
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "alb_dns_name" {
  description = "Application Load Balancer DNS name"
  value       = module.ecs.alb_dns_name
}

output "alb_zone_id" {
  description = "Application Load Balancer zone ID"
  value       = module.ecs.alb_zone_id
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.db_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.elasticache.redis_endpoint
  sensitive   = true
}

output "s3_bucket_resumes" {
  description = "S3 bucket for resumes"
  value       = module.s3.resumes_bucket_name
}

output "s3_bucket_models" {
  description = "S3 bucket for models"
  value       = module.s3.models_bucket_name
}

output "ecr_backend_url" {
  description = "ECR repository URL for backend"
  value       = module.ecr.backend_repository_url
}

output "ecr_ml_url" {
  description = "ECR repository URL for ML"
  value       = module.ecr.ml_repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = module.cloudwatch.log_group_name
}
```

### 2. Terraform Modules

#### VPC Module

Create `infra/terraform/modules/vpc/main.tf`:

```hcl
# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(var.tags, {
    Name = "${var.name_prefix}-vpc"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = merge(var.tags, {
    Name = "${var.name_prefix}-igw"
  })
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.azs)
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.cidr_block, 8, count.index)
  availability_zone       = var.azs[count.index]
  map_public_ip_on_launch = true
  
  tags = merge(var.tags, {
    Name = "${var.name_prefix}-public-${var.azs[count.index]}"
    Type = "Public"
  })
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(var.azs)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.cidr_block, 8, count.index + 10)
  availability_zone = var.azs[count.index]
  
  tags = merge(var.tags, {
    Name = "${var.name_prefix}-private-${var.azs[count.index]}"
    Type = "Private"
  })
}

# NAT Gateways
resource "aws_eip" "nat" {
  count = length(var.azs)
  
  domain = "vpc"
  
  tags = merge(var.tags, {
    Name = "${var.name_prefix}-nat-eip-${var.azs[count.index]}"
  })
  
  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  count = length(var.azs)
  
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = merge(var.tags, {
    Name = "${var.name_prefix}-nat-${var.azs[count.index]}"
  })
  
  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = merge(var.tags, {
    Name = "${var.name_prefix}-public-rt"
  })
}

resource "aws_route_table" "private" {
  count = length(var.azs)
  
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }
  
  tags = merge(var.tags, {
    Name = "${var.name_prefix}-private-rt-${var.azs[count.index]}"
  })
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(var.azs)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(var.azs)
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}
```

Create `infra/terraform/modules/vpc/variables.tf`:

```hcl
variable "name_prefix" {
  description = "Name prefix for resources"
  type        = string
}

variable "cidr_block" {
  description = "CIDR block for VPC"
  type        = string
}

variable "azs" {
  description = "Availability zones"
  type        = list(string)
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
```

Create `infra/terraform/modules/vpc/outputs.tf`:

```hcl
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "internet_gateway_id" {
  description = "Internet Gateway ID"
  value       = aws_internet_gateway.main.id
}

output "nat_gateway_ids" {
  description = "NAT Gateway IDs"
  value       = aws_nat_gateway.main[*].id
}
```

### 3. Infrastructure Deployment

#### Initialize Terraform

```bash
cd infra/terraform

# Initialize Terraform
terraform init

# Create workspace for environment
terraform workspace new prod

# Plan deployment
terraform plan -var-file="environments/prod.tfvars"

# Apply infrastructure
terraform apply -var-file="environments/prod.tfvars"
```

#### Environment Variables File

Create `infra/terraform/environments/prod.tfvars`:

```hcl
aws_region   = "us-east-1"
environment  = "prod"
project_name = "talentflow-ai"

vpc_cidr = "10.0.0.0/16"

db_name     = "talentflow"
db_username = "postgres"
db_password = "your-secure-password-here"

secret_key = "your-jwt-secret-key-here"

aws_access_key_id     = "your-aws-access-key"
aws_secret_access_key = "your-aws-secret-key"

domain_name     = "api.talentflow.com"
certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"
```

## CI/CD Pipeline Configuration

### 1. GitHub Actions Workflows

#### Backend CI/CD Pipeline

Create `.github/workflows/backend-ci-cd.yml`:

```yaml
name: Backend CI/CD

on:
  push:
    branches: [main, develop]
    paths:
      - 'backend/**'
      - 'ml/**'
      - 'requirements.txt'
      - 'docker/Dockerfile.backend'
      - '.github/workflows/backend-ci-cd.yml'
  pull_request:
    branches: [main]
    paths:
      - 'backend/**'
      - 'ml/**'
      - 'requirements.txt'
      - 'docker/Dockerfile.backend'

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY_BACKEND: talentflow-ai-backend
  ECR_REPOSITORY_ML: talentflow-ai-ml
  ECS_SERVICE_BACKEND: talentflow-ai-backend-service
  ECS_SERVICE_WORKER: talentflow-ai-worker-service
  ECS_CLUSTER: talentflow-ai-cluster

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: talentflow_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
    
    - name: Run linting
      run: |
        black --check .
        pylint backend/ ml/
        mypy backend/ ml/
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/talentflow_test
        REDIS_URL: redis://localhost:6379/0
        SECRET_KEY: test-secret-key
        MLFLOW_TRACKING_URI: http://localhost:5000
      run: |
        pytest tests/ --cov=backend --cov=ml --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v2
    
    - name: Build, tag, and push backend image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -f docker/Dockerfile.backend -t $ECR_REGISTRY/$ECR_REPOSITORY_BACKEND:$IMAGE_TAG .
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY_BACKEND:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY_BACKEND:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY_BACKEND:$IMAGE_TAG
        docker push $ECR_REGISTRY/$ECR_REPOSITORY_BACKEND:latest
    
    - name: Build, tag, and push ML image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -f docker/Dockerfile.ml -t $ECR_REGISTRY/$ECR_REPOSITORY_ML:$IMAGE_TAG .
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY_ML:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY_ML:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY_ML:$IMAGE_TAG
        docker push $ECR_REGISTRY/$ECR_REPOSITORY_ML:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Update ECS service - Backend
      run: |
        aws ecs update-service \
          --cluster $ECS_CLUSTER \
          --service $ECS_SERVICE_BACKEND \
          --force-new-deployment
    
    - name: Update ECS service - Worker
      run: |
        aws ecs update-service \
          --cluster $ECS_CLUSTER \
          --service $ECS_SERVICE_WORKER \
          --force-new-deployment
    
    - name: Wait for deployment to complete
      run: |
        aws ecs wait services-stable \
          --cluster $ECS_CLUSTER \
          --services $ECS_SERVICE_BACKEND $ECS_SERVICE_WORKER
    
    - name: Notify deployment success
      if: success()
      run: |
        echo "Deployment successful!"
        # Add Slack/email notification here if needed
    
    - name: Notify deployment failure
      if: failure()
      run: |
        echo "Deployment failed!"
        # Add Slack/email notification here if needed
```

#### Infrastructure CI/CD Pipeline

Create `.github/workflows/infrastructure.yml`:

```yaml
name: Infrastructure CI/CD

on:
  push:
    branches: [main]
    paths:
      - 'infra/terraform/**'
      - '.github/workflows/infrastructure.yml'
  pull_request:
    branches: [main]
    paths:
      - 'infra/terraform/**'

env:
  AWS_REGION: us-east-1
  TF_VERSION: 1.6.0

jobs:
  terraform-check:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v3
      with:
        terraform_version: ${{ env.TF_VERSION }}
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Terraform Format Check
      working-directory: infra/terraform
      run: terraform fmt -check -recursive
    
    - name: Terraform Init
      working-directory: infra/terraform
      run: terraform init
    
    - name: Terraform Validate
      working-directory: infra/terraform
      run: terraform validate
    
    - name: Terraform Plan
      working-directory: infra/terraform
      run: |
        terraform plan \
          -var="db_password=${{ secrets.DB_PASSWORD }}" \
          -var="secret_key=${{ secrets.SECRET_KEY }}" \
          -var="aws_access_key_id=${{ secrets.APP_AWS_ACCESS_KEY_ID }}" \
          -var="aws_secret_access_key=${{ secrets.APP_AWS_SECRET_ACCESS_KEY }}" \
          -out=tfplan
    
    - name: Upload Terraform Plan
      uses: actions/upload-artifact@v3
      with:
        name: terraform-plan
        path: infra/terraform/tfplan

  terraform-apply:
    needs: terraform-check
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v3
      with:
        terraform_version: ${{ env.TF_VERSION }}
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Download Terraform Plan
      uses: actions/download-artifact@v3
      with:
        name: terraform-plan
        path: infra/terraform/
    
    - name: Terraform Init
      working-directory: infra/terraform
      run: terraform init
    
    - name: Terraform Apply
      working-directory: infra/terraform
      run: terraform apply -auto-approve tfplan
```

### 2. GitHub Secrets Configuration

Configure the following secrets in your GitHub repository:

```bash
# AWS Credentials for CI/CD
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# Application AWS Credentials (for the app to use)
APP_AWS_ACCESS_KEY_ID=AKIA...
APP_AWS_SECRET_ACCESS_KEY=...

# Database Password
DB_PASSWORD=your-secure-database-password

# JWT Secret Key
SECRET_KEY=your-jwt-secret-key

# Optional: Notification webhooks
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## Production Deployment Process

### 1. Pre-Deployment Checklist

Before deploying to production, ensure:

- [ ] All tests pass in CI/CD pipeline
- [ ] Code review completed and approved
- [ ] Security scan completed
- [ ] Database migrations tested
- [ ] Environment variables configured
- [ ] SSL certificates valid
- [ ] Monitoring and alerting configured
- [ ] Backup strategy in place
- [ ] Rollback plan prepared

### 2. Deployment Steps

#### Step 1: Infrastructure Deployment

```bash
# 1. Deploy infrastructure
cd infra/terraform
terraform workspace select prod
terraform plan -var-file="environments/prod.tfvars"
terraform apply -var-file="environments/prod.tfvars"

# 2. Note the outputs
terraform output
```

#### Step 2: Database Setup

```bash
# 1. Connect to RDS instance
export DB_HOST=$(terraform output -raw rds_endpoint)
export DB_NAME="talentflow"
export DB_USER="postgres"
export DB_PASSWORD="your-password"

# 2. Run migrations
alembic upgrade head

# 3. Create initial admin user (optional)
python scripts/create_admin_user.py
```

#### Step 3: Build and Push Docker Images

```bash
# 1. Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# 2. Build and push backend image
docker build -f docker/Dockerfile.backend -t talentflow-backend .
docker tag talentflow-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/talentflow-ai-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/talentflow-ai-backend:latest

# 3. Build and push ML image
docker build -f docker/Dockerfile.ml -t talentflow-ml .
docker tag talentflow-ml:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/talentflow-ai-ml:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/talentflow-ai-ml:latest
```

#### Step 4: Deploy ECS Services

```bash
# 1. Update ECS services
aws ecs update-service \
  --cluster talentflow-ai-cluster \
  --service talentflow-ai-backend-service \
  --force-new-deployment

aws ecs update-service \
  --cluster talentflow-ai-cluster \
  --service talentflow-ai-worker-service \
  --force-new-deployment

# 2. Wait for deployment to complete
aws ecs wait services-stable \
  --cluster talentflow-ai-cluster \
  --services talentflow-ai-backend-service talentflow-ai-worker-service
```

#### Step 5: Verify Deployment

```bash
# 1. Check service health
curl https://api.talentflow.com/health

# 2. Check API documentation
curl https://api.talentflow.com/docs

# 3. Run smoke tests
python scripts/smoke_tests.py --env prod
```

### 3. Post-Deployment Tasks

#### Health Checks

```bash
# 1. Verify all services are running
aws ecs describe-services \
  --cluster talentflow-ai-cluster \
  --services talentflow-ai-backend-service talentflow-ai-worker-service

# 2. Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix "/ecs/talentflow"

# 3. Monitor metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=talentflow-ai-backend-service \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T01:00:00Z \
  --period 300 \
  --statistics Average
```

#### Performance Testing

```bash
# 1. Load testing with Apache Bench
ab -n 1000 -c 10 https://api.talentflow.com/health

# 2. API endpoint testing
python scripts/load_test.py --url https://api.talentflow.com --concurrent 10 --requests 1000
```

## Environment Configuration

### 1. Production Environment Variables

Create production environment configuration:

```bash
# Application
APP_NAME=TalentFlow AI
APP_VERSION=1.0.0
DEBUG=False
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql+asyncpg://postgres:${DB_PASSWORD}@${DB_HOST}:5432/talentflow

# Redis
REDIS_URL=redis://${REDIS_HOST}:6379/0

# JWT Authentication
SECRET_KEY=${SECRET_KEY}
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# AWS
AWS_REGION=us-east-1
S3_BUCKET_RESUMES=talentflow-prod-resumes
S3_BUCKET_MODELS=talentflow-prod-models
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

# MLflow
MLFLOW_TRACKING_URI=http://mlflow.internal:5000
MLFLOW_ARTIFACT_ROOT=s3://talentflow-prod-models/mlflow-artifacts
MLFLOW_BACKEND_STORE_URI=postgresql://postgres:${DB_PASSWORD}@${DB_HOST}:5432/mlflow

# File Upload
MAX_UPLOAD_SIZE=10485760
ALLOWED_EXTENSIONS=.pdf,.docx

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100

# CORS
CORS_ORIGINS=https://app.talentflow.com,https://admin.talentflow.com

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Monitoring
SENTRY_DSN=${SENTRY_DSN}
NEW_RELIC_LICENSE_KEY=${NEW_RELIC_LICENSE_KEY}
```

### 2. Secrets Management

Use AWS Secrets Manager for sensitive configuration:

```bash
# Create secrets
aws secretsmanager create-secret \
  --name "talentflow-prod-secrets" \
  --description "TalentFlow AI production secrets" \
  --secret-string '{
    "SECRET_KEY": "your-jwt-secret-key",
    "DB_PASSWORD": "your-database-password",
    "AWS_ACCESS_KEY_ID": "your-aws-access-key",
    "AWS_SECRET_ACCESS_KEY": "your-aws-secret-key",
    "SENTRY_DSN": "your-sentry-dsn"
  }'

# Retrieve secrets in application
aws secretsmanager get-secret-value \
  --secret-id "talentflow-prod-secrets" \
  --query SecretString \
  --output text
```

## Monitoring and Logging

### 1. CloudWatch Configuration

#### Log Groups

```bash
# Create log groups
aws logs create-log-group --log-group-name "/ecs/talentflow-backend"
aws logs create-log-group --log-group-name "/ecs/talentflow-worker"
aws logs create-log-group --log-group-name "/ecs/talentflow-mlflow"

# Set retention policy
aws logs put-retention-policy \
  --log-group-name "/ecs/talentflow-backend" \
  --retention-in-days 30
```

#### Custom Metrics

```python
# In your application code
import boto3

cloudwatch = boto3.client('cloudwatch')

# Custom metric for API requests
cloudwatch.put_metric_data(
    Namespace='TalentFlow/API',
    MetricData=[
        {
            'MetricName': 'RequestCount',
            'Value': 1,
            'Unit': 'Count',
            'Dimensions': [
                {
                    'Name': 'Endpoint',
                    'Value': '/api/v1/resumes/upload'
                }
            ]
        }
    ]
)
```

#### Alarms

```bash
# Create CPU utilization alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "TalentFlow-High-CPU" \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:talentflow-alerts

# Create error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "TalentFlow-High-Error-Rate" \
  --alarm-description "Alert when error rate exceeds 5%" \
  --metric-name ErrorRate \
  --namespace TalentFlow/API \
  --statistic Average \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:talentflow-alerts
```

### 2. Application Monitoring

#### Health Check Endpoint

```python
# backend/app/api/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.core.database import get_db
from backend.app.services.model_registry import model_registry
import redis
import boto3

router = APIRouter()

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Database check
    try:
        await db.execute("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Redis check
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        r.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # MLflow check
    try:
        await model_registry.health_check()
        health_status["checks"]["mlflow"] = "healthy"
    except Exception as e:
        health_status["checks"]["mlflow"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # S3 check
    try:
        s3 = boto3.client('s3')
        s3.head_bucket(Bucket=settings.S3_BUCKET_RESUMES)
        health_status["checks"]["s3"] = "healthy"
    except Exception as e:
        health_status["checks"]["s3"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return health_status
```

#### Structured Logging

```python
# backend/app/core/logging.py
import logging
import json
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Set JSON formatter
for handler in logging.root.handlers:
    handler.setFormatter(JSONFormatter())
```

### 3. Performance Monitoring

#### APM Integration (New Relic)

```python
# requirements.txt
newrelic

# In your application startup
import newrelic.agent

newrelic.agent.initialize('newrelic.ini')

# Decorator for custom metrics
@newrelic.agent.function_trace()
async def score_candidate(candidate_id: str, job_id: str):
    # Your scoring logic here
    pass
```

#### Custom Metrics Dashboard

Create CloudWatch dashboard:

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization", "ServiceName", "talentflow-ai-backend-service"],
          [".", "MemoryUtilization", ".", "."]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "ECS Service Metrics"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["TalentFlow/API", "RequestCount", "Endpoint", "/api/v1/resumes/upload"],
          [".", ".", ".", "/api/v1/scores/compute"],
          [".", "ResponseTime", ".", "/api/v1/resumes/upload"]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "us-east-1",
        "title": "API Metrics"
      }
    }
  ]
}
```

## Security Configuration

### 1. Network Security

#### Security Groups

```hcl
# ALB Security Group
resource "aws_security_group" "alb" {
  name_prefix = "${var.name_prefix}-alb-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

# ECS Security Group
resource "aws_security_group" "ecs" {
  name_prefix = "${var.name_prefix}-ecs-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name_prefix = "${var.name_prefix}-rds-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  tags = var.tags
}
```

#### WAF Configuration

```hcl
resource "aws_wafv2_web_acl" "main" {
  name  = "${var.name_prefix}-waf"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "RateLimitRule"
    priority = 1

    override_action {
      none {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }

    action {
      block {}
    }
  }

  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  tags = var.tags
}

resource "aws_wafv2_web_acl_association" "main" {
  resource_arn = aws_lb.main.arn
  web_acl_arn  = aws_wafv2_web_acl.main.arn
}
```

### 2. Data Encryption

#### S3 Bucket Encryption

```hcl
resource "aws_s3_bucket_server_side_encryption_configuration" "resumes" {
  bucket = aws_s3_bucket.resumes.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "resumes" {
  bucket = aws_s3_bucket.resumes.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

#### RDS Encryption

```hcl
resource "aws_db_instance" "main" {
  # ... other configuration ...
  
  storage_encrypted = true
  kms_key_id       = aws_kms_key.rds.arn
}

resource "aws_kms_key" "rds" {
  description = "KMS key for RDS encryption"
  
  tags = var.tags
}
```

### 3. IAM Roles and Policies

#### ECS Task Role

```hcl
resource "aws_iam_role" "ecs_task_role" {
  name = "${var.name_prefix}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${var.name_prefix}-ecs-task-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.resumes.arn}/*",
          "${aws_s3_bucket.models.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.app_secrets.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}
```

## Backup and Disaster Recovery

### 1. Database Backup

#### Automated RDS Backups

```hcl
resource "aws_db_instance" "main" {
  # ... other configuration ...
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  final_snapshot_identifier = "${var.name_prefix}-final-snapshot"
  skip_final_snapshot      = false
  
  tags = var.tags
}
```

#### Manual Backup Script

```bash
#!/bin/bash
# scripts/backup_database.sh

DB_HOST="your-rds-endpoint"
DB_NAME="talentflow"
DB_USER="postgres"
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_DIR/talentflow_backup_$DATE.sql

# Upload to S3
aws s3 cp $BACKUP_DIR/talentflow_backup_$DATE.sql s3://talentflow-backups/database/

# Clean up local backup
rm $BACKUP_DIR/talentflow_backup_$DATE.sql

# Keep only last 30 days of backups in S3
aws s3 ls s3://talentflow-backups/database/ | while read -r line; do
  createDate=$(echo $line | awk '{print $1" "$2}')
  createDate=$(date -d "$createDate" +%s)
  olderThan=$(date -d "30 days ago" +%s)
  if [[ $createDate -lt $olderThan ]]; then
    fileName=$(echo $line | awk '{print $4}')
    if [[ $fileName != "" ]]; then
      aws s3 rm s3://talentflow-backups/database/$fileName
    fi
  fi
done
```

### 2. S3 Cross-Region Replication

```hcl
resource "aws_s3_bucket_replication_configuration" "resumes" {
  role   = aws_iam_role.replication.arn
  bucket = aws_s3_bucket.resumes.id

  rule {
    id     = "ReplicateToSecondaryRegion"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.resumes_replica.arn
      storage_class = "STANDARD_IA"
    }
  }

  depends_on = [aws_s3_bucket_versioning.resumes]
}

resource "aws_s3_bucket" "resumes_replica" {
  provider = aws.replica
  bucket   = "${var.name_prefix}-resumes-replica"
  
  tags = var.tags
}
```

### 3. Disaster Recovery Plan

#### Multi-Region Setup

```hcl
# Primary region provider
provider "aws" {
  alias  = "primary"
  region = "us-east-1"
}

# Secondary region provider
provider "aws" {
  alias  = "secondary"
  region = "us-west-2"
}

# Deploy infrastructure in secondary region
module "secondary_infrastructure" {
  source = "./modules/infrastructure"
  
  providers = {
    aws = aws.secondary
  }
  
  name_prefix = "${var.name_prefix}-dr"
  # ... other configuration ...
}
```

#### Failover Procedure

```bash
#!/bin/bash
# scripts/failover.sh

echo "Starting disaster recovery failover..."

# 1. Update Route 53 to point to secondary region
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456789 \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.talentflow.com",
        "Type": "CNAME",
        "TTL": 60,
        "ResourceRecords": [{"Value": "secondary-alb-dns-name"}]
      }
    }]
  }'

# 2. Restore database from latest backup
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier talentflow-dr \
  --db-snapshot-identifier talentflow-backup-latest

# 3. Update ECS services to use restored database
aws ecs update-service \
  --cluster talentflow-ai-dr-cluster \
  --service talentflow-ai-backend-service \
  --force-new-deployment

echo "Failover completed. Monitor services for stability."
```

## Troubleshooting

### 1. Common Issues

#### ECS Service Won't Start

```bash
# Check service events
aws ecs describe-services \
  --cluster talentflow-ai-cluster \
  --services talentflow-ai-backend-service

# Check task definition
aws ecs describe-task-definition \
  --task-definition talentflow-ai-backend

# Check logs
aws logs get-log-events \
  --log-group-name "/ecs/talentflow-backend" \
  --log-stream-name "ecs/backend/task-id"
```

#### Database Connection Issues

```bash
# Test database connectivity
psql -h your-rds-endpoint -U postgres -d talentflow -c "SELECT 1;"

# Check security groups
aws ec2 describe-security-groups \
  --group-ids sg-12345678 \
  --query 'SecurityGroups[0].IpPermissions'

# Check RDS status
aws rds describe-db-instances \
  --db-instance-identifier talentflow-prod
```

#### High Memory Usage

```bash
# Check ECS service metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name MemoryUtilization \
  --dimensions Name=ServiceName,Value=talentflow-ai-backend-service \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T01:00:00Z \
  --period 300 \
  --statistics Average,Maximum

# Scale up service if needed
aws ecs update-service \
  --cluster talentflow-ai-cluster \
  --service talentflow-ai-backend-service \
  --desired-count 4
```

### 2. Debugging Tools

#### ECS Exec for Container Access

```bash
# Enable ECS Exec on service
aws ecs update-service \
  --cluster talentflow-ai-cluster \
  --service talentflow-ai-backend-service \
  --enable-execute-command

# Connect to running container
aws ecs execute-command \
  --cluster talentflow-ai-cluster \
  --task task-id \
  --container backend \
  --interactive \
  --command "/bin/bash"
```

#### CloudWatch Insights Queries

```sql
-- Find errors in logs
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 100

-- API response times
fields @timestamp, @message
| filter @message like /response_time/
| stats avg(response_time) by bin(5m)

-- Memory usage patterns
fields @timestamp, @message
| filter @message like /memory_usage/
| sort @timestamp desc
| limit 50
```

### 3. Performance Optimization

#### Auto Scaling Configuration

```hcl
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.backend.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_policy_cpu" {
  name               = "${var.name_prefix}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
```

#### Database Performance Tuning

```sql
-- Monitor slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE tablename = 'candidates';

-- Analyze table statistics
ANALYZE candidates;
ANALYZE jobs;
ANALYZE scores;
```

---

This deployment guide provides comprehensive coverage of AWS infrastructure setup, CI/CD pipeline configuration, and production deployment processes for TalentFlow AI. Follow the steps sequentially and customize the configuration based on your specific requirements and environment.

For additional support or questions, refer to the project documentation or create an issue in the repository.