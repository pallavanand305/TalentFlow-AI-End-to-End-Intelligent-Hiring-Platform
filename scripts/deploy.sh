#!/bin/bash

# TalentFlow AI Deployment Script
# This script handles the complete deployment process for TalentFlow AI

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AWS_REGION="${AWS_REGION:-us-east-1}"
ENVIRONMENT="${ENVIRONMENT:-prod}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required tools are installed
    local tools=("aws" "terraform" "docker" "jq")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        exit 1
    fi
    
    # Check if required environment variables are set
    local required_vars=("DB_PASSWORD" "SECRET_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_error "Environment variable $var is not set"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying infrastructure..."
    
    cd "$PROJECT_ROOT/infra/terraform"
    
    # Initialize Terraform
    terraform init
    
    # Select workspace
    terraform workspace select "$ENVIRONMENT" || terraform workspace new "$ENVIRONMENT"
    
    # Plan infrastructure changes
    log_info "Planning infrastructure changes..."
    terraform plan \
        -var="environment=$ENVIRONMENT" \
        -var="db_password=$DB_PASSWORD" \
        -var="secret_key=$SECRET_KEY" \
        -var="aws_access_key_id=${APP_AWS_ACCESS_KEY_ID}" \
        -var="aws_secret_access_key=${APP_AWS_SECRET_ACCESS_KEY}" \
        -var="certificate_arn=${SSL_CERTIFICATE_ARN}" \
        -var="domain_name=${DOMAIN_NAME}" \
        -out=tfplan
    
    # Apply infrastructure changes
    log_info "Applying infrastructure changes..."
    terraform apply -auto-approve tfplan
    
    # Get outputs
    terraform output -json > terraform-outputs.json
    
    log_success "Infrastructure deployment completed"
    
    cd "$PROJECT_ROOT"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    # Get ECR login
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com"
    
    # Get ECR repository URLs from Terraform outputs
    local backend_repo=$(jq -r '.ecr_backend_url.value' "$PROJECT_ROOT/infra/terraform/terraform-outputs.json")
    local ml_repo=$(jq -r '.ecr_ml_url.value' "$PROJECT_ROOT/infra/terraform/terraform-outputs.json")
    
    # Build and push backend image
    log_info "Building backend image..."
    docker build -f docker/Dockerfile.backend -t "$backend_repo:latest" -t "$backend_repo:$(git rev-parse --short HEAD)" .
    
    log_info "Pushing backend image..."
    docker push "$backend_repo:latest"
    docker push "$backend_repo:$(git rev-parse --short HEAD)"
    
    # Build and push ML image
    log_info "Building ML image..."
    docker build -f docker/Dockerfile.ml -t "$ml_repo:latest" -t "$ml_repo:$(git rev-parse --short HEAD)" .
    
    log_info "Pushing ML image..."
    docker push "$ml_repo:latest"
    docker push "$ml_repo:$(git rev-parse --short HEAD)"
    
    log_success "Docker images built and pushed successfully"
}

# Deploy ECS services
deploy_services() {
    log_info "Deploying ECS services..."
    
    # Get cluster name from Terraform outputs
    local cluster_name=$(jq -r '.ecs_cluster_name.value' "$PROJECT_ROOT/infra/terraform/terraform-outputs.json")
    
    # Update backend service
    log_info "Updating backend service..."
    aws ecs update-service \
        --cluster "$cluster_name" \
        --service "talentflow-ai-backend-service" \
        --force-new-deployment \
        --no-cli-pager
    
    # Update worker service
    log_info "Updating worker service..."
    aws ecs update-service \
        --cluster "$cluster_name" \
        --service "talentflow-ai-worker-service" \
        --force-new-deployment \
        --no-cli-pager
    
    # Update MLflow service
    log_info "Updating MLflow service..."
    aws ecs update-service \
        --cluster "$cluster_name" \
        --service "talentflow-ai-mlflow-service" \
        --force-new-deployment \
        --no-cli-pager
    
    # Wait for services to stabilize
    log_info "Waiting for services to stabilize..."
    aws ecs wait services-stable \
        --cluster "$cluster_name" \
        --services "talentflow-ai-backend-service" "talentflow-ai-worker-service" "talentflow-ai-mlflow-service" \
        --no-cli-pager
    
    log_success "ECS services deployed successfully"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Get database endpoint from Terraform outputs
    local db_endpoint=$(jq -r '.rds_endpoint.value' "$PROJECT_ROOT/infra/terraform/terraform-outputs.json")
    
    # Set database URL
    export DATABASE_URL="postgresql+asyncpg://postgres:$DB_PASSWORD@$db_endpoint:5432/talentflow"
    
    # Run migrations
    cd "$PROJECT_ROOT"
    alembic upgrade head
    
    log_success "Database migrations completed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Get ALB DNS name from Terraform outputs
    local alb_dns=$(jq -r '.alb_dns_name.value' "$PROJECT_ROOT/infra/terraform/terraform-outputs.json")
    
    # Health check
    local health_url="https://$alb_dns/health"
    if [[ -n "$DOMAIN_NAME" ]]; then
        health_url="https://$DOMAIN_NAME/health"
    fi
    
    log_info "Checking health endpoint: $health_url"
    
    # Wait for service to be ready
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$health_url" > /dev/null; then
            log_success "Health check passed"
            break
        else
            log_warning "Health check failed (attempt $attempt/$max_attempts)"
            if [[ $attempt -eq $max_attempts ]]; then
                log_error "Health check failed after $max_attempts attempts"
                exit 1
            fi
            sleep 10
            ((attempt++))
        fi
    done
    
    # Test API endpoints
    log_info "Testing API endpoints..."
    
    # Test authentication endpoint
    local auth_response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$health_url/../api/v1/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"username":"test","password":"test"}')
    
    if [[ "$auth_response" == "422" || "$auth_response" == "401" ]]; then
        log_success "Authentication endpoint is responding"
    else
        log_warning "Authentication endpoint returned unexpected status: $auth_response"
    fi
    
    log_success "Deployment verification completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f "$PROJECT_ROOT/infra/terraform/tfplan"
    rm -f "$PROJECT_ROOT/infra/terraform/terraform-outputs.json"
}

# Main deployment function
main() {
    log_info "Starting TalentFlow AI deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "AWS Region: $AWS_REGION"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    deploy_infrastructure
    build_and_push_images
    deploy_services
    run_migrations
    verify_deployment
    
    log_success "🚀 TalentFlow AI deployment completed successfully!"
    
    # Display important information
    echo ""
    echo "=== Deployment Information ==="
    if [[ -f "$PROJECT_ROOT/infra/terraform/terraform-outputs.json" ]]; then
        echo "ALB DNS Name: $(jq -r '.alb_dns_name.value' "$PROJECT_ROOT/infra/terraform/terraform-outputs.json")"
        echo "S3 Bucket (Resumes): $(jq -r '.s3_bucket_resumes.value' "$PROJECT_ROOT/infra/terraform/terraform-outputs.json")"
        echo "S3 Bucket (Models): $(jq -r '.s3_bucket_models.value' "$PROJECT_ROOT/infra/terraform/terraform-outputs.json")"
        echo "ECS Cluster: $(jq -r '.ecs_cluster_name.value' "$PROJECT_ROOT/infra/terraform/terraform-outputs.json")"
    fi
    echo "Environment: $ENVIRONMENT"
    echo "Git Commit: $(git rev-parse --short HEAD)"
    echo "Deployment Time: $(date)"
    echo "=============================="
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "infrastructure")
        check_prerequisites
        deploy_infrastructure
        ;;
    "images")
        check_prerequisites
        build_and_push_images
        ;;
    "services")
        check_prerequisites
        deploy_services
        ;;
    "migrations")
        check_prerequisites
        run_migrations
        ;;
    "verify")
        verify_deployment
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy         Full deployment (default)"
        echo "  infrastructure Deploy infrastructure only"
        echo "  images         Build and push Docker images only"
        echo "  services       Deploy ECS services only"
        echo "  migrations     Run database migrations only"
        echo "  verify         Verify deployment only"
        echo "  help           Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  ENVIRONMENT    Deployment environment (default: prod)"
        echo "  AWS_REGION     AWS region (default: us-east-1)"
        echo "  DB_PASSWORD    Database password (required)"
        echo "  SECRET_KEY     JWT secret key (required)"
        echo ""
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac