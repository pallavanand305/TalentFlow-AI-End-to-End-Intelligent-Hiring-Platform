#!/bin/bash

# TalentFlow AI Infrastructure Deployment Script
# This script deploys the complete AWS infrastructure for TalentFlow AI

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
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

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENV     Target environment (prod|staging) [default: prod]"
    echo "  -r, --region REGION       AWS region [default: us-east-1]"
    echo "  -p, --plan-only          Only run terraform plan, don't apply"
    echo "  -d, --destroy            Destroy infrastructure instead of creating"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Required Environment Variables:"
    echo "  DB_PASSWORD              Database password (minimum 8 characters)"
    echo "  SECRET_KEY               JWT secret key (minimum 32 characters)"
    echo "  APP_AWS_ACCESS_KEY_ID    Application AWS access key ID"
    echo "  APP_AWS_SECRET_ACCESS_KEY Application AWS secret access key"
    echo ""
    echo "Optional Environment Variables:"
    echo "  REDIS_AUTH_TOKEN         Redis authentication token"
    echo "  SSL_CERTIFICATE_ARN      SSL certificate ARN for HTTPS"
    echo "  DOMAIN_NAME              Domain name for the application"
    echo ""
    echo "Examples:"
    echo "  $0                       Deploy to production"
    echo "  $0 -e staging            Deploy to staging"
    echo "  $0 -p                    Plan only (no apply)"
    echo "  $0 -d                    Destroy infrastructure"
    echo ""
}

# Parse command line arguments
PLAN_ONLY=false
DESTROY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -p|--plan-only)
            PLAN_ONLY=true
            shift
            ;;
        -d|--destroy)
            DESTROY=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ "$ENVIRONMENT" != "prod" && "$ENVIRONMENT" != "staging" ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be 'prod' or 'staging'"
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required tools are installed
    local tools=("aws" "terraform" "jq" "git")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            log_info "Please install required tools. See README.md for installation instructions."
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        log_info "Run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        exit 1
    fi
    
    # Check required environment variables
    local required_vars=("DB_PASSWORD" "SECRET_KEY" "APP_AWS_ACCESS_KEY_ID" "APP_AWS_SECRET_ACCESS_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_error "Environment variable $var is not set"
            log_info "Please set all required environment variables. See README.md for details."
            exit 1
        fi
    done
    
    # Validate password strength
    if [[ ${#DB_PASSWORD} -lt 8 ]]; then
        log_error "DB_PASSWORD must be at least 8 characters long"
        exit 1
    fi
    
    if [[ ${#SECRET_KEY} -lt 32 ]]; then
        log_error "SECRET_KEY must be at least 32 characters long"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Initialize Terraform
init_terraform() {
    log_info "Initializing Terraform..."
    
    cd "$SCRIPT_DIR"
    
    # Check if this is the first run (no backend configured)
    if ! terraform init -backend=false &> /dev/null; then
        log_error "Terraform initialization failed"
        exit 1
    fi
    
    # Check if backend resources exist
    local backend_exists=false
    if aws s3api head-bucket --bucket "talentflow-ai-terraform-state" &> /dev/null; then
        backend_exists=true
    fi
    
    if [[ "$backend_exists" == "false" ]]; then
        log_info "Creating Terraform backend resources..."
        
        # Create backend resources first
        terraform apply -auto-approve \
            -target=aws_s3_bucket.terraform_state \
            -target=aws_dynamodb_table.terraform_locks \
            -var="environment=$ENVIRONMENT" \
            -var="db_password=dummy" \
            -var="secret_key=dummy-secret-key-32-characters-long" \
            -var="aws_access_key_id=dummy" \
            -var="aws_secret_access_key=dummy"
        
        # Get the actual bucket name
        local bucket_name=$(terraform output -raw terraform_backend_config | jq -r '.bucket')
        
        # Update main.tf with the correct bucket name
        sed -i.bak "s/talentflow-ai-terraform-state/$bucket_name/g" main.tf
        
        log_info "Re-initializing Terraform with backend..."
        terraform init -migrate-state -force-copy
    else
        log_info "Backend already exists, initializing with backend..."
        terraform init
    fi
    
    # Select or create workspace
    terraform workspace select "$ENVIRONMENT" || terraform workspace new "$ENVIRONMENT"
    
    log_success "Terraform initialized successfully"
}

# Plan infrastructure changes
plan_infrastructure() {
    log_info "Planning infrastructure changes for environment: $ENVIRONMENT"
    
    local tf_vars=(
        "-var-file=environments/${ENVIRONMENT}.tfvars"
        "-var=db_password=$DB_PASSWORD"
        "-var=secret_key=$SECRET_KEY"
        "-var=aws_access_key_id=$APP_AWS_ACCESS_KEY_ID"
        "-var=aws_secret_access_key=$APP_AWS_SECRET_ACCESS_KEY"
    )
    
    # Add optional variables if set
    if [[ -n "$REDIS_AUTH_TOKEN" ]]; then
        tf_vars+=("-var=redis_auth_token=$REDIS_AUTH_TOKEN")
    fi
    
    if [[ -n "$SSL_CERTIFICATE_ARN" ]]; then
        tf_vars+=("-var=certificate_arn=$SSL_CERTIFICATE_ARN")
    fi
    
    if [[ -n "$DOMAIN_NAME" ]]; then
        tf_vars+=("-var=domain_name=$DOMAIN_NAME")
    fi
    
    if [[ "$DESTROY" == "true" ]]; then
        log_warning "Planning infrastructure DESTRUCTION..."
        terraform plan -destroy "${tf_vars[@]}" -out=destroy-plan
    else
        terraform plan "${tf_vars[@]}" -out=tfplan
    fi
    
    log_success "Infrastructure plan completed"
}

# Apply infrastructure changes
apply_infrastructure() {
    if [[ "$PLAN_ONLY" == "true" ]]; then
        log_info "Plan-only mode enabled. Skipping apply."
        return
    fi
    
    if [[ "$DESTROY" == "true" ]]; then
        log_warning "This will DESTROY all infrastructure for environment: $ENVIRONMENT"
        log_warning "This action cannot be undone!"
        echo ""
        read -p "Are you absolutely sure you want to destroy the infrastructure? (type 'yes' to confirm): " confirm
        
        if [[ "$confirm" != "yes" ]]; then
            log_info "Destruction cancelled"
            exit 0
        fi
        
        log_info "Destroying infrastructure..."
        terraform apply -auto-approve destroy-plan
        log_success "Infrastructure destroyed successfully"
        return
    fi
    
    log_info "Applying infrastructure changes..."
    terraform apply -auto-approve tfplan
    
    # Save outputs
    terraform output -json > terraform-outputs.json
    
    log_success "Infrastructure deployment completed successfully"
}

# Display deployment information
show_deployment_info() {
    if [[ "$DESTROY" == "true" || "$PLAN_ONLY" == "true" ]]; then
        return
    fi
    
    log_info "Deployment completed successfully!"
    echo ""
    echo "=== Deployment Information ==="
    
    if [[ -f "terraform-outputs.json" ]]; then
        echo "Environment: $(jq -r '.environment.value' terraform-outputs.json)"
        echo "AWS Region: $(jq -r '.aws_region.value' terraform-outputs.json)"
        echo "VPC ID: $(jq -r '.vpc_id.value' terraform-outputs.json)"
        echo "ECS Cluster: $(jq -r '.ecs_cluster_name.value' terraform-outputs.json)"
        echo "ALB DNS: $(jq -r '.alb_dns_name.value' terraform-outputs.json)"
        echo "Application URL: $(jq -r '.application_url.value' terraform-outputs.json)"
        echo "MLflow URL: $(jq -r '.mlflow_url.value' terraform-outputs.json)"
        echo "S3 Bucket (Resumes): $(jq -r '.s3_bucket_resumes.value' terraform-outputs.json)"
        echo "S3 Bucket (Models): $(jq -r '.s3_bucket_models.value' terraform-outputs.json)"
        echo "RDS Endpoint: $(jq -r '.rds_endpoint.value' terraform-outputs.json)"
        echo "Redis Endpoint: $(jq -r '.redis_endpoint.value' terraform-outputs.json)"
    fi
    
    echo "Deployment Time: $(date)"
    echo "Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo "=============================="
    echo ""
    
    log_info "Next steps:"
    echo "1. Build and push Docker images: cd ../../ && ./scripts/deploy.sh images"
    echo "2. Deploy ECS services: ./scripts/deploy.sh services"
    echo "3. Run database migrations: ./scripts/deploy.sh migrations"
    echo "4. Verify deployment: ./scripts/deploy.sh verify"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f tfplan destroy-plan
}

# Main function
main() {
    log_info "Starting TalentFlow AI infrastructure deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "AWS Region: $AWS_REGION"
    
    if [[ "$DESTROY" == "true" ]]; then
        log_warning "DESTROY MODE ENABLED"
    elif [[ "$PLAN_ONLY" == "true" ]]; then
        log_info "PLAN-ONLY MODE ENABLED"
    fi
    
    echo ""
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    init_terraform
    plan_infrastructure
    apply_infrastructure
    show_deployment_info
    
    if [[ "$DESTROY" == "true" ]]; then
        log_success "🗑️  Infrastructure destruction completed!"
    elif [[ "$PLAN_ONLY" == "true" ]]; then
        log_success "📋 Infrastructure planning completed!"
    else
        log_success "🚀 Infrastructure deployment completed!"
    fi
}

# Run main function
main "$@"