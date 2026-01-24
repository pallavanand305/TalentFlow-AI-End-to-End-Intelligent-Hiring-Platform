#!/bin/bash

# TalentFlow AI Rollback Script
# This script handles rollback to a previous deployment

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

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --target-revision REVISION    Target Git revision to rollback to"
    echo "  -s, --service SERVICE             Rollback specific service only (backend|worker|mlflow)"
    echo "  -d, --dry-run                     Show what would be done without executing"
    echo "  -f, --force                       Force rollback without confirmation"
    echo "  -h, --help                        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --target-revision abc123       Rollback to specific Git commit"
    echo "  $0 --service backend              Rollback only the backend service"
    echo "  $0 --dry-run                      Preview rollback actions"
    echo ""
}

# Parse command line arguments
TARGET_REVISION=""
TARGET_SERVICE=""
DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target-revision)
            TARGET_REVISION="$2"
            shift 2
            ;;
        -s|--service)
            TARGET_SERVICE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required tools are installed
    local tools=("aws" "jq" "git")
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
    
    log_success "Prerequisites check passed"
}

# Get current deployment info
get_current_deployment() {
    log_info "Getting current deployment information..."
    
    local cluster_name="talentflow-ai-cluster"
    local services=("talentflow-ai-backend-service" "talentflow-ai-worker-service" "talentflow-ai-mlflow-service")
    
    echo "Current deployment status:"
    for service in "${services[@]}"; do
        local task_def=$(aws ecs describe-services \
            --cluster "$cluster_name" \
            --services "$service" \
            --query 'services[0].taskDefinition' \
            --output text)
        
        local image=$(aws ecs describe-task-definition \
            --task-definition "$task_def" \
            --query 'taskDefinition.containerDefinitions[0].image' \
            --output text)
        
        echo "  $service: $image"
    done
}

# List available revisions
list_available_revisions() {
    log_info "Available Git revisions (last 10):"
    git log --oneline -10
    echo ""
}

# Get ECR images for a revision
get_ecr_images() {
    local revision="$1"
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local backend_repo="$account_id.dkr.ecr.$AWS_REGION.amazonaws.com/talentflow-ai-backend"
    local ml_repo="$account_id.dkr.ecr.$AWS_REGION.amazonaws.com/talentflow-ai-ml"
    
    # Check if images exist for the revision
    local backend_image="$backend_repo:$revision"
    local ml_image="$ml_repo:$revision"
    
    # Verify images exist
    if aws ecr describe-images --repository-name "talentflow-ai-backend" --image-ids imageTag="$revision" &> /dev/null; then
        echo "backend:$backend_image"
    else
        log_warning "Backend image not found for revision $revision"
        echo "backend:$backend_repo:latest"
    fi
    
    if aws ecr describe-images --repository-name "talentflow-ai-ml" --image-ids imageTag="$revision" &> /dev/null; then
        echo "ml:$ml_image"
    else
        log_warning "ML image not found for revision $revision"
        echo "ml:$ml_repo:latest"
    fi
}

# Create new task definition with rollback image
create_rollback_task_definition() {
    local service_name="$1"
    local new_image="$2"
    local cluster_name="talentflow-ai-cluster"
    
    # Get current task definition
    local current_task_def=$(aws ecs describe-services \
        --cluster "$cluster_name" \
        --services "$service_name" \
        --query 'services[0].taskDefinition' \
        --output text)
    
    # Get task definition details
    local task_def_json=$(aws ecs describe-task-definition \
        --task-definition "$current_task_def" \
        --query 'taskDefinition')
    
    # Update image in task definition
    local new_task_def=$(echo "$task_def_json" | jq \
        --arg new_image "$new_image" \
        '.containerDefinitions[0].image = $new_image | 
         del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .placementConstraints, .compatibilities, .registeredAt, .registeredBy)')
    
    # Register new task definition
    local new_task_def_arn=$(echo "$new_task_def" | aws ecs register-task-definition \
        --cli-input-json file:///dev/stdin \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)
    
    echo "$new_task_def_arn"
}

# Rollback service
rollback_service() {
    local service_name="$1"
    local target_image="$2"
    local cluster_name="talentflow-ai-cluster"
    
    log_info "Rolling back service: $service_name"
    log_info "Target image: $target_image"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback $service_name to $target_image"
        return
    fi
    
    # Create new task definition
    local new_task_def_arn=$(create_rollback_task_definition "$service_name" "$target_image")
    log_info "Created new task definition: $new_task_def_arn"
    
    # Update service
    aws ecs update-service \
        --cluster "$cluster_name" \
        --service "$service_name" \
        --task-definition "$new_task_def_arn" \
        --no-cli-pager
    
    log_success "Service $service_name rollback initiated"
}

# Wait for rollback to complete
wait_for_rollback() {
    local services=("$@")
    local cluster_name="talentflow-ai-cluster"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would wait for services to stabilize"
        return
    fi
    
    log_info "Waiting for rollback to complete..."
    
    # Convert array to space-separated string for AWS CLI
    local services_str=$(printf "%s " "${services[@]}")
    
    aws ecs wait services-stable \
        --cluster "$cluster_name" \
        --services $services_str \
        --no-cli-pager
    
    log_success "Rollback completed successfully"
}

# Verify rollback
verify_rollback() {
    log_info "Verifying rollback..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would verify rollback"
        return
    fi
    
    # Health check
    local max_attempts=10
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "https://api.talentflow.com/health" > /dev/null; then
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
    
    log_success "Rollback verification completed"
}

# Main rollback function
main() {
    log_info "Starting TalentFlow AI rollback..."
    log_info "Environment: $ENVIRONMENT"
    log_info "AWS Region: $AWS_REGION"
    
    check_prerequisites
    
    # Show current deployment
    get_current_deployment
    echo ""
    
    # If no target revision specified, show available options
    if [[ -z "$TARGET_REVISION" ]]; then
        list_available_revisions
        
        if [[ "$FORCE" != "true" ]]; then
            read -p "Enter target revision to rollback to: " TARGET_REVISION
            if [[ -z "$TARGET_REVISION" ]]; then
                log_error "No target revision specified"
                exit 1
            fi
        else
            log_error "Target revision required when using --force"
            exit 1
        fi
    fi
    
    # Validate target revision
    if ! git rev-parse --verify "$TARGET_REVISION" &> /dev/null; then
        log_error "Invalid Git revision: $TARGET_REVISION"
        exit 1
    fi
    
    log_info "Target revision: $TARGET_REVISION"
    
    # Get ECR images for target revision
    local images=$(get_ecr_images "$TARGET_REVISION")
    local backend_image=$(echo "$images" | grep "^backend:" | cut -d: -f2-)
    local ml_image=$(echo "$images" | grep "^ml:" | cut -d: -f2-)
    
    # Determine which services to rollback
    local services_to_rollback=()
    
    if [[ -n "$TARGET_SERVICE" ]]; then
        case "$TARGET_SERVICE" in
            "backend")
                services_to_rollback=("talentflow-ai-backend-service")
                ;;
            "worker")
                services_to_rollback=("talentflow-ai-worker-service")
                ;;
            "mlflow")
                services_to_rollback=("talentflow-ai-mlflow-service")
                ;;
            *)
                log_error "Invalid service: $TARGET_SERVICE"
                exit 1
                ;;
        esac
    else
        services_to_rollback=("talentflow-ai-backend-service" "talentflow-ai-worker-service")
    fi
    
    # Show rollback plan
    echo ""
    log_info "Rollback plan:"
    for service in "${services_to_rollback[@]}"; do
        case "$service" in
            "talentflow-ai-backend-service")
                echo "  $service -> $backend_image"
                ;;
            "talentflow-ai-worker-service")
                echo "  $service -> $ml_image"
                ;;
            "talentflow-ai-mlflow-service")
                echo "  $service -> (MLflow image)"
                ;;
        esac
    done
    echo ""
    
    # Confirmation
    if [[ "$FORCE" != "true" && "$DRY_RUN" != "true" ]]; then
        read -p "Are you sure you want to proceed with the rollback? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            log_info "Rollback cancelled"
            exit 0
        fi
    fi
    
    # Execute rollback
    for service in "${services_to_rollback[@]}"; do
        case "$service" in
            "talentflow-ai-backend-service")
                rollback_service "$service" "$backend_image"
                ;;
            "talentflow-ai-worker-service")
                rollback_service "$service" "$ml_image"
                ;;
            "talentflow-ai-mlflow-service")
                # MLflow uses the same ML image
                rollback_service "$service" "$ml_image"
                ;;
        esac
    done
    
    # Wait for rollback to complete
    wait_for_rollback "${services_to_rollback[@]}"
    
    # Verify rollback
    verify_rollback
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_success "🔍 Rollback dry run completed"
    else
        log_success "🔄 Rollback completed successfully!"
        
        echo ""
        echo "=== Rollback Information ==="
        echo "Target Revision: $TARGET_REVISION"
        echo "Services Rolled Back: ${services_to_rollback[*]}"
        echo "Environment: $ENVIRONMENT"
        echo "Rollback Time: $(date)"
        echo "============================"
    fi
}

# Run main function
main "$@"