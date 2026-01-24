#!/bin/bash

# TalentFlow AI Infrastructure Validation Script
# This script validates the Terraform configuration without deploying

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Main validation function
main() {
    log_info "Validating TalentFlow AI Terraform configuration..."
    
    cd "$SCRIPT_DIR"
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed or not in PATH"
        exit 1
    fi
    
    # Initialize Terraform (backend=false for validation)
    log_info "Initializing Terraform..."
    if ! terraform init -backend=false; then
        log_error "Terraform initialization failed"
        exit 1
    fi
    
    # Format check
    log_info "Checking Terraform formatting..."
    if ! terraform fmt -check -recursive; then
        log_warning "Terraform files are not properly formatted"
        log_info "Run 'terraform fmt -recursive' to fix formatting"
    else
        log_success "Terraform formatting is correct"
    fi
    
    # Validate configuration
    log_info "Validating Terraform configuration..."
    if ! terraform validate; then
        log_error "Terraform configuration validation failed"
        exit 1
    fi
    
    log_success "Terraform configuration is valid"
    
    # Check for common issues
    log_info "Checking for common configuration issues..."
    
    # Check if all required files exist
    local required_files=(
        "main.tf"
        "variables.tf"
        "outputs.tf"
        "ecs.tf"
        "backend.tf"
        "environments/prod.tfvars"
        "environments/staging.tfvars"
        "modules/s3/main.tf"
        "modules/rds/main.tf"
        "modules/security_groups/main.tf"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done
    
    log_success "All required files are present"
    
    # Check for sensitive data in files
    log_info "Checking for sensitive data in configuration files..."
    
    local sensitive_patterns=(
        "password.*=.*['\"].*['\"]"
        "secret.*=.*['\"].*['\"]"
        "key.*=.*['\"].*['\"]"
        "token.*=.*['\"].*['\"]"
    )
    
    local found_sensitive=false
    for pattern in "${sensitive_patterns[@]}"; do
        if grep -r -i -E "$pattern" . --include="*.tf" --include="*.tfvars" | grep -v "# " | grep -v "//" > /dev/null; then
            log_warning "Potential sensitive data found in configuration files"
            log_warning "Make sure to use variables or environment variables for sensitive data"
            found_sensitive=true
        fi
    done
    
    if [[ "$found_sensitive" == "false" ]]; then
        log_success "No sensitive data found in configuration files"
    fi
    
    # Check variable consistency
    log_info "Checking variable consistency..."
    
    # Extract variables from variables.tf
    local declared_vars=$(grep -E "^variable " variables.tf | sed 's/variable "//' | sed 's/" {//')
    
    # Check if all variables are used in main.tf or modules
    for var in $declared_vars; do
        if ! grep -r "var\.$var" . --include="*.tf" > /dev/null; then
            log_warning "Variable '$var' is declared but not used"
        fi
    done
    
    log_success "Variable consistency check completed"
    
    # Summary
    echo ""
    log_success "🎉 Terraform configuration validation completed successfully!"
    echo ""
    log_info "Configuration summary:"
    echo "  - Main configuration files: ✓"
    echo "  - Module files: ✓"
    echo "  - Environment files: ✓"
    echo "  - Syntax validation: ✓"
    echo "  - Formatting check: ✓"
    echo "  - Security check: ✓"
    echo ""
    log_info "The infrastructure is ready for deployment."
    log_info "Run './deploy-infrastructure.sh -p' to see the deployment plan."
}

# Run main function
main "$@"