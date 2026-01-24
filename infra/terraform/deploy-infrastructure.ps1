# TalentFlow AI Infrastructure Deployment Script (PowerShell)
# This script deploys the complete AWS infrastructure for TalentFlow AI

param(
    [string]$Environment = "prod",
    [string]$Region = "us-east-1",
    [switch]$PlanOnly,
    [switch]$Destroy,
    [switch]$Help
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
}

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

# Show usage
function Show-Usage {
    Write-Host "Usage: .\deploy-infrastructure.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Environment ENV     Target environment (prod|staging) [default: prod]"
    Write-Host "  -Region REGION       AWS region [default: us-east-1]"
    Write-Host "  -PlanOnly           Only run terraform plan, don't apply"
    Write-Host "  -Destroy            Destroy infrastructure instead of creating"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "Required Environment Variables:"
    Write-Host "  DB_PASSWORD              Database password (minimum 8 characters)"
    Write-Host "  SECRET_KEY               JWT secret key (minimum 32 characters)"
    Write-Host "  APP_AWS_ACCESS_KEY_ID    Application AWS access key ID"
    Write-Host "  APP_AWS_SECRET_ACCESS_KEY Application AWS secret access key"
    Write-Host ""
    Write-Host "Optional Environment Variables:"
    Write-Host "  REDIS_AUTH_TOKEN         Redis authentication token"
    Write-Host "  SSL_CERTIFICATE_ARN      SSL certificate ARN for HTTPS"
    Write-Host "  DOMAIN_NAME              Domain name for the application"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy-infrastructure.ps1                    Deploy to production"
    Write-Host "  .\deploy-infrastructure.ps1 -Environment staging Deploy to staging"
    Write-Host "  .\deploy-infrastructure.ps1 -PlanOnly          Plan only (no apply)"
    Write-Host "  .\deploy-infrastructure.ps1 -Destroy           Destroy infrastructure"
    Write-Host ""
}

# Show help if requested
if ($Help) {
    Show-Usage
    exit 0
}

# Validate environment
if ($Environment -notin @("prod", "staging")) {
    Write-Error "Invalid environment: $Environment. Must be 'prod' or 'staging'"
    exit 1
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check if required tools are installed
    $tools = @("aws", "terraform", "git")
    foreach ($tool in $tools) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            Write-Error "$tool is not installed or not in PATH"
            Write-Info "Please install required tools. See README.md for installation instructions."
            exit 1
        }
    }
    
    # Check AWS credentials
    try {
        aws sts get-caller-identity | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "AWS credentials check failed"
        }
    }
    catch {
        Write-Error "AWS credentials not configured or invalid"
        Write-Info "Run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        exit 1
    }
    
    # Check required environment variables
    $requiredVars = @("DB_PASSWORD", "SECRET_KEY", "APP_AWS_ACCESS_KEY_ID", "APP_AWS_SECRET_ACCESS_KEY")
    foreach ($var in $requiredVars) {
        if (-not (Get-Item "env:$var" -ErrorAction SilentlyContinue)) {
            Write-Error "Environment variable $var is not set"
            Write-Info "Please set all required environment variables. See README.md for details."
            exit 1
        }
    }
    
    # Validate password strength
    if ($env:DB_PASSWORD.Length -lt 8) {
        Write-Error "DB_PASSWORD must be at least 8 characters long"
        exit 1
    }
    
    if ($env:SECRET_KEY.Length -lt 32) {
        Write-Error "SECRET_KEY must be at least 32 characters long"
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

# Initialize Terraform
function Initialize-Terraform {
    Write-Info "Initializing Terraform..."
    
    Set-Location $ScriptDir
    
    # Check if this is the first run (no backend configured)
    terraform init -backend=false | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Terraform initialization failed"
        exit 1
    }
    
    # Check if backend resources exist
    $backendExists = $false
    try {
        aws s3api head-bucket --bucket "talentflow-ai-terraform-state" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $backendExists = $true
        }
    }
    catch {
        # Bucket doesn't exist
    }
    
    if (-not $backendExists) {
        Write-Info "Creating Terraform backend resources..."
        
        # Create backend resources first
        terraform apply -auto-approve `
            -target=aws_s3_bucket.terraform_state `
            -target=aws_dynamodb_table.terraform_locks `
            -var="environment=$Environment" `
            -var="db_password=dummy" `
            -var="secret_key=dummy-secret-key-32-characters-long" `
            -var="aws_access_key_id=dummy" `
            -var="aws_secret_access_key=dummy"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create backend resources"
            exit 1
        }
        
        # Get the actual bucket name
        $backendConfig = terraform output -json terraform_backend_config | ConvertFrom-Json
        $bucketName = $backendConfig.bucket
        
        # Update main.tf with the correct bucket name
        $mainTfContent = Get-Content "main.tf" -Raw
        $mainTfContent = $mainTfContent -replace "talentflow-ai-terraform-state", $bucketName
        Set-Content "main.tf" $mainTfContent
        
        Write-Info "Re-initializing Terraform with backend..."
        terraform init -migrate-state -force-copy
    }
    else {
        Write-Info "Backend already exists, initializing with backend..."
        terraform init
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Terraform initialization failed"
        exit 1
    }
    
    # Select or create workspace
    terraform workspace select $Environment
    if ($LASTEXITCODE -ne 0) {
        terraform workspace new $Environment
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create workspace: $Environment"
            exit 1
        }
    }
    
    Write-Success "Terraform initialized successfully"
}

# Plan infrastructure changes
function Invoke-TerraformPlan {
    Write-Info "Planning infrastructure changes for environment: $Environment"
    
    $tfVars = @(
        "-var-file=environments\$Environment.tfvars",
        "-var=db_password=$env:DB_PASSWORD",
        "-var=secret_key=$env:SECRET_KEY",
        "-var=aws_access_key_id=$env:APP_AWS_ACCESS_KEY_ID",
        "-var=aws_secret_access_key=$env:APP_AWS_SECRET_ACCESS_KEY"
    )
    
    # Add optional variables if set
    if ($env:REDIS_AUTH_TOKEN) {
        $tfVars += "-var=redis_auth_token=$env:REDIS_AUTH_TOKEN"
    }
    
    if ($env:SSL_CERTIFICATE_ARN) {
        $tfVars += "-var=certificate_arn=$env:SSL_CERTIFICATE_ARN"
    }
    
    if ($env:DOMAIN_NAME) {
        $tfVars += "-var=domain_name=$env:DOMAIN_NAME"
    }
    
    if ($Destroy) {
        Write-Warning "Planning infrastructure DESTRUCTION..."
        terraform plan -destroy @tfVars -out=destroy-plan
    }
    else {
        terraform plan @tfVars -out=tfplan
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Terraform plan failed"
        exit 1
    }
    
    Write-Success "Infrastructure plan completed"
}

# Apply infrastructure changes
function Invoke-TerraformApply {
    if ($PlanOnly) {
        Write-Info "Plan-only mode enabled. Skipping apply."
        return
    }
    
    if ($Destroy) {
        Write-Warning "This will DESTROY all infrastructure for environment: $Environment"
        Write-Warning "This action cannot be undone!"
        Write-Host ""
        $confirm = Read-Host "Are you absolutely sure you want to destroy the infrastructure? (type 'yes' to confirm)"
        
        if ($confirm -ne "yes") {
            Write-Info "Destruction cancelled"
            exit 0
        }
        
        Write-Info "Destroying infrastructure..."
        terraform apply -auto-approve destroy-plan
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Infrastructure destruction failed"
            exit 1
        }
        
        Write-Success "Infrastructure destroyed successfully"
        return
    }
    
    Write-Info "Applying infrastructure changes..."
    terraform apply -auto-approve tfplan
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Infrastructure deployment failed"
        exit 1
    }
    
    # Save outputs
    terraform output -json > terraform-outputs.json
    
    Write-Success "Infrastructure deployment completed successfully"
}

# Display deployment information
function Show-DeploymentInfo {
    if ($Destroy -or $PlanOnly) {
        return
    }
    
    Write-Info "Deployment completed successfully!"
    Write-Host ""
    Write-Host "=== Deployment Information ==="
    
    if (Test-Path "terraform-outputs.json") {
        $outputs = Get-Content "terraform-outputs.json" | ConvertFrom-Json
        
        Write-Host "Environment: $($outputs.environment.value)"
        Write-Host "AWS Region: $($outputs.aws_region.value)"
        Write-Host "VPC ID: $($outputs.vpc_id.value)"
        Write-Host "ECS Cluster: $($outputs.ecs_cluster_name.value)"
        Write-Host "ALB DNS: $($outputs.alb_dns_name.value)"
        Write-Host "Application URL: $($outputs.application_url.value)"
        Write-Host "MLflow URL: $($outputs.mlflow_url.value)"
        Write-Host "S3 Bucket (Resumes): $($outputs.s3_bucket_resumes.value)"
        Write-Host "S3 Bucket (Models): $($outputs.s3_bucket_models.value)"
        Write-Host "RDS Endpoint: $($outputs.rds_endpoint.value)"
        Write-Host "Redis Endpoint: $($outputs.redis_endpoint.value)"
    }
    
    Write-Host "Deployment Time: $(Get-Date)"
    try {
        $gitCommit = git rev-parse --short HEAD 2>$null
        Write-Host "Git Commit: $gitCommit"
    }
    catch {
        Write-Host "Git Commit: unknown"
    }
    Write-Host "=============================="
    Write-Host ""
    
    Write-Info "Next steps:"
    Write-Host "1. Build and push Docker images: cd ..\..\; .\scripts\deploy.sh images"
    Write-Host "2. Deploy ECS services: .\scripts\deploy.sh services"
    Write-Host "3. Run database migrations: .\scripts\deploy.sh migrations"
    Write-Host "4. Verify deployment: .\scripts\deploy.sh verify"
}

# Cleanup function
function Invoke-Cleanup {
    Write-Info "Cleaning up temporary files..."
    Remove-Item -Path "tfplan", "destroy-plan" -ErrorAction SilentlyContinue
}

# Main function
function Main {
    Write-Info "Starting TalentFlow AI infrastructure deployment..."
    Write-Info "Environment: $Environment"
    Write-Info "AWS Region: $Region"
    
    if ($Destroy) {
        Write-Warning "DESTROY MODE ENABLED"
    }
    elseif ($PlanOnly) {
        Write-Info "PLAN-ONLY MODE ENABLED"
    }
    
    Write-Host ""
    
    try {
        # Run deployment steps
        Test-Prerequisites
        Initialize-Terraform
        Invoke-TerraformPlan
        Invoke-TerraformApply
        Show-DeploymentInfo
        
        if ($Destroy) {
            Write-Success "🗑️  Infrastructure destruction completed!"
        }
        elseif ($PlanOnly) {
            Write-Success "📋 Infrastructure planning completed!"
        }
        else {
            Write-Success "🚀 Infrastructure deployment completed!"
        }
    }
    finally {
        Invoke-Cleanup
    }
}

# Run main function
Main