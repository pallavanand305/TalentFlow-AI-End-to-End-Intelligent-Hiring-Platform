# Staging Environment Configuration for TalentFlow AI

# Basic Configuration
aws_region   = "us-east-1"
environment  = "staging"
project_name = "talentflow-ai"

# Network Configuration
vpc_cidr = "10.1.0.0/16"

# Database Configuration
db_name     = "talentflow_staging"
db_username = "postgres"
# db_password will be provided via environment variable or command line

# Application Configuration
# secret_key will be provided via environment variable or command line

# AWS Credentials for Application
# aws_access_key_id and aws_secret_access_key will be provided via environment variables

# Domain and SSL Configuration
domain_name     = "staging-api.talentflow.com"
# certificate_arn will be provided via environment variable

# RDS Configuration
instance_class              = "db.t3.micro"
allocated_storage          = 20
max_allocated_storage      = 100
backup_retention_period    = 7
deletion_protection        = false

# Redis Configuration
redis_node_type = "cache.t3.micro"
# redis_auth_token will be provided via environment variable

# Tags
common_tags = {
  Project     = "TalentFlow-AI"
  Environment = "Staging"
  Owner       = "DevOps Team"
  ManagedBy   = "Terraform"
  CostCenter  = "Engineering"
}