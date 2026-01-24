# Production Environment Configuration for TalentFlow AI

# Basic Configuration
aws_region   = "us-east-1"
environment  = "prod"
project_name = "talentflow-ai"

# Network Configuration
vpc_cidr = "10.0.0.0/16"

# Database Configuration
db_name     = "talentflow"
db_username = "postgres"
# db_password will be provided via environment variable or command line

# Application Configuration
# secret_key will be provided via environment variable or command line

# AWS Credentials for Application
# aws_access_key_id and aws_secret_access_key will be provided via environment variables

# Domain and SSL Configuration
# domain_name and certificate_arn will be provided via environment variables

# RDS Configuration
instance_class              = "db.t3.small"
allocated_storage          = 50
max_allocated_storage      = 200
backup_retention_period    = 14
deletion_protection        = true

# Redis Configuration
redis_node_type = "cache.t3.small"
# redis_auth_token will be provided via environment variable

# Tags
common_tags = {
  Project     = "TalentFlow-AI"
  Environment = "Production"
  Owner       = "DevOps Team"
  ManagedBy   = "Terraform"
  CostCenter  = "Engineering"
}