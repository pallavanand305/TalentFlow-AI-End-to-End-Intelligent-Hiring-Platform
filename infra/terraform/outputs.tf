# TalentFlow AI - Terraform Outputs

# VPC and Networking
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

# Load Balancer
output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.main.zone_id
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.main.arn
}

# Target Groups
output "backend_target_group_arn" {
  description = "ARN of the backend target group"
  value       = aws_lb_target_group.backend.arn
}

output "mlflow_target_group_arn" {
  description = "ARN of the MLflow target group"
  value       = aws_lb_target_group.mlflow.arn
}

# ECS
output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

# ECR
output "ecr_backend_url" {
  description = "URL of the backend ECR repository"
  value       = aws_ecr_repository.backend.repository_url
}

output "ecr_ml_url" {
  description = "URL of the ML ECR repository"
  value       = aws_ecr_repository.ml.repository_url
}

# IAM
output "ecs_task_execution_role_arn" {
  description = "ARN of the ECS task execution role"
  value       = aws_iam_role.ecs_task_execution.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task.arn
}

# CloudWatch
output "cloudwatch_log_group_backend" {
  description = "Name of the backend CloudWatch log group"
  value       = aws_cloudwatch_log_group.backend.name
}

output "cloudwatch_log_group_worker" {
  description = "Name of the worker CloudWatch log group"
  value       = aws_cloudwatch_log_group.worker.name
}

output "cloudwatch_log_group_mlflow" {
  description = "Name of the MLflow CloudWatch log group"
  value       = aws_cloudwatch_log_group.mlflow.name
}

# S3 Buckets (from module)
output "s3_bucket_resumes" {
  description = "Name of the resumes S3 bucket"
  value       = module.s3.resumes_bucket_name
}

output "s3_bucket_models" {
  description = "Name of the models S3 bucket"
  value       = module.s3.models_bucket_name
}

output "s3_bucket_backups" {
  description = "Name of the backups S3 bucket"
  value       = module.s3.backups_bucket_name
}

# RDS (from module)
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_endpoint
}

output "rds_port" {
  description = "RDS instance port"
  value       = module.rds.db_port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.db_name
}

# ElastiCache Redis
output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "redis_port" {
  description = "Redis cluster port"
  value       = aws_elasticache_replication_group.redis.port
}

# Security Groups (from module)
output "alb_security_group_id" {
  description = "ID of the ALB security group"
  value       = module.security_groups.alb_security_group_id
}

output "ecs_security_group_id" {
  description = "ID of the ECS security group"
  value       = module.security_groups.ecs_security_group_id
}

output "rds_security_group_id" {
  description = "ID of the RDS security group"
  value       = module.security_groups.rds_security_group_id
}

output "redis_security_group_id" {
  description = "ID of the Redis security group"
  value       = module.security_groups.redis_security_group_id
}

# Secrets Manager
output "secrets_manager_arn" {
  description = "ARN of the Secrets Manager secret"
  value       = aws_secretsmanager_secret.app_secrets.arn
}

# Application URLs
output "application_url" {
  description = "Application URL"
  value       = var.domain_name != "" ? "https://${var.domain_name}" : "https://${aws_lb.main.dns_name}"
}

output "mlflow_url" {
  description = "MLflow URL"
  value       = var.domain_name != "" ? "https://${var.domain_name}/mlflow" : "https://${aws_lb.main.dns_name}/mlflow"
}

# Environment Information
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "deployment_info" {
  description = "Deployment information"
  value = {
    environment    = var.environment
    region        = var.aws_region
    vpc_id        = aws_vpc.main.id
    cluster_name  = aws_ecs_cluster.main.name
    alb_dns       = aws_lb.main.dns_name
    s3_resumes    = module.s3.resumes_bucket_name
    s3_models     = module.s3.models_bucket_name
    rds_endpoint  = module.rds.db_endpoint
    redis_endpoint = aws_elasticache_replication_group.redis.primary_endpoint_address
  }
}