# TalentFlow AI - Main Terraform Configuration
# This file defines the complete AWS infrastructure for TalentFlow AI

terraform {
  required_version = ">= 1.6.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
  
  backend "s3" {
    bucket         = "talentflow-ai-terraform-state"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "talentflow-ai-terraform-locks"
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = var.common_tags
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# Local values
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  # Merge common tags with environment-specific tags
  tags = merge(var.common_tags, {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "Terraform"
  })
  
  # AZ selection - use first 2 available AZs
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 2)
}

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(local.tags, {
    Name = "${local.name_prefix}-vpc"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = merge(local.tags, {
    Name = "${local.name_prefix}-igw"
  })
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(local.availability_zones)
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = local.availability_zones[count.index]
  map_public_ip_on_launch = true
  
  tags = merge(local.tags, {
    Name = "${local.name_prefix}-public-subnet-${count.index + 1}"
    Type = "Public"
  })
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(local.availability_zones)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = local.availability_zones[count.index]
  
  tags = merge(local.tags, {
    Name = "${local.name_prefix}-private-subnet-${count.index + 1}"
    Type = "Private"
  })
}

# NAT Gateway Elastic IPs
resource "aws_eip" "nat" {
  count = length(local.availability_zones)
  
  domain = "vpc"
  
  tags = merge(local.tags, {
    Name = "${local.name_prefix}-nat-eip-${count.index + 1}"
  })
  
  depends_on = [aws_internet_gateway.main]
}

# NAT Gateways
resource "aws_nat_gateway" "main" {
  count = length(local.availability_zones)
  
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = merge(local.tags, {
    Name = "${local.name_prefix}-nat-gateway-${count.index + 1}"
  })
  
  depends_on = [aws_internet_gateway.main]
}

# Public Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = merge(local.tags, {
    Name = "${local.name_prefix}-public-rt"
  })
}

# Private Route Tables
resource "aws_route_table" "private" {
  count = length(local.availability_zones)
  
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }
  
  tags = merge(local.tags, {
    Name = "${local.name_prefix}-private-rt-${count.index + 1}"
  })
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Groups Module
module "security_groups" {
  source = "./modules/security_groups"
  
  name_prefix = local.name_prefix
  vpc_id      = aws_vpc.main.id
  tags        = local.tags
}

# S3 Buckets Module
module "s3" {
  source = "./modules/s3"
  
  name_prefix = local.name_prefix
  tags        = local.tags
}

# RDS Module
module "rds" {
  source = "./modules/rds"
  
  name_prefix            = local.name_prefix
  private_subnet_ids     = aws_subnet.private[*].id
  security_group_ids     = [module.security_groups.rds_security_group_id]
  db_name                = var.db_name
  db_username            = var.db_username
  db_password            = var.db_password
  instance_class         = var.instance_class
  allocated_storage      = var.allocated_storage
  max_allocated_storage  = var.max_allocated_storage
  backup_retention_period = var.backup_retention_period
  deletion_protection    = var.deletion_protection
  tags                   = local.tags
}

# ElastiCache Redis Cluster
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${local.name_prefix}-redis-subnet-group"
  subnet_ids = aws_subnet.private[*].id
  
  tags = local.tags
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${local.name_prefix}-redis"
  description                = "Redis cluster for TalentFlow AI"
  
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.redis.name
  security_group_ids = [module.security_groups.redis_security_group_id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token
  
  tags = local.tags
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [module.security_groups.alb_security_group_id]
  subnets            = aws_subnet.public[*].id
  
  enable_deletion_protection = var.environment == "prod" ? true : false
  
  access_logs {
    bucket  = module.s3.backups_bucket_name
    prefix  = "alb-logs"
    enabled = true
  }
  
  tags = local.tags
}

# ALB Target Groups
resource "aws_lb_target_group" "backend" {
  name     = "${local.name_prefix}-backend-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }
  
  tags = local.tags
}

resource "aws_lb_target_group" "mlflow" {
  name     = "${local.name_prefix}-mlflow-tg"
  port     = 5000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }
  
  tags = local.tags
}

# ALB Listeners
resource "aws_lb_listener" "https" {
  count = var.certificate_arn != "" ? 1 : 0
  
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.certificate_arn
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.backend.arn
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {
    type = "redirect"
    
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# ALB Listener Rules
resource "aws_lb_listener_rule" "mlflow" {
  count = var.certificate_arn != "" ? 1 : 0
  
  listener_arn = aws_lb_listener.https[0].arn
  priority     = 100
  
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }
  
  condition {
    path_pattern {
      values = ["/mlflow/*"]
    }
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"
  
  configuration {
    execute_command_configuration {
      logging = "OVERRIDE"
      
      log_configuration {
        cloud_watch_encryption_enabled = true
        cloud_watch_log_group_name     = aws_cloudwatch_log_group.ecs.name
      }
    }
  }
  
  tags = local.tags
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name
  
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  
  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = "FARGATE"
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/aws/ecs/${local.name_prefix}"
  retention_in_days = 7
  
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "backend" {
  name              = "/aws/ecs/${local.name_prefix}/backend"
  retention_in_days = 7
  
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/aws/ecs/${local.name_prefix}/worker"
  retention_in_days = 7
  
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/aws/ecs/${local.name_prefix}/mlflow"
  retention_in_days = 7
  
  tags = local.tags
}

# ECR Repositories
resource "aws_ecr_repository" "backend" {
  name                 = "${local.name_prefix}-backend"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = local.tags
}

resource "aws_ecr_repository" "ml" {
  name                 = "${local.name_prefix}-ml"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = local.tags
}

# ECR Lifecycle Policies
resource "aws_ecr_lifecycle_policy" "backend" {
  repository = aws_ecr_repository.backend.name
  
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images older than 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "ml" {
  repository = aws_ecr_repository.ml.name
  
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images older than 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# IAM Roles and Policies
resource "aws_iam_role" "ecs_task_execution" {
  name = "${local.name_prefix}-ecs-task-execution-role"
  
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
  
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name = "${local.name_prefix}-ecs-task-role"
  
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
  
  tags = local.tags
}

# IAM Policy for S3 Access
resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "${local.name_prefix}-ecs-task-s3-policy"
  role = aws_iam_role.ecs_task.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          module.s3.resumes_bucket_arn,
          "${module.s3.resumes_bucket_arn}/*",
          module.s3.models_bucket_arn,
          "${module.s3.models_bucket_arn}/*"
        ]
      }
    ]
  })
}

# Secrets Manager for sensitive configuration
resource "aws_secretsmanager_secret" "app_secrets" {
  name        = "${local.name_prefix}-app-secrets"
  description = "Application secrets for TalentFlow AI"
  
  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  
  secret_string = jsonencode({
    db_password            = var.db_password
    secret_key            = var.secret_key
    aws_access_key_id     = var.aws_access_key_id
    aws_secret_access_key = var.aws_secret_access_key
    redis_auth_token      = var.redis_auth_token
  })
}

# IAM Policy for Secrets Manager Access
resource "aws_iam_role_policy" "ecs_task_secrets" {
  name = "${local.name_prefix}-ecs-task-secrets-policy"
  role = aws_iam_role.ecs_task.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.app_secrets.arn
      }
    ]
  })
}