# TalentFlow AI - ECS Task Definitions and Services

# Backend Task Definition
resource "aws_ecs_task_definition" "backend" {
  family                   = "${local.name_prefix}-backend"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "backend"
      image = "${aws_ecr_repository.backend.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "DATABASE_URL"
          value = "postgresql+asyncpg://${var.db_username}:${var.db_password}@${module.rds.db_endpoint}:${module.rds.db_port}/${var.db_name}"
        },
        {
          name  = "REDIS_URL"
          value = "redis://:${var.redis_auth_token}@${aws_elasticache_replication_group.redis.primary_endpoint_address}:${aws_elasticache_replication_group.redis.port}/0"
        },
        {
          name  = "S3_BUCKET_RESUMES"
          value = module.s3.resumes_bucket_name
        },
        {
          name  = "S3_BUCKET_MODELS"
          value = module.s3.models_bucket_name
        },
        {
          name  = "MLFLOW_TRACKING_URI"
          value = "http://localhost:5000"
        }
      ]
      
      secrets = [
        {
          name      = "SECRET_KEY"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:secret_key::"
        },
        {
          name      = "AWS_ACCESS_KEY_ID"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:aws_access_key_id::"
        },
        {
          name      = "AWS_SECRET_ACCESS_KEY"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:aws_secret_access_key::"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.backend.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
      
      essential = true
    }
  ])
  
  tags = local.tags
}

# Worker Task Definition
resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name_prefix}-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "worker"
      image = "${aws_ecr_repository.ml.repository_url}:latest"
      
      command = ["python", "/app/scripts/worker.py"]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "DATABASE_URL"
          value = "postgresql+asyncpg://${var.db_username}:${var.db_password}@${module.rds.db_endpoint}:${module.rds.db_port}/${var.db_name}"
        },
        {
          name  = "REDIS_URL"
          value = "redis://:${var.redis_auth_token}@${aws_elasticache_replication_group.redis.primary_endpoint_address}:${aws_elasticache_replication_group.redis.port}/0"
        },
        {
          name  = "S3_BUCKET_RESUMES"
          value = module.s3.resumes_bucket_name
        },
        {
          name  = "S3_BUCKET_MODELS"
          value = module.s3.models_bucket_name
        },
        {
          name  = "MLFLOW_TRACKING_URI"
          value = "http://mlflow.${local.name_prefix}.local:5000"
        }
      ]
      
      secrets = [
        {
          name      = "SECRET_KEY"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:secret_key::"
        },
        {
          name      = "AWS_ACCESS_KEY_ID"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:aws_access_key_id::"
        },
        {
          name      = "AWS_SECRET_ACCESS_KEY"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:aws_secret_access_key::"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.worker.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      essential = true
    }
  ])
  
  tags = local.tags
}

# MLflow Task Definition
resource "aws_ecs_task_definition" "mlflow" {
  family                   = "${local.name_prefix}-mlflow"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "mlflow"
      image = "${aws_ecr_repository.ml.repository_url}:latest"
      
      command = [
        "mlflow", "server",
        "--backend-store-uri", "postgresql://${var.db_username}:${var.db_password}@${module.rds.db_endpoint}:${module.rds.db_port}/${var.db_name}",
        "--default-artifact-root", "s3://${module.s3.models_bucket_name}/mlflow-artifacts",
        "--host", "0.0.0.0",
        "--port", "5000"
      ]
      
      portMappings = [
        {
          containerPort = 5000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        }
      ]
      
      secrets = [
        {
          name      = "AWS_ACCESS_KEY_ID"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:aws_access_key_id::"
        },
        {
          name      = "AWS_SECRET_ACCESS_KEY"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:aws_secret_access_key::"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.mlflow.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:5000 || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
      
      essential = true
    }
  ])
  
  tags = local.tags
}

# Backend Service
resource "aws_ecs_service" "backend" {
  name            = "${local.name_prefix}-backend-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = var.environment == "prod" ? 2 : 1
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [module.security_groups.ecs_security_group_id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = 8000
  }
  
  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100
  
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  
  depends_on = [aws_lb_listener.https, aws_lb_listener.http]
  
  tags = local.tags
}

# Worker Service
resource "aws_ecs_service" "worker" {
  name            = "${local.name_prefix}-worker-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.environment == "prod" ? 2 : 1
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [module.security_groups.ecs_security_group_id]
    assign_public_ip = false
  }
  
  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 50
  
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  
  tags = local.tags
}

# MLflow Service
resource "aws_ecs_service" "mlflow" {
  name            = "${local.name_prefix}-mlflow-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.mlflow.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [module.security_groups.ecs_security_group_id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.mlflow.arn
    container_name   = "mlflow"
    container_port   = 5000
  }
  
  service_registries {
    registry_arn = aws_service_discovery_service.mlflow.arn
  }
  
  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100
  
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  
  depends_on = [aws_lb_listener.https, aws_lb_listener.http]
  
  tags = local.tags
}

# Service Discovery for internal communication
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "${local.name_prefix}.local"
  description = "Private DNS namespace for TalentFlow AI services"
  vpc         = aws_vpc.main.id
  
  tags = local.tags
}

resource "aws_service_discovery_service" "mlflow" {
  name = "mlflow"
  
  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    
    dns_records {
      ttl  = 10
      type = "A"
    }
    
    routing_policy = "MULTIVALUE"
  }
  
  health_check_custom_config {
    failure_threshold = 1
  }
  
  tags = local.tags
}

# Auto Scaling for Backend Service
resource "aws_appautoscaling_target" "backend" {
  max_capacity       = var.environment == "prod" ? 10 : 3
  min_capacity       = var.environment == "prod" ? 2 : 1
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.backend.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "backend_cpu" {
  name               = "${local.name_prefix}-backend-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.backend.resource_id
  scalable_dimension = aws_appautoscaling_target.backend.scalable_dimension
  service_namespace  = aws_appautoscaling_target.backend.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 300
  }
}

resource "aws_appautoscaling_policy" "backend_memory" {
  name               = "${local.name_prefix}-backend-memory-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.backend.resource_id
  scalable_dimension = aws_appautoscaling_target.backend.scalable_dimension
  service_namespace  = aws_appautoscaling_target.backend.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    
    target_value       = 80.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 300
  }
}

# Auto Scaling for Worker Service
resource "aws_appautoscaling_target" "worker" {
  max_capacity       = var.environment == "prod" ? 5 : 2
  min_capacity       = var.environment == "prod" ? 2 : 1
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "worker_cpu" {
  name               = "${local.name_prefix}-worker-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.worker.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 300
  }
}