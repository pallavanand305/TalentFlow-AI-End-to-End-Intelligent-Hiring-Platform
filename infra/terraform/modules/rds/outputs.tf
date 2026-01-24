output "db_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
}

output "db_port" {
  description = "RDS instance port"
  value       = aws_db_instance.main.port
}

output "db_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "db_username" {
  description = "Database username"
  value       = aws_db_instance.main.username
  sensitive   = true
}

output "db_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.main.id
}