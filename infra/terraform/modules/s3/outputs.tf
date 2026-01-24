output "resumes_bucket_name" {
  description = "Name of the resumes S3 bucket"
  value       = aws_s3_bucket.resumes.bucket
}

output "resumes_bucket_arn" {
  description = "ARN of the resumes S3 bucket"
  value       = aws_s3_bucket.resumes.arn
}

output "models_bucket_name" {
  description = "Name of the models S3 bucket"
  value       = aws_s3_bucket.models.bucket
}

output "models_bucket_arn" {
  description = "ARN of the models S3 bucket"
  value       = aws_s3_bucket.models.arn
}

output "backups_bucket_name" {
  description = "Name of the backups S3 bucket"
  value       = aws_s3_bucket.backups.bucket
}

output "backups_bucket_arn" {
  description = "ARN of the backups S3 bucket"
  value       = aws_s3_bucket.backups.arn
}