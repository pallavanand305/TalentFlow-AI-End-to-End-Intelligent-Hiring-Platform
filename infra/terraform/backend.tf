# TalentFlow AI - Terraform Backend Configuration
# This file sets up the S3 backend for Terraform state management

# S3 Bucket for Terraform State
resource "aws_s3_bucket" "terraform_state" {
  bucket = "talentflow-ai-terraform-state-${random_id.bucket_suffix.hex}"
  
  tags = merge(local.tags, {
    Name    = "TalentFlow AI Terraform State"
    Purpose = "Terraform state storage"
  })
  
  lifecycle {
    prevent_destroy = true
  }
}

# Random ID for bucket suffix to ensure uniqueness
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Server-side Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# S3 Bucket Public Access Block
resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# DynamoDB Table for Terraform State Locking
resource "aws_dynamodb_table" "terraform_locks" {
  name           = "talentflow-ai-terraform-locks"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "LockID"
  
  attribute {
    name = "LockID"
    type = "S"
  }
  
  tags = merge(local.tags, {
    Name    = "TalentFlow AI Terraform Locks"
    Purpose = "Terraform state locking"
  })
  
  lifecycle {
    prevent_destroy = true
  }
}

# Output the backend configuration
output "terraform_backend_config" {
  description = "Terraform backend configuration"
  value = {
    bucket         = aws_s3_bucket.terraform_state.bucket
    key            = "terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    dynamodb_table = aws_dynamodb_table.terraform_locks.name
  }
}