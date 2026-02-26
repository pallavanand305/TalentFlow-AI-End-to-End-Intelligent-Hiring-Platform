# S3 Bucket for Resume Files
resource "aws_s3_bucket" "resumes" {
  bucket = "${var.name_prefix}-resumes"

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-resumes"
    Purpose     = "Resume file storage"
    Environment = var.tags.Environment
  })
}

# S3 Bucket for ML Model Artifacts
resource "aws_s3_bucket" "models" {
  bucket = "${var.name_prefix}-models"

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-models"
    Purpose     = "ML model artifact storage"
    Environment = var.tags.Environment
  })
}

# S3 Bucket for Backups
resource "aws_s3_bucket" "backups" {
  bucket = "${var.name_prefix}-backups"

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-backups"
    Purpose     = "Database and application backups"
    Environment = var.tags.Environment
  })
}

# Versioning Configuration
resource "aws_s3_bucket_versioning" "resumes" {
  bucket = aws_s3_bucket.resumes.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "resumes" {
  bucket = aws_s3_bucket.resumes.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Public Access Block
resource "aws_s3_bucket_public_access_block" "resumes" {
  bucket = aws_s3_bucket.resumes.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "backups" {
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle Configuration
resource "aws_s3_bucket_lifecycle_configuration" "resumes" {
  bucket = aws_s3_bucket.resumes.id

  rule {
    id     = "resume_lifecycle"
    status = "Enabled"

    filter {}

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    id     = "model_lifecycle"
    status = "Enabled"

    filter {}

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = 180
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "backup_lifecycle"
    status = "Enabled"

    filter {}

    transition {
      days          = 7
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    transition {
      days          = 90
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = 2555  # 7 years
    }
  }
}

# CORS Configuration for Resume Bucket
resource "aws_s3_bucket_cors_configuration" "resumes" {
  bucket = aws_s3_bucket.resumes.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST"]
    allowed_origins = ["https://*.talentflow.com"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# Notification Configuration
resource "aws_s3_bucket_notification" "resumes" {
  bucket = aws_s3_bucket.resumes.id

  # Optional: Add SQS/SNS notifications for file uploads
  # This can trigger resume parsing workflows
}

# Bucket Policy for Application Access
resource "aws_s3_bucket_policy" "resumes" {
  bucket = aws_s3_bucket.resumes.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowApplicationAccess"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.name_prefix}-ecs-task-role"
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.resumes.arn}/*"
      },
      {
        Sid    = "AllowApplicationListBucket"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.name_prefix}-ecs-task-role"
        }
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.resumes.arn
      }
    ]
  })
}

resource "aws_s3_bucket_policy" "models" {
  bucket = aws_s3_bucket.models.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowApplicationAccess"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.name_prefix}-ecs-task-role"
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.models.arn}/*"
      },
      {
        Sid    = "AllowApplicationListBucket"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.name_prefix}-ecs-task-role"
        }
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.models.arn
      }
    ]
  })
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}