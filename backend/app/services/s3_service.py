"""S3 service for file storage"""

import boto3
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO
from datetime import datetime, timedelta
import os

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.core.exceptions import ValidationException

logger = get_logger(__name__)


class S3Service:
    """Service for S3 file operations"""
    
    def __init__(self):
        """Initialize S3 client"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME
        self.resume_prefix = "resumes/"
    
    async def upload_resume(
        self,
        file_content: BinaryIO,
        candidate_id: str,
        filename: str,
        content_type: str = "application/pdf"
    ) -> str:
        """
        Upload resume file to S3
        
        Args:
            file_content: File content as binary stream
            candidate_id: Candidate UUID
            filename: Original filename
            content_type: MIME type of file
        
        Returns:
            S3 file path
        
        Raises:
            ValidationException: If upload fails
        """
        try:
            # Generate S3 key
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_extension = os.path.splitext(filename)[1]
            s3_key = f"{self.resume_prefix}{candidate_id}/{timestamp}_{filename}"
            
            # Upload to S3 with access controls
            self.s3_client.upload_fileobj(
                file_content,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'ServerSideEncryption': 'AES256',  # Encrypt at rest
                    'ACL': 'private'  # Private access only
                }
            )
            
            logger.info(f"Uploaded resume to S3: {s3_key}")
            return s3_key
        
        except ClientError as e:
            logger.error(f"S3 upload failed: {str(e)}")
            raise ValidationException(f"Failed to upload resume: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {str(e)}")
            raise ValidationException(f"Failed to upload resume: {str(e)}")
    
    async def download_resume(self, s3_key: str) -> bytes:
        """
        Download resume file from S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            File content as bytes
        
        Raises:
            ValidationException: If download fails
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            content = response['Body'].read()
            logger.info(f"Downloaded resume from S3: {s3_key}")
            return content
        
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error(f"Resume not found in S3: {s3_key}")
                raise ValidationException(f"Resume not found: {s3_key}")
            else:
                logger.error(f"S3 download failed: {str(e)}")
                raise ValidationException(f"Failed to download resume: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during S3 download: {str(e)}")
            raise ValidationException(f"Failed to download resume: {str(e)}")
    
    async def delete_resume(self, s3_key: str) -> bool:
        """
        Delete resume file from S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            True if deleted successfully
        
        Raises:
            ValidationException: If deletion fails
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            logger.info(f"Deleted resume from S3: {s3_key}")
            return True
        
        except ClientError as e:
            logger.error(f"S3 deletion failed: {str(e)}")
            raise ValidationException(f"Failed to delete resume: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during S3 deletion: {str(e)}")
            raise ValidationException(f"Failed to delete resume: {str(e)}")
    
    async def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate presigned URL for secure file access
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default 1 hour)
        
        Returns:
            Presigned URL
        
        Raises:
            ValidationException: If URL generation fails
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for: {s3_key}")
            return url
        
        except ClientError as e:
            logger.error(f"Presigned URL generation failed: {str(e)}")
            raise ValidationException(f"Failed to generate download URL: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during presigned URL generation: {str(e)}")
            raise ValidationException(f"Failed to generate download URL: {str(e)}")
    
    async def check_file_exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking file existence: {str(e)}")
                return False
    
    async def list_candidate_resumes(self, candidate_id: str) -> list:
        """
        List all resumes for a candidate
        
        Args:
            candidate_id: Candidate UUID
        
        Returns:
            List of S3 keys for candidate's resumes
        """
        try:
            prefix = f"{self.resume_prefix}{candidate_id}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            keys = [obj['Key'] for obj in response['Contents']]
            logger.info(f"Found {len(keys)} resumes for candidate {candidate_id}")
            return keys
        
        except ClientError as e:
            logger.error(f"Error listing resumes: {str(e)}")
            return []
