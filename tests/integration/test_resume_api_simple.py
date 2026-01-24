"""Simple integration tests for resume API endpoints without database"""

import pytest
from io import BytesIO
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from backend.app.main import app


class TestResumeAPISimple:
    """Simple integration tests for resume API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client"""
        return TestClient(app)
    
    def test_upload_resume_invalid_format(self, client: TestClient):
        """Test resume upload with invalid file format"""
        files = {"file": ("test.txt", BytesIO(b"Invalid content"), "text/plain")}
        
        response = client.post("/api/v1/resumes/upload", files=files)
        
        # Should return 403 because no authentication token
        assert response.status_code == 403
    
    def test_upload_resume_no_auth(self, client: TestClient):
        """Test resume upload without authentication"""
        files = {"file": ("test_resume.pdf", BytesIO(b"PDF content"), "application/pdf")}
        
        response = client.post("/api/v1/resumes/upload", files=files)
        
        assert response.status_code == 403
        assert "detail" in response.json()
    
    def test_get_resume_details_no_auth(self, client: TestClient):
        """Test resume details retrieval without authentication"""
        fake_id = "123e4567-e89b-12d3-a456-426614174000"
        response = client.get(f"/api/v1/resumes/{fake_id}")
        
        assert response.status_code == 403
    
    def test_list_resumes_no_auth(self, client: TestClient):
        """Test resume listing without authentication"""
        response = client.get("/api/v1/resumes")
        
        assert response.status_code == 403
    
    def test_delete_resume_no_auth(self, client: TestClient):
        """Test resume deletion without authentication"""
        fake_id = "123e4567-e89b-12d3-a456-426614174000"
        response = client.delete(f"/api/v1/resumes/{fake_id}")
        
        assert response.status_code == 403
    
    def test_download_resume_no_auth(self, client: TestClient):
        """Test resume download without authentication"""
        fake_id = "123e4567-e89b-12d3-a456-426614174000"
        response = client.get(f"/api/v1/resumes/{fake_id}/download")
        
        assert response.status_code == 403
    
    def test_get_resume_url_no_auth(self, client: TestClient):
        """Test presigned URL generation without authentication"""
        fake_id = "123e4567-e89b-12d3-a456-426614174000"
        response = client.get(f"/api/v1/resumes/{fake_id}/url")
        
        assert response.status_code == 403
    
    def test_api_documentation_accessible(self, client: TestClient):
        """Test that API documentation is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_health_check(self, client: TestClient):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        result = response.json()
        assert "name" in result
        assert "version" in result
        assert "status" in result
        assert result["status"] == "running"