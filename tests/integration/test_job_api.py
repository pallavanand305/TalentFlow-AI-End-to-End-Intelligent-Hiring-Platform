"""Integration tests for job API endpoints"""

import pytest
from uuid import uuid4
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.main import app
from backend.app.core.database import get_db
from backend.app.models.user import User, UserRole
from backend.app.models.job import Job, JobStatus, ExperienceLevel
from backend.app.services.auth_service import AuthService
from tests.conftest import test_db_session


class TestJobAPI:
    """Integration tests for job API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client with database override"""
        app.dependency_overrides[get_db] = lambda: test_db_session
        return TestClient(app)
    
    @pytest.fixture
    async def hiring_manager(self, test_db_session: AsyncSession):
        """Create test hiring manager user"""
        user = User(
            id=uuid4(),
            username="test_hiring_manager",
            email="hm@test.com",
            password_hash=AuthService.hash_password("testpass123"),
            role=UserRole.HIRING_MANAGER
        )
        test_db_session.add(user)
        await test_db_session.commit()
        return user
    
    @pytest.fixture
    async def admin_user(self, test_db_session: AsyncSession):
        """Create test admin user"""
        user = User(
            id=uuid4(),
            username="test_admin",
            email="admin@test.com",
            password_hash=AuthService.hash_password("adminpass123"),
            role=UserRole.ADMIN
        )
        test_db_session.add(user)
        await test_db_session.commit()
        return user
    
    @pytest.fixture
    async def recruiter_user(self, test_db_session: AsyncSession):
        """Create test recruiter user"""
        user = User(
            id=uuid4(),
            username="test_recruiter",
            email="recruiter@test.com",
            password_hash=AuthService.hash_password("testpass123"),
            role=UserRole.RECRUITER
        )
        test_db_session.add(user)
        await test_db_session.commit()
        return user
    
    @pytest.fixture
    def hm_headers(self, client: TestClient, hiring_manager: User):
        """Get authentication headers for hiring manager"""
        response = client.post("/api/v1/auth/login", json={
            "username": hiring_manager.username,
            "password": "testpass123"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    def admin_headers(self, client: TestClient, admin_user: User):
        """Get authentication headers for admin user"""
        response = client.post("/api/v1/auth/login", json={
            "username": admin_user.username,
            "password": "adminpass123"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    def recruiter_headers(self, client: TestClient, recruiter_user: User):
        """Get authentication headers for recruiter user"""
        response = client.post("/api/v1/auth/login", json={
            "username": recruiter_user.username,
            "password": "testpass123"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    def sample_job_data(self):
        """Sample job data for testing"""
        return {
            "title": "Senior Python Developer",
            "description": "We are looking for an experienced Python developer to join our team and work on exciting projects.",
            "required_skills": ["Python", "Django", "PostgreSQL", "Docker"],
            "experience_level": "senior",
            "location": "San Francisco, CA",
            "salary_min": 120000,
            "salary_max": 180000
        }
    
    def test_create_job_success(self, client: TestClient, hm_headers: dict, sample_job_data: dict):
        """Test successful job creation"""
        response = client.post(
            "/api/v1/jobs",
            json=sample_job_data,
            headers=hm_headers
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["title"] == sample_job_data["title"]
        assert result["description"] == sample_job_data["description"]
        assert result["required_skills"] == sample_job_data["required_skills"]
        assert result["experience_level"] == sample_job_data["experience_level"]
        assert result["location"] == sample_job_data["location"]
        assert result["salary_min"] == sample_job_data["salary_min"]
        assert result["salary_max"] == sample_job_data["salary_max"]
        assert result["status"] == "active"
        assert "id" in result
        assert "created_at" in result
        assert "updated_at" in result
    
    def test_create_job_invalid_data(self, client: TestClient, hm_headers: dict):
        """Test job creation with invalid data"""
        invalid_data = {
            "title": "AB",  # Too short
            "description": "Short",  # Too short
            "required_skills": [],  # Empty
            "experience_level": "senior"
        }
        
        response = client.post(
            "/api/v1/jobs",
            json=invalid_data,
            headers=hm_headers
        )
        
        assert response.status_code == 422
    
    def test_create_job_unauthorized(self, client: TestClient, sample_job_data: dict):
        """Test job creation without authentication"""
        response = client.post("/api/v1/jobs", json=sample_job_data)
        assert response.status_code == 403  # 403 because middleware blocks unauthenticated requests
    
    def test_create_job_insufficient_role(self, client: TestClient, recruiter_headers: dict, sample_job_data: dict):
        """Test job creation with insufficient role"""
        response = client.post(
            "/api/v1/jobs",
            json=sample_job_data,
            headers=recruiter_headers
        )
        assert response.status_code == 403
    
    async def test_get_job_success(
        self, 
        client: TestClient, 
        hm_headers: dict, 
        test_db_session: AsyncSession,
        hiring_manager: User
    ):
        """Test successful job retrieval"""
        # Create test job
        job = Job(
            id=uuid4(),
            title="Test Job",
            description="This is a test job description for testing purposes.",
            required_skills=["Python", "FastAPI"],
            experience_level=ExperienceLevel.MID,
            location="Remote",
            salary_min=80000,
            salary_max=120000,
            status=JobStatus.ACTIVE,
            created_by=hiring_manager.id
        )
        test_db_session.add(job)
        await test_db_session.commit()
        
        response = client.get(f"/api/v1/jobs/{job.id}", headers=hm_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == str(job.id)
        assert result["title"] == job.title
        assert result["description"] == job.description
        assert result["required_skills"] == job.required_skills
        assert result["experience_level"] == job.experience_level.value
        assert result["location"] == job.location
        assert result["salary_min"] == job.salary_min
        assert result["salary_max"] == job.salary_max
        assert result["status"] == job.status.value
    
    def test_get_job_not_found(self, client: TestClient, hm_headers: dict):
        """Test job retrieval for non-existent job"""
        fake_id = uuid4()
        response = client.get(f"/api/v1/jobs/{fake_id}", headers=hm_headers)
        assert response.status_code == 404
    
    def test_get_job_unauthorized(self, client: TestClient):
        """Test job retrieval without authentication"""
        fake_id = uuid4()
        response = client.get(f"/api/v1/jobs/{fake_id}")
        assert response.status_code == 403  # 403 because middleware blocks unauthenticated requests
    
    async def test_update_job_success(
        self, 
        client: TestClient, 
        hm_headers: dict, 
        test_db_session: AsyncSession,
        hiring_manager: User
    ):
        """Test successful job update"""
        # Create test job
        job = Job(
            id=uuid4(),
            title="Original Title",
            description="Original description for testing purposes.",
            required_skills=["Python"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.ACTIVE,
            created_by=hiring_manager.id
        )
        test_db_session.add(job)
        await test_db_session.commit()
        
        update_data = {
            "title": "Updated Title",
            "required_skills": ["Python", "FastAPI", "PostgreSQL"]
        }
        
        response = client.put(
            f"/api/v1/jobs/{job.id}",
            json=update_data,
            headers=hm_headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["title"] == update_data["title"]
        assert result["required_skills"] == update_data["required_skills"]
        assert result["description"] == job.description  # Unchanged
    
    def test_update_job_not_found(self, client: TestClient, hm_headers: dict):
        """Test job update for non-existent job"""
        fake_id = uuid4()
        update_data = {"title": "Updated Title"}
        
        response = client.put(
            f"/api/v1/jobs/{fake_id}",
            json=update_data,
            headers=hm_headers
        )
        assert response.status_code == 404
    
    def test_update_job_unauthorized(self, client: TestClient):
        """Test job update without authentication"""
        fake_id = uuid4()
        update_data = {"title": "Updated Title"}
        
        response = client.put(f"/api/v1/jobs/{fake_id}", json=update_data)
        assert response.status_code == 403  # 403 because middleware blocks unauthenticated requests
    
    async def test_delete_job_success(
        self, 
        client: TestClient, 
        hm_headers: dict, 
        test_db_session: AsyncSession,
        hiring_manager: User
    ):
        """Test successful job deletion"""
        # Create test job
        job = Job(
            id=uuid4(),
            title="Job to Delete",
            description="This job will be deleted for testing purposes.",
            required_skills=["Python"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.ACTIVE,
            created_by=hiring_manager.id
        )
        test_db_session.add(job)
        await test_db_session.commit()
        
        response = client.delete(f"/api/v1/jobs/{job.id}", headers=hm_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "deleted successfully" in result["message"]
    
    def test_delete_job_not_found(self, client: TestClient, hm_headers: dict):
        """Test job deletion for non-existent job"""
        fake_id = uuid4()
        response = client.delete(f"/api/v1/jobs/{fake_id}", headers=hm_headers)
        assert response.status_code == 404
    
    def test_delete_job_unauthorized(self, client: TestClient):
        """Test job deletion without authentication"""
        fake_id = uuid4()
        response = client.delete(f"/api/v1/jobs/{fake_id}")
        assert response.status_code == 403  # 403 because middleware blocks unauthenticated requests
    
    async def test_list_jobs_success(
        self, 
        client: TestClient, 
        hm_headers: dict, 
        test_db_session: AsyncSession,
        hiring_manager: User
    ):
        """Test successful job listing"""
        # Create test jobs
        jobs = []
        for i in range(3):
            job = Job(
                id=uuid4(),
                title=f"Test Job {i}",
                description=f"This is test job {i} description for testing purposes.",
                required_skills=["Python", "FastAPI"] if i % 2 == 0 else ["JavaScript", "React"],
                experience_level=ExperienceLevel.MID,
                status=JobStatus.ACTIVE,
                created_by=hiring_manager.id
            )
            jobs.append(job)
            test_db_session.add(job)
        
        await test_db_session.commit()
        
        response = client.get("/api/v1/jobs", headers=hm_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert "jobs" in result
        assert "total" in result
        assert "skip" in result
        assert "limit" in result
        assert "has_more" in result
        assert len(result["jobs"]) >= 3
    
    def test_list_jobs_with_filters(self, client: TestClient, hm_headers: dict):
        """Test job listing with filters"""
        response = client.get(
            "/api/v1/jobs?query=python&experience_level=senior&limit=10",
            headers=hm_headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "jobs" in result
        assert isinstance(result["jobs"], list)
    
    def test_list_jobs_unauthorized(self, client: TestClient):
        """Test job listing without authentication"""
        response = client.get("/api/v1/jobs")
        assert response.status_code == 403  # 403 because middleware blocks unauthenticated requests
    
    async def test_close_job_success(
        self, 
        client: TestClient, 
        hm_headers: dict, 
        test_db_session: AsyncSession,
        hiring_manager: User
    ):
        """Test successful job closure"""
        # Create test job
        job = Job(
            id=uuid4(),
            title="Job to Close",
            description="This job will be closed for testing purposes.",
            required_skills=["Python"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.ACTIVE,
            created_by=hiring_manager.id
        )
        test_db_session.add(job)
        await test_db_session.commit()
        
        response = client.post(f"/api/v1/jobs/{job.id}/close", headers=hm_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "closed successfully" in result["message"]
        assert result["job"]["status"] == "closed"
    
    async def test_reopen_job_success(
        self, 
        client: TestClient, 
        hm_headers: dict, 
        test_db_session: AsyncSession,
        hiring_manager: User
    ):
        """Test successful job reopening"""
        # Create test job (closed)
        job = Job(
            id=uuid4(),
            title="Job to Reopen",
            description="This job will be reopened for testing purposes.",
            required_skills=["Python"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.CLOSED,
            created_by=hiring_manager.id
        )
        test_db_session.add(job)
        await test_db_session.commit()
        
        response = client.post(f"/api/v1/jobs/{job.id}/reopen", headers=hm_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "reopened successfully" in result["message"]
        assert result["job"]["status"] == "active"
    
    def test_get_job_stats(self, client: TestClient, hm_headers: dict):
        """Test job statistics retrieval"""
        response = client.get("/api/v1/jobs/stats/summary", headers=hm_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert "active_jobs_total" in result
        assert "user_jobs_count" in result
        assert "user_recent_jobs" in result
        assert isinstance(result["active_jobs_total"], int)
        assert isinstance(result["user_jobs_count"], int)
        assert isinstance(result["user_recent_jobs"], list)


class TestJobAPIWorkflow:
    """Test complete job workflow"""
    
    @pytest.fixture
    def client(self):
        """Test client with database override"""
        app.dependency_overrides[get_db] = lambda: test_db_session
        return TestClient(app)
    
    @pytest.fixture
    async def hiring_manager(self, test_db_session: AsyncSession):
        """Create test hiring manager user"""
        user = User(
            id=uuid4(),
            username="workflow_hm",
            email="workflow_hm@test.com",
            password_hash=AuthService.hash_password("testpass123"),
            role=UserRole.HIRING_MANAGER
        )
        test_db_session.add(user)
        await test_db_session.commit()
        return user
    
    @pytest.fixture
    def hm_headers(self, client: TestClient, hiring_manager: User):
        """Get authentication headers for hiring manager"""
        response = client.post("/api/v1/auth/login", json={
            "username": hiring_manager.username,
            "password": "testpass123"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_complete_job_lifecycle(self, client: TestClient, hm_headers: dict):
        """Test complete job lifecycle: create → update → close → reopen → delete"""
        # Step 1: Create job
        job_data = {
            "title": "Full Stack Developer",
            "description": "We need a full stack developer with experience in modern web technologies.",
            "required_skills": ["JavaScript", "React", "Node.js", "PostgreSQL"],
            "experience_level": "mid",
            "location": "New York, NY",
            "salary_min": 90000,
            "salary_max": 130000
        }
        
        create_response = client.post(
            "/api/v1/jobs",
            json=job_data,
            headers=hm_headers
        )
        
        assert create_response.status_code == 201
        job = create_response.json()
        job_id = job["id"]
        
        # Step 2: Update job
        update_data = {
            "title": "Senior Full Stack Developer",
            "salary_max": 150000,
            "required_skills": ["JavaScript", "React", "Node.js", "PostgreSQL", "Docker"]
        }
        
        update_response = client.put(
            f"/api/v1/jobs/{job_id}",
            json=update_data,
            headers=hm_headers
        )
        
        assert update_response.status_code == 200
        updated_job = update_response.json()
        assert updated_job["title"] == update_data["title"]
        assert updated_job["salary_max"] == update_data["salary_max"]
        
        # Step 3: Close job
        close_response = client.post(
            f"/api/v1/jobs/{job_id}/close",
            headers=hm_headers
        )
        
        assert close_response.status_code == 200
        assert close_response.json()["job"]["status"] == "closed"
        
        # Step 4: Reopen job
        reopen_response = client.post(
            f"/api/v1/jobs/{job_id}/reopen",
            headers=hm_headers
        )
        
        assert reopen_response.status_code == 200
        assert reopen_response.json()["job"]["status"] == "active"
        
        # Step 5: Delete job
        delete_response = client.delete(
            f"/api/v1/jobs/{job_id}",
            headers=hm_headers
        )
        
        assert delete_response.status_code == 200
        assert delete_response.json()["success"] is True
        
        # Verify job is soft deleted (marked inactive)
        get_response = client.get(f"/api/v1/jobs/{job_id}", headers=hm_headers)
        # Job should still exist but be inactive (or return 404 based on access control)
        assert get_response.status_code in [200, 404]
    
    def test_job_search_and_filter_workflow(self, client: TestClient, hm_headers: dict):
        """Test job search and filtering workflow"""
        # Create multiple jobs with different characteristics
        jobs_data = [
            {
                "title": "Python Backend Developer",
                "description": "Backend development with Python and Django framework.",
                "required_skills": ["Python", "Django", "PostgreSQL"],
                "experience_level": "mid",
                "location": "San Francisco, CA",
                "salary_min": 100000,
                "salary_max": 140000
            },
            {
                "title": "Frontend React Developer",
                "description": "Frontend development with React and modern JavaScript.",
                "required_skills": ["JavaScript", "React", "CSS"],
                "experience_level": "senior",
                "location": "Remote",
                "salary_min": 110000,
                "salary_max": 160000
            },
            {
                "title": "DevOps Engineer",
                "description": "Infrastructure and deployment automation specialist.",
                "required_skills": ["Docker", "Kubernetes", "AWS"],
                "experience_level": "senior",
                "location": "Seattle, WA",
                "salary_min": 120000,
                "salary_max": 180000
            }
        ]
        
        # Create all jobs
        created_jobs = []
        for job_data in jobs_data:
            response = client.post("/api/v1/jobs", json=job_data, headers=hm_headers)
            assert response.status_code == 201
            created_jobs.append(response.json())
        
        # Test 1: List all jobs
        all_jobs_response = client.get("/api/v1/jobs", headers=hm_headers)
        assert all_jobs_response.status_code == 200
        all_jobs = all_jobs_response.json()
        assert len(all_jobs["jobs"]) >= 3
        
        # Test 2: Search by text query
        search_response = client.get(
            "/api/v1/jobs?query=Python",
            headers=hm_headers
        )
        assert search_response.status_code == 200
        # Should find the Python job
        
        # Test 3: Filter by experience level
        senior_response = client.get(
            "/api/v1/jobs?experience_level=senior",
            headers=hm_headers
        )
        assert senior_response.status_code == 200
        # Should find React and DevOps jobs
        
        # Test 4: Filter by skills
        react_response = client.get(
            "/api/v1/jobs?skills=React",
            headers=hm_headers
        )
        assert react_response.status_code == 200
        # Should find React job
        
        # Test 5: Filter by salary range
        high_salary_response = client.get(
            "/api/v1/jobs?salary_min=150000",
            headers=hm_headers
        )
        assert high_salary_response.status_code == 200
        # Should find DevOps job
        
        # Test 6: Combined filters
        combined_response = client.get(
            "/api/v1/jobs?experience_level=senior&salary_min=100000&limit=5",
            headers=hm_headers
        )
        assert combined_response.status_code == 200
        # Should find senior jobs with high salary
    
    def test_error_handling_workflow(self, client: TestClient, hm_headers: dict):
        """Test error handling in job workflow"""
        # Test 1: Create job with invalid data
        invalid_job = {
            "title": "AB",  # Too short
            "description": "Short",  # Too short
            "required_skills": [],  # Empty
            "experience_level": "invalid"  # Invalid enum
        }
        
        response = client.post("/api/v1/jobs", json=invalid_job, headers=hm_headers)
        assert response.status_code == 422
        
        # Test 2: Access non-existent job
        fake_id = uuid4()
        response = client.get(f"/api/v1/jobs/{fake_id}", headers=hm_headers)
        assert response.status_code == 404
        
        # Test 3: Update non-existent job
        response = client.put(
            f"/api/v1/jobs/{fake_id}",
            json={"title": "Updated Title"},
            headers=hm_headers
        )
        assert response.status_code == 404
        
        # Test 4: Delete non-existent job
        response = client.delete(f"/api/v1/jobs/{fake_id}", headers=hm_headers)
        assert response.status_code == 404
        
        # Test 5: Invalid salary range
        invalid_salary_job = {
            "title": "Test Job",
            "description": "This is a test job with invalid salary range.",
            "required_skills": ["Python"],
            "experience_level": "mid",
            "salary_min": 150000,
            "salary_max": 100000  # Max less than min
        }
        
        response = client.post("/api/v1/jobs", json=invalid_salary_job, headers=hm_headers)
        assert response.status_code == 422