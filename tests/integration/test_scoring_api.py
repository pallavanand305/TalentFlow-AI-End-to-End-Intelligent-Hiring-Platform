"""Integration tests for scoring API endpoints"""

import pytest
from uuid import uuid4
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.main import app
from backend.app.core.database import get_db
from backend.app.models.user import User, UserRole
from backend.app.models.job import Job, JobStatus, ExperienceLevel
from backend.app.models.candidate import Candidate
from backend.app.models.score import Score
from backend.app.services.auth_service import AuthService
from tests.conftest import test_db_session


class TestScoringAPIIntegration:
    """Comprehensive integration tests for scoring API endpoints"""
    
    @pytest.fixture
    def client(self, test_db_session: AsyncSession):
        """Test client with database override"""
        app.dependency_overrides[get_db] = lambda: test_db_session
        return TestClient(app)
    
    @pytest.fixture
    async def test_user(self, test_db_session: AsyncSession):
        """Create test user"""
        user = User(
            id=uuid4(),
            username="testuser",
            email="test@example.com",
            password_hash=AuthService.hash_password("testpass123"),
            role=UserRole.RECRUITER
        )
        test_db_session.add(user)
        await test_db_session.commit()
        await test_db_session.refresh(user)
        return user
    
    @pytest.fixture
    async def hiring_manager(self, test_db_session: AsyncSession):
        """Create test hiring manager"""
        user = User(
            id=uuid4(),
            username="hiring_manager",
            email="hm@example.com",
            password_hash=AuthService.hash_password("testpass123"),
            role=UserRole.HIRING_MANAGER
        )
        test_db_session.add(user)
        await test_db_session.commit()
        await test_db_session.refresh(user)
        return user
    
    @pytest.fixture
    async def test_job(self, test_db_session: AsyncSession, test_user: User):
        """Create test job"""
        job = Job(
            id=uuid4(),
            title="Software Engineer",
            description="We are looking for a skilled software engineer with experience in Python, FastAPI, and PostgreSQL. The ideal candidate should have strong problem-solving skills and experience with web development.",
            required_skills=["Python", "FastAPI", "PostgreSQL"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.ACTIVE,
            created_by=test_user.id
        )
        test_db_session.add(job)
        await test_db_session.commit()
        await test_db_session.refresh(job)
        return job
    
    @pytest.fixture
    async def test_candidate(self, test_db_session: AsyncSession):
        """Create test candidate with parsed resume data"""
        candidate = Candidate(
            id=uuid4(),
            name="John Doe",
            email="john@example.com",
            resume_file_path="s3://bucket/resume.pdf",
            parsed_data={
                "candidate_name": "John Doe",
                "email": "john@example.com",
                "work_experience": [
                    {
                        "company": "Tech Corp",
                        "title": "Software Developer",
                        "start_date": "2020-01-01",
                        "end_date": "2023-12-31",
                        "description": "Developed web applications using Python and FastAPI. Built RESTful APIs and worked with PostgreSQL databases.",
                        "confidence": 0.9
                    }
                ],
                "skills": ["Python", "FastAPI", "JavaScript", "PostgreSQL", "React"],
                "education": [
                    {
                        "institution": "University of Tech",
                        "degree": "Bachelor of Computer Science",
                        "graduation_date": "2019-05-15",
                        "confidence": 0.95
                    }
                ],
                "certifications": ["AWS Certified Developer"],
                "confidence_scores": {
                    "name": 0.98,
                    "email": 0.95,
                    "skills": 0.92,
                    "experience": 0.90,
                    "education": 0.95
                }
            },
            skills=["Python", "FastAPI", "JavaScript", "PostgreSQL", "React"],
            experience_years=4
        )
        test_db_session.add(candidate)
        await test_db_session.commit()
        await test_db_session.refresh(candidate)
        return candidate
    
    @pytest.fixture
    async def second_candidate(self, test_db_session: AsyncSession):
        """Create second test candidate for ranking tests"""
        candidate = Candidate(
            id=uuid4(),
            name="Jane Smith",
            email="jane@example.com",
            resume_file_path="s3://bucket/resume2.pdf",
            parsed_data={
                "candidate_name": "Jane Smith",
                "email": "jane@example.com",
                "work_experience": [
                    {
                        "company": "Data Corp",
                        "title": "Backend Developer",
                        "start_date": "2019-06-01",
                        "end_date": "2024-01-31",
                        "description": "Specialized in Python backend development and database optimization. Worked extensively with FastAPI and PostgreSQL.",
                        "confidence": 0.88
                    }
                ],
                "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"],
                "education": [
                    {
                        "institution": "State University",
                        "degree": "Master of Computer Science",
                        "graduation_date": "2019-05-20",
                        "confidence": 0.93
                    }
                ],
                "certifications": ["PostgreSQL Certified Professional"],
                "confidence_scores": {
                    "name": 0.97,
                    "email": 0.94,
                    "skills": 0.89,
                    "experience": 0.88,
                    "education": 0.93
                }
            },
            skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"],
            experience_years=5
        )
        test_db_session.add(candidate)
        await test_db_session.commit()
        await test_db_session.refresh(candidate)
        return candidate
    
    @pytest.fixture
    async def third_candidate(self, test_db_session: AsyncSession):
        """Create third test candidate with lower matching score"""
        candidate = Candidate(
            id=uuid4(),
            name="Bob Wilson",
            email="bob@example.com",
            resume_file_path="s3://bucket/resume3.pdf",
            parsed_data={
                "candidate_name": "Bob Wilson",
                "email": "bob@example.com",
                "work_experience": [
                    {
                        "company": "Frontend Solutions",
                        "title": "Frontend Developer",
                        "start_date": "2021-03-01",
                        "end_date": "2024-02-28",
                        "description": "Focused on React and JavaScript development. Some experience with Node.js backends.",
                        "confidence": 0.85
                    }
                ],
                "skills": ["JavaScript", "React", "Node.js", "HTML", "CSS"],
                "education": [
                    {
                        "institution": "Community College",
                        "degree": "Associate in Web Development",
                        "graduation_date": "2020-12-15",
                        "confidence": 0.90
                    }
                ],
                "certifications": ["React Developer Certification"],
                "confidence_scores": {
                    "name": 0.96,
                    "email": 0.93,
                    "skills": 0.87,
                    "experience": 0.85,
                    "education": 0.90
                }
            },
            skills=["JavaScript", "React", "Node.js", "HTML", "CSS"],
            experience_years=3
        )
        test_db_session.add(candidate)
        await test_db_session.commit()
        await test_db_session.refresh(candidate)
        return candidate
    
    @pytest.fixture
    def auth_headers(self, client: TestClient, test_user: User):
        """Create authentication headers"""
        response = client.post("/api/v1/auth/login", json={
            "username": test_user.username,
            "password": "testpass123"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    def hm_headers(self, client: TestClient, hiring_manager: User):
        """Create authentication headers for hiring manager"""
        response = client.post("/api/v1/auth/login", json={
            "username": hiring_manager.username,
            "password": "testpass123"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    
    # Test complete scoring workflow
    def test_complete_scoring_workflow(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_candidate: Candidate, 
        test_job: Job
    ):
        """Test complete end-to-end scoring workflow"""
        # Step 1: Compute score
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": str(test_candidate.id),
                                 "job_id": str(test_job.id),
                                 "generate_explanation": True
                             })
        
        assert response.status_code == 201
        score_data = response.json()
        
        # Verify score response structure
        assert "score_id" in score_data
        assert "candidate_id" in score_data
        assert "job_id" in score_data
        assert "score" in score_data
        assert "explanation" in score_data
        assert "timestamp" in score_data
        assert "model_version" in score_data
        
        # Verify score is in valid range
        assert 0.0 <= score_data["score"] <= 1.0
        assert score_data["candidate_id"] == str(test_candidate.id)
        assert score_data["job_id"] == str(test_job.id)
        assert score_data["explanation"] is not None
        
        score_id = score_data["score_id"]
        
        # Step 2: Retrieve score details
        response = client.get(f"/api/v1/scores/{score_id}", headers=auth_headers)
        assert response.status_code == 200
        
        score_details = response.json()
        assert score_details["score_id"] == score_id
        assert score_details["score"] == score_data["score"]
        assert score_details["explanation"] == score_data["explanation"]
        
        # Step 3: Verify score appears in job candidates ranking
        response = client.get(f"/api/v1/jobs/{test_job.id}/candidates", headers=auth_headers)
        assert response.status_code == 200
        
        candidates = response.json()
        assert len(candidates) >= 1
        
        # Find our candidate in the ranking
        our_candidate = next((c for c in candidates if c["candidate_id"] == str(test_candidate.id)), None)
        assert our_candidate is not None
        assert our_candidate["score"] == score_data["score"]
        assert our_candidate["rank"] >= 1
        
        # Step 4: Verify score appears in top candidates
        response = client.get(f"/api/v1/jobs/{test_job.id}/top-candidates", headers=auth_headers)
        assert response.status_code == 200
        
        top_candidates = response.json()
        assert len(top_candidates) >= 1
        
        # Our candidate should be in top candidates (since it's the only one)
        top_candidate = next((c for c in top_candidates if c["candidate_id"] == str(test_candidate.id)), None)
        assert top_candidate is not None
        assert top_candidate["score"] == score_data["score"]
    
    def test_candidate_ranking_order(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_candidate: Candidate,
        second_candidate: Candidate,
        third_candidate: Candidate
    ):
        """Test that candidates are ranked in descending order by score"""
        # Score all three candidates
        candidates = [test_candidate, second_candidate, third_candidate]
        scores = []
        
        for candidate in candidates:
            response = client.post("/api/v1/scores/compute", 
                                 headers=auth_headers,
                                 json={
                                     "candidate_id": str(candidate.id),
                                     "job_id": str(test_job.id)
                                 })
            assert response.status_code == 201
            scores.append(response.json())
        
        # Get ranked candidates
        response = client.get(f"/api/v1/jobs/{test_job.id}/candidates", headers=auth_headers)
        assert response.status_code == 200
        
        ranked_candidates = response.json()
        assert len(ranked_candidates) == 3
        
        # Verify ranking order (scores should be in descending order)
        for i in range(len(ranked_candidates) - 1):
            current_score = ranked_candidates[i]["score"]
            next_score = ranked_candidates[i + 1]["score"]
            assert current_score >= next_score, f"Ranking order violated: {current_score} < {next_score}"
        
        # Verify rank numbers are sequential
        for i, candidate in enumerate(ranked_candidates):
            assert candidate["rank"] == i + 1
    
    def test_score_filtering_by_minimum_threshold(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_candidate: Candidate,
        second_candidate: Candidate,
        third_candidate: Candidate
    ):
        """Test filtering candidates by minimum score threshold"""
        # Score all candidates first
        candidates = [test_candidate, second_candidate, third_candidate]
        
        for candidate in candidates:
            response = client.post("/api/v1/scores/compute", 
                                 headers=auth_headers,
                                 json={
                                     "candidate_id": str(candidate.id),
                                     "job_id": str(test_job.id)
                                 })
            assert response.status_code == 201
        
        # Get all candidates without filter
        response = client.get(f"/api/v1/jobs/{test_job.id}/candidates", headers=auth_headers)
        assert response.status_code == 200
        all_candidates = response.json()
        
        # Apply minimum score filter (use median score as threshold)
        if len(all_candidates) >= 2:
            median_score = sorted([c["score"] for c in all_candidates])[len(all_candidates) // 2]
            
            response = client.get(
                f"/api/v1/jobs/{test_job.id}/candidates?min_score={median_score}", 
                headers=auth_headers
            )
            assert response.status_code == 200
            filtered_candidates = response.json()
            
            # All returned candidates should meet the threshold
            for candidate in filtered_candidates:
                assert candidate["score"] >= median_score
            
            # Should have fewer candidates than the unfiltered list
            assert len(filtered_candidates) <= len(all_candidates)
    
    def test_top_candidates_limit(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_candidate: Candidate,
        second_candidate: Candidate,
        third_candidate: Candidate
    ):
        """Test top candidates endpoint respects limit parameter"""
        # Score all candidates
        candidates = [test_candidate, second_candidate, third_candidate]
        
        for candidate in candidates:
            response = client.post("/api/v1/scores/compute", 
                                 headers=auth_headers,
                                 json={
                                     "candidate_id": str(candidate.id),
                                     "job_id": str(test_job.id)
                                 })
            assert response.status_code == 201
        
        # Test different limits
        for limit in [1, 2, 3]:
            response = client.get(
                f"/api/v1/jobs/{test_job.id}/top-candidates?limit={limit}", 
                headers=auth_headers
            )
            assert response.status_code == 200
            
            top_candidates = response.json()
            assert len(top_candidates) == min(limit, 3)  # Should not exceed available candidates
            
            # Verify they are the top scoring candidates
            if len(top_candidates) > 1:
                for i in range(len(top_candidates) - 1):
                    assert top_candidates[i]["score"] >= top_candidates[i + 1]["score"]
    
    def test_force_rescore_functionality(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_candidate: Candidate, 
        test_job: Job
    ):
        """Test force rescoring overwrites existing scores"""
        # Initial score
        response1 = client.post("/api/v1/scores/compute", 
                              headers=auth_headers,
                              json={
                                  "candidate_id": str(test_candidate.id),
                                  "job_id": str(test_job.id)
                              })
        assert response1.status_code == 201
        initial_score = response1.json()
        
        # Score again without force_rescore (should return existing score)
        response2 = client.post("/api/v1/scores/compute", 
                              headers=auth_headers,
                              json={
                                  "candidate_id": str(test_candidate.id),
                                  "job_id": str(test_job.id)
                              })
        assert response2.status_code == 201
        second_score = response2.json()
        
        # Should be the same score ID (existing score returned)
        assert second_score["score_id"] == initial_score["score_id"]
        
        # Force rescore
        response3 = client.post("/api/v1/scores/compute", 
                              headers=auth_headers,
                              json={
                                  "candidate_id": str(test_candidate.id),
                                  "job_id": str(test_job.id),
                                  "force_rescore": True
                              })
        assert response3.status_code == 201
        forced_score = response3.json()
        
        # Should be a new score ID (new score created)
        assert forced_score["score_id"] != initial_score["score_id"]
    
    def test_score_explanation_generation(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_candidate: Candidate, 
        test_job: Job
    ):
        """Test score explanation generation"""
        # Score with explanation
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": str(test_candidate.id),
                                 "job_id": str(test_job.id),
                                 "generate_explanation": True
                             })
        
        assert response.status_code == 201
        score_with_explanation = response.json()
        assert score_with_explanation["explanation"] is not None
        assert len(score_with_explanation["explanation"]) > 0
        
        # Score without explanation (using different candidate to avoid cache)
        # Force rescore to ensure we get a new score
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": str(test_candidate.id),
                                 "job_id": str(test_job.id),
                                 "generate_explanation": False,
                                 "force_rescore": True
                             })
        
        assert response.status_code == 201
        score_without_explanation = response.json()
        # Explanation should be None or empty when not requested
        assert score_without_explanation["explanation"] is None or score_without_explanation["explanation"] == ""
    
    
    # Error handling and edge case tests
    def test_authentication_required_for_all_endpoints(self, client: TestClient):
        """Test that all scoring endpoints require authentication"""
        endpoints = [
            ("POST", "/api/v1/scores/compute", {"candidate_id": str(uuid4()), "job_id": str(uuid4())}),
            ("GET", f"/api/v1/scores/{uuid4()}", None),
            ("GET", f"/api/v1/jobs/{uuid4()}/candidates", None),
            ("GET", f"/api/v1/jobs/{uuid4()}/top-candidates", None),
        ]
        
        for method, path, data in endpoints:
            if method == "POST":
                response = client.post(path, json=data)
            else:
                response = client.get(path)
            
            assert response.status_code == 401, f"Endpoint {method} {path} should require authentication"
    
    def test_invalid_candidate_id_returns_404(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job
    ):
        """Test scoring with non-existent candidate returns 404"""
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": str(uuid4()),  # Non-existent candidate
                                 "job_id": str(test_job.id)
                             })
        
        assert response.status_code == 404
        error_detail = response.json()["detail"]
        assert "not found" in error_detail.lower()
    
    def test_invalid_job_id_returns_404(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_candidate: Candidate
    ):
        """Test scoring with non-existent job returns 404"""
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": str(test_candidate.id),
                                 "job_id": str(uuid4())  # Non-existent job
                             })
        
        assert response.status_code == 404
        error_detail = response.json()["detail"]
        assert "not found" in error_detail.lower()
    
    def test_invalid_uuid_format_returns_422(self, client: TestClient, auth_headers: dict):
        """Test that invalid UUID format returns validation error"""
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": "invalid-uuid-format",
                                 "job_id": str(uuid4())
                             })
        
        assert response.status_code == 422
    
    def test_missing_required_fields_returns_422(self, client: TestClient, auth_headers: dict):
        """Test that missing required fields return validation error"""
        # Missing job_id
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": str(uuid4())
                             })
        assert response.status_code == 422
        
        # Missing candidate_id
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "job_id": str(uuid4())
                             })
        assert response.status_code == 422
        
        # Empty request body
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={})
        assert response.status_code == 422
    
    async def test_candidate_without_parsed_data_returns_422(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_db_session: AsyncSession
    ):
        """Test that candidate without parsed resume data returns validation error"""
        # Create candidate without parsed_data
        candidate_no_data = Candidate(
            id=uuid4(),
            name="No Data Candidate",
            email="nodata@example.com",
            resume_file_path="s3://bucket/empty.pdf",
            parsed_data=None,  # No parsed data
            skills=[]
        )
        test_db_session.add(candidate_no_data)
        await test_db_session.commit()
        
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": str(candidate_no_data.id),
                                 "job_id": str(test_job.id)
                             })
        
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert "parsed resume data" in error_detail.lower()
    
    def test_query_parameter_validation(self, client: TestClient, auth_headers: dict, test_job: Job):
        """Test that query parameters are properly validated"""
        # Invalid min_score (> 1.0)
        response = client.get(
            f"/api/v1/jobs/{test_job.id}/candidates?min_score=1.5",
            headers=auth_headers
        )
        assert response.status_code == 422
        
        # Invalid min_score (< 0.0)
        response = client.get(
            f"/api/v1/jobs/{test_job.id}/candidates?min_score=-0.1",
            headers=auth_headers
        )
        assert response.status_code == 422
        
        # Invalid limit (< 1)
        response = client.get(
            f"/api/v1/jobs/{test_job.id}/top-candidates?limit=0",
            headers=auth_headers
        )
        assert response.status_code == 422
        
        # Invalid limit (too large)
        response = client.get(
            f"/api/v1/jobs/{test_job.id}/top-candidates?limit=1001",
            headers=auth_headers
        )
        assert response.status_code == 422
    
    def test_non_existent_score_returns_404(self, client: TestClient, auth_headers: dict):
        """Test that requesting non-existent score returns 404"""
        response = client.get(f"/api/v1/scores/{uuid4()}", headers=auth_headers)
        assert response.status_code == 404
    
    def test_non_existent_job_for_candidates_returns_404(self, client: TestClient, auth_headers: dict):
        """Test that requesting candidates for non-existent job returns 404"""
        response = client.get(f"/api/v1/jobs/{uuid4()}/candidates", headers=auth_headers)
        assert response.status_code == 404
        
        response = client.get(f"/api/v1/jobs/{uuid4()}/top-candidates", headers=auth_headers)
        assert response.status_code == 404
    
    def test_empty_candidate_list_returns_empty_array(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job
    ):
        """Test that job with no scored candidates returns empty array"""
        response = client.get(f"/api/v1/jobs/{test_job.id}/candidates", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []
        
        response = client.get(f"/api/v1/jobs/{test_job.id}/top-candidates", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []
    
    def test_score_bounds_validation(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_candidate: Candidate, 
        test_job: Job
    ):
        """Test that computed scores are always within valid bounds [0.0, 1.0]"""
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": str(test_candidate.id),
                                 "job_id": str(test_job.id)
                             })
        
        assert response.status_code == 201
        score_data = response.json()
        
        # Verify score is within bounds
        score = score_data["score"]
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0
    
    def test_response_schema_consistency(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_candidate: Candidate, 
        test_job: Job
    ):
        """Test that API responses have consistent schema structure"""
        # Test compute score response schema
        response = client.post("/api/v1/scores/compute", 
                             headers=auth_headers,
                             json={
                                 "candidate_id": str(test_candidate.id),
                                 "job_id": str(test_job.id),
                                 "generate_explanation": True
                             })
        
        assert response.status_code == 201
        score_data = response.json()
        
        # Required fields
        required_fields = ["score_id", "candidate_id", "job_id", "score", "timestamp", "model_version"]
        for field in required_fields:
            assert field in score_data, f"Missing required field: {field}"
        
        # Optional fields should be present when requested
        assert "explanation" in score_data
        
        # Test score details response schema
        score_id = score_data["score_id"]
        response = client.get(f"/api/v1/scores/{score_id}", headers=auth_headers)
        assert response.status_code == 200
        
        details_data = response.json()
        for field in required_fields:
            assert field in details_data, f"Missing required field in details: {field}"
        
        # Test candidates ranking response schema
        response = client.get(f"/api/v1/jobs/{test_job.id}/candidates", headers=auth_headers)
        assert response.status_code == 200
        
        candidates = response.json()
        if candidates:  # If there are candidates
            candidate = candidates[0]
            ranking_fields = ["candidate_id", "candidate_name", "score", "rank"]
            for field in ranking_fields:
                assert field in candidate, f"Missing required field in ranking: {field}"


class TestAdvancedScoringScenarios:
    """Test advanced scoring scenarios and edge cases"""
    
    @pytest.fixture
    def client(self, test_db_session: AsyncSession):
        """Test client with database override"""
        app.dependency_overrides[get_db] = lambda: test_db_session
        return TestClient(app)
    
    @pytest.fixture
    async def admin_user(self, test_db_session: AsyncSession):
        """Create admin user for advanced operations"""
        user = User(
            id=uuid4(),
            username="admin",
            email="admin@example.com",
            password_hash=AuthService.hash_password("adminpass123"),
            role=UserRole.ADMIN
        )
        test_db_session.add(user)
        await test_db_session.commit()
        await test_db_session.refresh(user)
        return user
    
    @pytest.fixture
    def admin_headers(self, client: TestClient, admin_user: User):
        """Create authentication headers for admin"""
        response = client.post("/api/v1/auth/login", json={
            "username": admin_user.username,
            "password": "adminpass123"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_batch_scoring_workflow(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_candidate: Candidate,
        second_candidate: Candidate,
        third_candidate: Candidate
    ):
        """Test batch scoring multiple candidates for a job"""
        candidate_ids = [
            str(test_candidate.id),
            str(second_candidate.id),
            str(third_candidate.id)
        ]
        
        # Test batch scoring endpoint
        response = client.post("/api/v1/scores/batch", 
                             headers=auth_headers,
                             json={
                                 "candidate_ids": candidate_ids,
                                 "job_id": str(test_job.id),
                                 "generate_explanations": True
                             })
        
        assert response.status_code == 201
        batch_result = response.json()
        
        # Verify batch response structure
        assert "job_id" in batch_result
        assert "total_candidates" in batch_result
        assert "successful_scores" in batch_result
        assert "failed_scores" in batch_result
        assert "scores" in batch_result
        
        assert batch_result["job_id"] == str(test_job.id)
        assert batch_result["total_candidates"] == 3
        assert batch_result["successful_scores"] >= 0
        assert batch_result["failed_scores"] >= 0
        assert batch_result["successful_scores"] + batch_result["failed_scores"] == 3
        
        # Verify individual scores in batch
        scores = batch_result["scores"]
        assert len(scores) == batch_result["successful_scores"]
        
        for score in scores:
            assert "score_id" in score
            assert "candidate_id" in score
            assert "job_id" in score
            assert "score" in score
            assert "explanation" in score  # Should have explanations
            assert 0.0 <= score["score"] <= 1.0
    
    def test_job_scoring_statistics(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_candidate: Candidate,
        second_candidate: Candidate,
        third_candidate: Candidate
    ):
        """Test job scoring statistics endpoint"""
        # First score some candidates
        candidates = [test_candidate, second_candidate, third_candidate]
        
        for candidate in candidates:
            response = client.post("/api/v1/scores/compute", 
                                 headers=auth_headers,
                                 json={
                                     "candidate_id": str(candidate.id),
                                     "job_id": str(test_job.id)
                                 })
            assert response.status_code == 201
        
        # Get job statistics
        response = client.get(f"/api/v1/scores/jobs/{test_job.id}/stats", headers=auth_headers)
        assert response.status_code == 200
        
        stats = response.json()
        
        # Verify statistics structure
        assert "job_id" in stats
        assert "job_title" in stats
        assert "total_candidates" in stats
        assert "average_score" in stats
        assert "median_score" in stats
        assert "min_score" in stats
        assert "max_score" in stats
        assert "score_distribution" in stats
        assert "top_candidates_count" in stats
        
        # Verify statistics values
        assert stats["job_id"] == str(test_job.id)
        assert stats["job_title"] == test_job.title
        assert stats["total_candidates"] == 3
        assert stats["average_score"] is not None
        assert 0.0 <= stats["average_score"] <= 1.0
        assert 0.0 <= stats["min_score"] <= 1.0
        assert 0.0 <= stats["max_score"] <= 1.0
        assert stats["min_score"] <= stats["average_score"] <= stats["max_score"]
    
    async def test_candidate_best_matches(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_candidate: Candidate,
        test_db_session: AsyncSession
    ):
        """Test finding best job matches for a candidate"""
        # Create additional jobs
        job2 = Job(
            id=uuid4(),
            title="Frontend Developer",
            description="Looking for a frontend developer with React and JavaScript experience.",
            required_skills=["JavaScript", "React", "HTML", "CSS"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.ACTIVE,
            created_by=test_job.created_by
        )
        test_db_session.add(job2)
        await test_db_session.commit()
        
        # Score candidate for both jobs
        for job in [test_job, job2]:
            response = client.post("/api/v1/scores/compute", 
                                 headers=auth_headers,
                                 json={
                                     "candidate_id": str(test_candidate.id),
                                     "job_id": str(job.id)
                                 })
            assert response.status_code == 201
        
        # Get best matches for candidate
        response = client.get(
            f"/api/v1/scores/candidates/{test_candidate.id}/best-matches?min_score=0.0&limit=10", 
            headers=auth_headers
        )
        assert response.status_code == 200
        
        matches = response.json()
        
        # Verify matches structure
        assert "candidate_id" in matches
        assert "candidate_name" in matches
        assert "matches" in matches
        assert "total_matches" in matches
        
        assert matches["candidate_id"] == str(test_candidate.id)
        assert matches["candidate_name"] == test_candidate.name
        assert matches["total_matches"] >= 1
        assert len(matches["matches"]) == matches["total_matches"]
        
        # Verify individual matches
        for match in matches["matches"]:
            assert "job_id" in match
            assert "job_title" in match
            assert "score" in match
            assert "created_at" in match
            assert 0.0 <= match["score"] <= 1.0
    
    def test_candidate_comparison(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_candidate: Candidate,
        second_candidate: Candidate
    ):
        """Test comparing specific candidates for a job"""
        # Score both candidates
        candidates = [test_candidate, second_candidate]
        
        for candidate in candidates:
            response = client.post("/api/v1/scores/compute", 
                                 headers=auth_headers,
                                 json={
                                     "candidate_id": str(candidate.id),
                                     "job_id": str(test_job.id),
                                     "generate_explanation": True
                                 })
            assert response.status_code == 201
        
        # Compare candidates
        response = client.post(f"/api/v1/scores/jobs/{test_job.id}/compare", 
                             headers=auth_headers,
                             json=[str(test_candidate.id), str(second_candidate.id)])
        
        assert response.status_code == 200
        comparison = response.json()
        
        # Verify comparison structure
        assert "job_id" in comparison
        assert "job_title" in comparison
        assert "candidates" in comparison
        assert "comparison_summary" in comparison
        
        assert comparison["job_id"] == str(test_job.id)
        assert len(comparison["candidates"]) == 2
        
        # Verify candidates are ranked
        candidates_list = comparison["candidates"]
        assert candidates_list[0]["rank"] == 1
        assert candidates_list[1]["rank"] == 2
        assert candidates_list[0]["score"] >= candidates_list[1]["score"]
        
        # Verify summary statistics
        summary = comparison["comparison_summary"]
        assert "total_candidates" in summary
        assert "average_score" in summary
        assert "score_range" in summary
        assert "top_candidate" in summary
    
    def test_rescore_job_candidates(
        self, 
        client: TestClient, 
        hm_headers: dict,  # Hiring manager required for rescore
        test_job: Job,
        test_candidate: Candidate,
        second_candidate: Candidate
    ):
        """Test rescoring all candidates for a job"""
        # Score candidates initially
        candidates = [test_candidate, second_candidate]
        
        for candidate in candidates:
            response = client.post("/api/v1/scores/compute", 
                                 headers=hm_headers,
                                 json={
                                     "candidate_id": str(candidate.id),
                                     "job_id": str(test_job.id)
                                 })
            assert response.status_code == 201
        
        # Rescore all candidates for the job
        response = client.post(f"/api/v1/scores/jobs/{test_job.id}/rescore", 
                             headers=hm_headers,
                             json={
                                 "generate_explanations": True
                             })
        
        assert response.status_code == 200
        rescore_result = response.json()
        
        # Verify rescore response
        assert "job_id" in rescore_result
        assert "job_title" in rescore_result
        assert "candidates_rescored" in rescore_result
        assert "candidates_failed" in rescore_result
        assert "message" in rescore_result
        assert "started_at" in rescore_result
        assert "completed_at" in rescore_result
        
        assert rescore_result["job_id"] == str(test_job.id)
        assert rescore_result["candidates_rescored"] >= 0
        assert rescore_result["candidates_failed"] >= 0
    
    def test_scoring_engine_info(self, client: TestClient, auth_headers: dict):
        """Test getting scoring engine information"""
        response = client.get("/api/v1/scores/engine/info", headers=auth_headers)
        assert response.status_code == 200
        
        engine_info = response.json()
        
        # Should contain basic engine information
        # The exact structure depends on the scoring engine implementation
        assert isinstance(engine_info, dict)
    
    def test_concurrent_scoring_requests(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_candidate: Candidate,
        second_candidate: Candidate
    ):
        """Test handling concurrent scoring requests"""
        import threading
        import time
        
        results = []
        errors = []
        
        def score_candidate(candidate_id):
            try:
                response = client.post("/api/v1/scores/compute", 
                                     headers=auth_headers,
                                     json={
                                         "candidate_id": str(candidate_id),
                                         "job_id": str(test_job.id),
                                         "force_rescore": True  # Ensure new scores
                                     })
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads to score candidates concurrently
        threads = []
        candidates = [test_candidate.id, second_candidate.id]
        
        for candidate_id in candidates:
            for _ in range(3):  # 3 requests per candidate
                thread = threading.Thread(target=score_candidate, args=(candidate_id,))
                threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent requests failed: {errors}"
        assert len(results) == 6  # 2 candidates * 3 requests each
        
        # All requests should succeed (either 201 for new score or existing score handling)
        for status_code in results:
            assert status_code in [201], f"Unexpected status code: {status_code}"
    
    async def test_large_candidate_pool_performance(
        self, 
        client: TestClient, 
        auth_headers: dict, 
        test_job: Job,
        test_db_session: AsyncSession
    ):
        """Test performance with larger candidate pools"""
        import time
        
        # Create multiple candidates for performance testing
        candidates = []
        for i in range(10):  # Create 10 candidates
            candidate = Candidate(
                id=uuid4(),
                name=f"Test Candidate {i}",
                email=f"candidate{i}@example.com",
                resume_file_path=f"s3://bucket/resume{i}.pdf",
                parsed_data={
                    "candidate_name": f"Test Candidate {i}",
                    "email": f"candidate{i}@example.com",
                    "work_experience": [
                        {
                            "company": f"Company {i}",
                            "title": "Software Developer",
                            "description": "Developed applications using various technologies.",
                            "confidence": 0.9
                        }
                    ],
                    "skills": ["Python", "JavaScript", "SQL"],
                    "education": [
                        {
                            "institution": f"University {i}",
                            "degree": "Bachelor of Science",
                            "confidence": 0.95
                        }
                    ]
                },
                skills=["Python", "JavaScript", "SQL"]
            )
            candidates.append(candidate)
            test_db_session.add(candidate)
        
        await test_db_session.commit()
        
        # Batch score all candidates
        candidate_ids = [str(c.id) for c in candidates]
        
        start_time = time.time()
        response = client.post("/api/v1/scores/batch", 
                             headers=auth_headers,
                             json={
                                 "candidate_ids": candidate_ids,
                                 "job_id": str(test_job.id)
                             })
        end_time = time.time()
        
        assert response.status_code == 201
        batch_result = response.json()
        
        # Verify batch processing completed successfully
        assert batch_result["total_candidates"] == 10
        assert batch_result["successful_scores"] >= 8  # Allow for some failures
        
        # Performance check - should complete within reasonable time
        processing_time = end_time - start_time
        assert processing_time < 30.0, f"Batch scoring took too long: {processing_time}s"
        
        # Test ranking performance
        start_time = time.time()
        response = client.get(f"/api/v1/jobs/{test_job.id}/candidates", headers=auth_headers)
        end_time = time.time()
        
        assert response.status_code == 200
        candidates_ranking = response.json()
        
        # Ranking should be fast
        ranking_time = end_time - start_time
        assert ranking_time < 5.0, f"Candidate ranking took too long: {ranking_time}s"
        
        # Verify ranking is correct
        if len(candidates_ranking) > 1:
            for i in range(len(candidates_ranking) - 1):
                assert candidates_ranking[i]["score"] >= candidates_ranking[i + 1]["score"]