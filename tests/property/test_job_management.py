"""Property-based tests for job management functionality"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from uuid import uuid4
from datetime import datetime

from backend.app.models.job import Job, JobStatus, ExperienceLevel
from backend.app.services.job_service import JobService
from backend.app.repositories.job_repository import JobRepository
from backend.app.core.exceptions import ValidationException, NotFoundException


# Test data strategies
job_titles = st.text(min_size=3, max_size=255).filter(lambda x: x.strip() and len(x.strip()) >= 3)
job_descriptions = st.text(min_size=10, max_size=5000).filter(lambda x: x.strip() and len(x.strip()) >= 10)
skill_names = st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and x.isalnum())
skills_lists = st.lists(skill_names, min_size=1, max_size=50, unique=True)
experience_levels = st.sampled_from(list(ExperienceLevel))
job_statuses = st.sampled_from(list(JobStatus))
locations = st.one_of(st.none(), st.text(min_size=1, max_size=255).filter(lambda x: x.strip()))
salaries = st.one_of(st.none(), st.floats(min_value=0, max_value=1000000, allow_nan=False, allow_infinity=False))
user_ids = st.builds(uuid4)


class MockJobRepository:
    """Mock job repository for testing"""
    
    def __init__(self):
        self.jobs = {}
        self.history = {}
        self.next_id = 1
    
    async def create(self, job_data):
        job_id = uuid4()
        job = Job(
            id=job_id,
            title=job_data['title'],
            description=job_data['description'],
            required_skills=job_data['required_skills'],
            experience_level=job_data['experience_level'],
            location=job_data.get('location'),
            salary_min=job_data.get('salary_min'),
            salary_max=job_data.get('salary_max'),
            status=job_data.get('status', JobStatus.ACTIVE),
            created_by=job_data['created_by'],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        self.jobs[job_id] = job
        return job
    
    async def get_by_id(self, job_id):
        return self.jobs.get(job_id)
    
    async def update(self, job_id, updates, updated_by):
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = datetime.utcnow()
        
        return job
    
    async def delete(self, job_id):
        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.INACTIVE
            return True
        return False
    
    async def search(self, **kwargs):
        return list(self.jobs.values())
    
    async def get_history(self, job_id):
        return self.history.get(job_id, [])
    
    async def count(self, **kwargs):
        return len(self.jobs)
    
    async def get_jobs_by_creator(self, creator_id, limit=10):
        return [job for job in self.jobs.values() if job.created_by == creator_id][:limit]
    
    async def get_active_jobs_count(self):
        return len([job for job in self.jobs.values() if job.status == JobStatus.ACTIVE])
    
    async def get_jobs_by_skills(self, skills, limit=10):
        matching_jobs = []
        for job in self.jobs.values():
            if job.status == JobStatus.ACTIVE:
                if any(skill in job.required_skills for skill in skills):
                    matching_jobs.append(job)
        return matching_jobs[:limit]


class TestJobManagementProperties:
    """Property-based tests for job management"""
    
    def create_job_service(self):
        """Create a fresh job service for each test"""
        mock_repo = MockJobRepository()
        return JobService(mock_repo)
    
    @given(
        title=job_titles,
        description=job_descriptions,
        required_skills=skills_lists,
        experience_level=experience_levels,
        created_by=user_ids,
        location=locations,
        salary_min=salaries,
        salary_max=salaries
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_10_job_creation_validation_and_storage(
        self, title, description, required_skills, experience_level, 
        created_by, location, salary_min, salary_max
    ):
        """
        **Feature: talentflow-ai, Property 10: Job creation validation and storage**
        
        For any job creation request with all required fields, the system should 
        validate the fields and successfully store the job in the Job_Repository.
        
        **Validates: Requirements 2.1**
        """
        job_service = self.create_job_service()
        
        # Skip invalid salary combinations
        if (salary_min is not None and salary_max is not None and 
            salary_min > salary_max):
            assume(False)
        
        # Create job
        job = await job_service.create_job(
            title=title,
            description=description,
            required_skills=required_skills,
            experience_level=experience_level,
            created_by=created_by,
            location=location,
            salary_min=salary_min,
            salary_max=salary_max
        )
        
        # Verify job was created with correct data
        assert job is not None
        assert job.id is not None
        assert job.title == title.strip()
        assert job.description == description.strip()
        assert job.required_skills == [skill.strip() for skill in required_skills if skill.strip()]
        assert job.experience_level == experience_level
        assert job.created_by == created_by
        assert job.location == (location.strip() if location else None)
        assert job.salary_min == salary_min
        assert job.salary_max == salary_max
        assert job.status == JobStatus.ACTIVE
        assert job.created_at is not None
        assert job.updated_at is not None
    
    @given(
        title=st.one_of(st.just(""), st.just("  "), st.just("ab")),  # Invalid titles
        description=job_descriptions,
        required_skills=skills_lists,
        experience_level=experience_levels,
        created_by=user_ids
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_11_required_field_enforcement_title(
        self, title, description, required_skills, experience_level, created_by
    ):
        """
        **Feature: talentflow-ai, Property 11: Required field enforcement**
        
        For any job creation request missing one or more required fields (title, description, 
        required_skills, experience_level), the system should reject the request with a validation error.
        
        **Validates: Requirements 2.2**
        """
        job_service = self.create_job_service()
        
        with pytest.raises(ValidationException):
            await job_service.create_job(
                title=title,
                description=description,
                required_skills=required_skills,
                experience_level=experience_level,
                created_by=created_by
            )
    
    @given(
        job_data=st.builds(
            dict,
            title=job_titles,
            description=job_descriptions,
            required_skills=skills_lists,
            experience_level=experience_levels,
            created_by=user_ids
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_12_job_id_uniqueness(self, job_data):
        """
        **Feature: talentflow-ai, Property 12: Job ID uniqueness**
        
        For any set of created jobs, all assigned job IDs should be unique across the system.
        
        **Validates: Requirements 2.3**
        """
        job_service = self.create_job_service()
        
        # Create multiple jobs
        jobs = []
        for _ in range(5):
            job = await job_service.create_job(**job_data)
            jobs.append(job)
        
        # Verify all job IDs are unique
        job_ids = [job.id for job in jobs]
        assert len(job_ids) == len(set(job_ids)), "All job IDs should be unique"
    
    @given(
        job_data=st.builds(
            dict,
            title=job_titles,
            description=job_descriptions,
            required_skills=skills_lists,
            experience_level=experience_levels,
            created_by=user_ids,
            location=locations,
            salary_min=salaries,
            salary_max=salaries
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_13_job_retrieval_round_trip(self, job_data):
        """
        **Feature: talentflow-ai, Property 13: Job retrieval round-trip**
        
        For any created job, retrieving it by ID should return all the original job details 
        including metadata that was stored during creation.
        
        **Validates: Requirements 2.4**
        """
        job_service = self.create_job_service()
        
        # Skip invalid salary combinations
        if (job_data.get('salary_min') is not None and 
            job_data.get('salary_max') is not None and 
            job_data['salary_min'] > job_data['salary_max']):
            assume(False)
        
        # Create job
        created_job = await job_service.create_job(**job_data)
        
        # Retrieve job
        retrieved_job = await job_service.get_job(created_job.id)
        
        # Verify all data matches
        assert retrieved_job.id == created_job.id
        assert retrieved_job.title == job_data['title'].strip()
        assert retrieved_job.description == job_data['description'].strip()
        assert retrieved_job.required_skills == [skill.strip() for skill in job_data['required_skills'] if skill.strip()]
        assert retrieved_job.experience_level == job_data['experience_level']
        assert retrieved_job.created_by == job_data['created_by']
        assert retrieved_job.location == (job_data['location'].strip() if job_data.get('location') else None)
        assert retrieved_job.salary_min == job_data.get('salary_min')
        assert retrieved_job.salary_max == job_data.get('salary_max')
        assert retrieved_job.status == JobStatus.ACTIVE
        assert retrieved_job.created_at is not None
        assert retrieved_job.updated_at is not None
    
    @given(
        job_data=st.builds(
            dict,
            title=job_titles,
            description=job_descriptions,
            required_skills=skills_lists,
            experience_level=experience_levels,
            created_by=user_ids
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_property_15_soft_delete_behavior(self, job_data):
        """
        **Feature: talentflow-ai, Property 15: Soft delete behavior**
        
        For any job deletion operation, the job record should remain in the database 
        but be marked with status='inactive' rather than being physically removed.
        
        **Validates: Requirements 2.6**
        """
        job_service = self.create_job_service()
        
        # Create job
        job = await job_service.create_job(**job_data)
        assert job.status == JobStatus.ACTIVE
        
        # Delete job
        success = await job_service.delete_job(job.id)
        assert success is True
        
        # Verify job still exists but is marked inactive
        deleted_job = await job_service.get_job(job.id)
        assert deleted_job is not None
        assert deleted_job.id == job.id
        assert deleted_job.status == JobStatus.INACTIVE