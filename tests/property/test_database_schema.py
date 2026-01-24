"""Property-based tests for database schema

Feature: talentflow-ai
Property 49: Schema migration versioning
Property 50: Referential integrity enforcement
"""

import pytest
from hypothesis import given, strategies as st
from hypothesis import settings as hypothesis_settings
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import uuid

from backend.app.core.config import settings
from backend.app.models import (
    User, Job, Candidate, Score, ModelVersion, BackgroundJob,
    UserRole, JobStatus, ExperienceLevel, BackgroundJobStatus
)


# Test database URL (use a separate test database)
TEST_DATABASE_URL = settings.DATABASE_URL.replace("/talentflow", "/talentflow_test")


@pytest.fixture(scope="module")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    yield engine
    await engine.dispose()


@pytest.fixture(scope="module")
async def test_session_factory(test_engine):
    """Create test session factory"""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    return async_session


@pytest.fixture
async def db_session(test_session_factory):
    """Create a test database session"""
    async with test_session_factory() as session:
        yield session
        await session.rollback()


class TestDatabaseSchema:
    """Property tests for database schema"""
    
    @pytest.mark.asyncio
    @hypothesis_settings(max_examples=10)
    @given(
        username=st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        email=st.emails(),
    )
    async def test_property_49_schema_migration_versioning(
        self, test_engine, username, email
    ):
        """
        Property 49: Schema migration versioning
        
        For any database connection, the alembic_version table should exist
        and contain exactly one version record, indicating the current schema version.
        
        Validates: Requirements 11.2
        """
        async with test_engine.connect() as conn:
            # Check if alembic_version table exists
            result = await conn.execute(
                text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables "
                    "WHERE table_name = 'alembic_version')"
                )
            )
            table_exists = result.scalar()
            assert table_exists, "alembic_version table should exist"
            
            # Check that there is exactly one version record
            result = await conn.execute(text("SELECT COUNT(*) FROM alembic_version"))
            version_count = result.scalar()
            assert version_count == 1, "Should have exactly one migration version"
            
            # Check that version is not null
            result = await conn.execute(text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
            assert version is not None, "Migration version should not be null"
            assert len(version) > 0, "Migration version should not be empty"
    
    @pytest.mark.asyncio
    @hypothesis_settings(max_examples=20)
    @given(
        job_title=st.text(min_size=5, max_size=100),
        candidate_name=st.text(min_size=3, max_size=100),
    )
    async def test_property_50_referential_integrity(
        self, db_session, job_title, candidate_name
    ):
        """
        Property 50: Referential integrity enforcement
        
        For any attempt to create a record with a foreign key reference,
        the database should enforce referential integrity by rejecting
        records that reference non-existent parent records.
        
        Validates: Requirements 11.3
        """
        # Try to create a job with non-existent user (should fail)
        non_existent_user_id = uuid.uuid4()
        
        job = Job(
            id=uuid.uuid4(),
            title=job_title,
            description="Test job description",
            required_skills=["Python", "SQL"],
            experience_level=ExperienceLevel.MID,
            status=JobStatus.ACTIVE,
            created_by=non_existent_user_id
        )
        
        db_session.add(job)
        
        # This should raise an integrity error
        with pytest.raises(Exception) as exc_info:
            await db_session.commit()
        
        # Verify it's a foreign key constraint violation
        assert "foreign key" in str(exc_info.value).lower() or \
               "violates" in str(exc_info.value).lower() or \
               "constraint" in str(exc_info.value).lower()
        
        await db_session.rollback()
        
        # Try to create a score with non-existent candidate and job (should fail)
        score = Score(
            id=uuid.uuid4(),
            candidate_id=uuid.uuid4(),
            job_id=uuid.uuid4(),
            score=0.85,
            model_version="v1.0",
            is_current=True
        )
        
        db_session.add(score)
        
        with pytest.raises(Exception) as exc_info:
            await db_session.commit()
        
        assert "foreign key" in str(exc_info.value).lower() or \
               "violates" in str(exc_info.value).lower() or \
               "constraint" in str(exc_info.value).lower()


class TestDatabaseIndexes:
    """Test that required indexes exist for performance"""
    
    @pytest.mark.asyncio
    async def test_required_indexes_exist(self, test_engine):
        """Verify that all required indexes are created"""
        async with test_engine.connect() as conn:
            inspector = inspect(conn.sync_connection)
            
            # Check users table indexes
            users_indexes = inspector.get_indexes('users')
            users_index_columns = [idx['column_names'] for idx in users_indexes]
            assert ['username'] in users_index_columns
            assert ['email'] in users_index_columns
            
            # Check jobs table indexes
            jobs_indexes = inspector.get_indexes('jobs')
            jobs_index_columns = [idx['column_names'] for idx in jobs_indexes]
            assert ['status'] in jobs_index_columns
            
            # Check candidates table indexes
            candidates_indexes = inspector.get_indexes('candidates')
            candidates_index_columns = [idx['column_names'] for idx in candidates_indexes]
            assert ['name'] in candidates_index_columns
            
            # Check scores table indexes
            scores_indexes = inspector.get_indexes('scores')
            scores_index_columns = [idx['column_names'] for idx in scores_indexes]
            assert ['candidate_id'] in scores_index_columns
            assert ['job_id'] in scores_index_columns


class TestDatabaseConstraints:
    """Test that unique constraints are properly enforced"""
    
    @pytest.mark.asyncio
    @hypothesis_settings(max_examples=10)
    @given(
        username=st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        email=st.emails(),
    )
    async def test_unique_constraints_enforced(self, db_session, username, email):
        """Verify unique constraints prevent duplicate records"""
        # Create first user
        user1 = User(
            id=uuid.uuid4(),
            username=username,
            email=email,
            password_hash="hashed_password",
            role=UserRole.RECRUITER
        )
        db_session.add(user1)
        await db_session.commit()
        
        # Try to create second user with same username (should fail)
        user2 = User(
            id=uuid.uuid4(),
            username=username,
            email=f"different_{email}",
            password_hash="hashed_password",
            role=UserRole.RECRUITER
        )
        db_session.add(user2)
        
        with pytest.raises(Exception) as exc_info:
            await db_session.commit()
        
        assert "unique" in str(exc_info.value).lower() or \
               "duplicate" in str(exc_info.value).lower()
        
        await db_session.rollback()
        
        # Clean up
        await db_session.delete(user1)
        await db_session.commit()
