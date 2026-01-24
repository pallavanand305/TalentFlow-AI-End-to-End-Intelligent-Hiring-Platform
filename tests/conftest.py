"""Pytest configuration and shared fixtures"""

import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from uuid import uuid4

from backend.app.core.config import settings
from backend.app.core.database import Base
from backend.app.models import (
    User, Job, Candidate, Score, ModelVersion, BackgroundJob
)
from backend.app.models.user import UserRole
from backend.app.services.auth_service import AuthService


# Test database URL - use SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_talentflow.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine and setup schema"""
    # Create engine
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        
        # Insert alembic version for migration tests
        await conn.execute(
            text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES ('001') ON CONFLICT DO NOTHING")
        )
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture(scope="session")
async def test_session_factory(test_engine):
    """Create test session factory"""
    return async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


@pytest.fixture
async def db_session(test_session_factory) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session with automatic rollback"""
    async with test_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


@pytest.fixture
async def test_db_session(test_session_factory) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session with automatic rollback (alias for compatibility)"""
    async with test_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


async def create_test_user(db_session: AsyncSession, role: UserRole = UserRole.RECRUITER) -> User:
    """Create a test user for authentication tests"""
    auth_service = AuthService()
    
    user = User(
        id=uuid4(),
        username=f"testuser_{uuid4().hex[:8]}",
        email=f"test_{uuid4().hex[:8]}@example.com",
        password_hash=auth_service.hash_password("testpassword123"),
        role=role
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


def get_auth_headers(user: User) -> dict:
    """Get authentication headers for a test user"""
    auth_service = AuthService()
    token = auth_service.create_access_token(str(user.id), user.role.value)
    return {"Authorization": f"Bearer {token}"}
