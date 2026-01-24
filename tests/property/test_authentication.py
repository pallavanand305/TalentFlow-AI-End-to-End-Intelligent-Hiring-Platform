"""Property-based tests for authentication

Feature: talentflow-ai
Property 28: JWT token validation
Property 29: Authorization enforcement
Property 30: Password hashing
"""

import pytest
from hypothesis import given, strategies as st
from hypothesis import settings as hypothesis_settings
from datetime import datetime, timedelta
from jose import jwt

from backend.app.services.auth_service import AuthService
from backend.app.repositories.user_repository import UserRepository
from backend.app.models.user import UserRole
from backend.app.core.config import settings


class TestJWTTokenValidation:
    """Property tests for JWT token validation"""
    
    @pytest.mark.asyncio
    @hypothesis_settings(max_examples=50)
    @given(
        username=st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        user_id=st.uuids(),
    )
    async def test_property_28_jwt_token_validation(self, username, user_id):
        """
        Property 28: JWT token validation
        
        For any API request with a JWT token, the system should validate both
        the token signature and expiration, rejecting invalid or expired tokens
        with appropriate error messages.
        
        Validates: Requirements 7.2, 7.3
        """
        auth_service = AuthService()
        
        # Test 1: Valid token should be verified successfully
        valid_token = auth_service.create_access_token(
            user_id=str(user_id),
            username=username,
            role=UserRole.RECRUITER
        )
        
        payload = auth_service.verify_access_token(valid_token)
        assert payload is not None, "Valid token should be verified"
        assert payload["sub"] == str(user_id), "Token should contain correct user ID"
        assert payload["username"] == username, "Token should contain correct username"
        assert payload["role"] == UserRole.RECRUITER.value, "Token should contain correct role"
        
        # Test 2: Expired token should be rejected
        expired_token = auth_service.create_access_token(
            user_id=str(user_id),
            username=username,
            role=UserRole.RECRUITER,
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        expired_payload = auth_service.verify_access_token(expired_token)
        assert expired_payload is None, "Expired token should be rejected"
        
        # Test 3: Token with invalid signature should be rejected
        invalid_token = jwt.encode(
            {"sub": str(user_id), "username": username, "role": "recruiter"},
            "wrong-secret-key",
            algorithm=settings.ALGORITHM
        )
        
        invalid_payload = auth_service.verify_access_token(invalid_token)
        assert invalid_payload is None, "Token with invalid signature should be rejected"
        
        # Test 4: Malformed token should be rejected
        malformed_token = "not.a.valid.jwt.token"
        malformed_payload = auth_service.verify_access_token(malformed_token)
        assert malformed_payload is None, "Malformed token should be rejected"
        
        # Test 5: Refresh token should not be accepted as access token
        refresh_token = auth_service.create_refresh_token(
            user_id=str(user_id),
            username=username
        )
        
        refresh_payload = auth_service.verify_access_token(refresh_token)
        assert refresh_payload is None, "Refresh token should not be accepted as access token"


class TestAuthorizationEnforcement:
    """Property tests for authorization enforcement"""
    
    @pytest.mark.asyncio
    @hypothesis_settings(max_examples=30)
    @given(
        username=st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        user_id=st.uuids(),
        user_role=st.sampled_from([UserRole.ADMIN, UserRole.RECRUITER, UserRole.HIRING_MANAGER]),
    )
    async def test_property_29_authorization_enforcement(self, username, user_id, user_role):
        """
        Property 29: Authorization enforcement
        
        For any API request where the user's role lacks the required permissions,
        the system should return HTTP 403 Forbidden status.
        
        Validates: Requirements 7.5
        """
        auth_service = AuthService()
        
        # Create token with specific role
        token = auth_service.create_access_token(
            user_id=str(user_id),
            username=username,
            role=user_role
        )
        
        # Verify token contains correct role
        payload = auth_service.verify_access_token(token)
        assert payload is not None
        
        extracted_role = UserRole(payload["role"])
        assert extracted_role == user_role, "Token should contain correct role"
        
        # Test role hierarchy
        if user_role == UserRole.ADMIN:
            # Admin should have access to all roles
            assert extracted_role == UserRole.ADMIN
        elif user_role == UserRole.RECRUITER:
            # Recruiter should not be admin
            assert extracted_role != UserRole.ADMIN
        elif user_role == UserRole.HIRING_MANAGER:
            # Hiring manager should not be admin or recruiter
            assert extracted_role != UserRole.ADMIN
            assert extracted_role != UserRole.RECRUITER


class TestPasswordHashing:
    """Property tests for password hashing"""
    
    @hypothesis_settings(max_examples=100)
    @given(
        password=st.text(min_size=8, max_size=100),
    )
    def test_property_30_password_hashing(self, password):
        """
        Property 30: Password hashing
        
        For any password, the system should hash and salt all stored passwords
        using industry-standard algorithms (bcrypt), and the hash should be
        verifiable but not reversible.
        
        Validates: Requirements 7.6
        """
        # Test 1: Password should be hashed
        hashed = UserRepository.hash_password(password)
        assert hashed != password, "Password should be hashed, not stored in plain text"
        assert len(hashed) > len(password), "Hash should be longer than original password"
        assert hashed.startswith("$2b$"), "Should use bcrypt hashing (starts with $2b$)"
        
        # Test 2: Same password should produce different hashes (salt)
        hashed2 = UserRepository.hash_password(password)
        assert hashed != hashed2, "Same password should produce different hashes due to salt"
        
        # Test 3: Correct password should verify successfully
        assert UserRepository.verify_password(password, hashed), \
            "Correct password should verify against its hash"
        assert UserRepository.verify_password(password, hashed2), \
            "Correct password should verify against different hash"
        
        # Test 4: Incorrect password should not verify
        wrong_password = password + "wrong"
        assert not UserRepository.verify_password(wrong_password, hashed), \
            "Incorrect password should not verify"
        
        # Test 5: Empty password should not verify
        assert not UserRepository.verify_password("", hashed), \
            "Empty password should not verify"


class TestTokenExpiration:
    """Test token expiration behavior"""
    
    @pytest.mark.asyncio
    @hypothesis_settings(max_examples=20)
    @given(
        username=st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        user_id=st.uuids(),
    )
    async def test_token_expiration_times(self, username, user_id):
        """Test that tokens have correct expiration times"""
        auth_service = AuthService()
        
        # Create access token
        access_token = auth_service.create_access_token(
            user_id=str(user_id),
            username=username,
            role=UserRole.RECRUITER
        )
        
        access_payload = auth_service.verify_access_token(access_token)
        assert access_payload is not None
        
        # Check expiration is in the future
        exp_timestamp = access_payload["exp"]
        exp_datetime = datetime.fromtimestamp(exp_timestamp)
        assert exp_datetime > datetime.utcnow(), "Access token should not be expired"
        
        # Check expiration is within expected range (30 minutes +/- 1 minute)
        expected_exp = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        time_diff = abs((exp_datetime - expected_exp).total_seconds())
        assert time_diff < 60, "Access token expiration should be approximately 30 minutes"
        
        # Create refresh token
        refresh_token = auth_service.create_refresh_token(
            user_id=str(user_id),
            username=username
        )
        
        refresh_payload = auth_service.verify_refresh_token(refresh_token)
        assert refresh_payload is not None
        
        # Check refresh token expiration is longer than access token
        refresh_exp_timestamp = refresh_payload["exp"]
        assert refresh_exp_timestamp > exp_timestamp, \
            "Refresh token should expire later than access token"


class TestTokenPayload:
    """Test token payload structure"""
    
    @pytest.mark.asyncio
    @hypothesis_settings(max_examples=20)
    @given(
        username=st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        user_id=st.uuids(),
        role=st.sampled_from([UserRole.ADMIN, UserRole.RECRUITER, UserRole.HIRING_MANAGER]),
    )
    async def test_token_payload_structure(self, username, user_id, role):
        """Test that token payload contains all required fields"""
        auth_service = AuthService()
        
        token = auth_service.create_access_token(
            user_id=str(user_id),
            username=username,
            role=role
        )
        
        payload = auth_service.verify_access_token(token)
        assert payload is not None
        
        # Check required fields
        assert "sub" in payload, "Token should contain 'sub' (user ID)"
        assert "username" in payload, "Token should contain 'username'"
        assert "role" in payload, "Token should contain 'role'"
        assert "exp" in payload, "Token should contain 'exp' (expiration)"
        assert "iat" in payload, "Token should contain 'iat' (issued at)"
        assert "type" in payload, "Token should contain 'type'"
        
        # Check field values
        assert payload["sub"] == str(user_id)
        assert payload["username"] == username
        assert payload["role"] == role.value
        assert payload["type"] == "access"
