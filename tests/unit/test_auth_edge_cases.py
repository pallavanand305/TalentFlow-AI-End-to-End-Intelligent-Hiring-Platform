"""Unit tests for authentication edge cases"""

import pytest
from datetime import timedelta
import uuid

from backend.app.services.auth_service import AuthService
from backend.app.repositories.user_repository import UserRepository
from backend.app.models.user import User, UserRole
from backend.app.core.security import check_permission


class TestAuthEdgeCases:
    """Unit tests for authentication edge cases"""
    
    def test_expired_token_rejected(self):
        """Test that expired tokens are properly rejected"""
        auth_service = AuthService()
        
        # Create token that expires immediately
        token = auth_service.create_access_token(
            user_id=str(uuid.uuid4()),
            username="testuser",
            role=UserRole.RECRUITER,
            expires_delta=timedelta(seconds=-10)  # Expired 10 seconds ago
        )
        
        payload = auth_service.verify_access_token(token)
        assert payload is None, "Expired token should be rejected"
    
    def test_invalid_signature_rejected(self):
        """Test that tokens with invalid signatures are rejected"""
        auth_service = AuthService()
        
        # Create token with wrong secret
        from jose import jwt
        token = jwt.encode(
            {"sub": str(uuid.uuid4()), "username": "test", "role": "recruiter"},
            "wrong-secret",
            algorithm="HS256"
        )
        
        payload = auth_service.verify_access_token(token)
        assert payload is None, "Token with invalid signature should be rejected"
    
    def test_missing_token_fields(self):
        """Test tokens with missing required fields"""
        auth_service = AuthService()
        from jose import jwt
        
        # Token without 'sub' field but with proper type
        token_no_sub = jwt.encode(
            {"username": "test", "role": "recruiter", "type": "access"},
            auth_service.secret_key,
            algorithm=auth_service.algorithm
        )
        
        payload = auth_service.verify_access_token(token_no_sub)
        # Should still verify but missing fields
        assert payload is not None
        assert "sub" not in payload
    
    def test_wrong_token_type(self):
        """Test that refresh tokens are not accepted as access tokens"""
        auth_service = AuthService()
        
        refresh_token = auth_service.create_refresh_token(
            user_id=str(uuid.uuid4()),
            username="testuser"
        )
        
        # Try to verify as access token
        payload = auth_service.verify_access_token(refresh_token)
        assert payload is None, "Refresh token should not be accepted as access token"
    
    def test_malformed_token(self):
        """Test that malformed tokens are rejected"""
        auth_service = AuthService()
        
        malformed_tokens = [
            "not.a.token",
            "invalid",
            "",
            "a.b",  # Only 2 parts instead of 3
            "a.b.c.d",  # Too many parts
        ]
        
        for token in malformed_tokens:
            payload = auth_service.verify_access_token(token)
            assert payload is None, f"Malformed token '{token}' should be rejected"


class TestPasswordHashingEdgeCases:
    """Unit tests for password hashing edge cases"""
    
    def test_empty_password(self):
        """Test hashing empty password should raise an error"""
        with pytest.raises((ValueError, Exception)):
            UserRepository.hash_password("")
    
    def test_very_long_password(self):
        """Test hashing very long password"""
        long_password = "a" * 1000
        hashed = UserRepository.hash_password(long_password)
        assert UserRepository.verify_password(long_password, hashed)
    
    def test_special_characters_password(self):
        """Test password with special characters"""
        special_password = "p@$$w0rd!#%&*()[]{}|\\:;\"'<>,.?/~`"
        hashed = UserRepository.hash_password(special_password)
        assert UserRepository.verify_password(special_password, hashed)
    
    def test_unicode_password(self):
        """Test password with unicode characters"""
        unicode_password = "пароль密码🔐"
        hashed = UserRepository.hash_password(unicode_password)
        assert UserRepository.verify_password(unicode_password, hashed)
    
    def test_password_case_sensitivity(self):
        """Test that password verification is case-sensitive"""
        password = "Password123"
        hashed = UserRepository.hash_password(password)
        
        assert UserRepository.verify_password(password, hashed)
        assert not UserRepository.verify_password("password123", hashed)
        assert not UserRepository.verify_password("PASSWORD123", hashed)
    
    def test_similar_passwords_different_hashes(self):
        """Test that similar passwords produce different hashes"""
        password1 = "password123"
        password2 = "password124"
        
        hash1 = UserRepository.hash_password(password1)
        hash2 = UserRepository.hash_password(password2)
        
        assert hash1 != hash2
        assert not UserRepository.verify_password(password1, hash2)
        assert not UserRepository.verify_password(password2, hash1)


class TestRolePermissionEdgeCases:
    """Unit tests for role permission edge cases"""
    
    def test_admin_has_all_permissions(self):
        """Test that admin role has all permissions"""
        admin_user = User(
            id=uuid.uuid4(),
            username="admin",
            email="admin@test.com",
            password_hash="hash",
            role=UserRole.ADMIN
        )
        
        # Admin should have all permissions
        assert check_permission(admin_user, UserRole.ADMIN)
        assert check_permission(admin_user, UserRole.RECRUITER)
        assert check_permission(admin_user, UserRole.HIRING_MANAGER)
    
    def test_recruiter_limited_permissions(self):
        """Test that recruiter has limited permissions"""
        recruiter_user = User(
            id=uuid.uuid4(),
            username="recruiter",
            email="recruiter@test.com",
            password_hash="hash",
            role=UserRole.RECRUITER
        )
        
        # Recruiter should only have recruiter permissions
        assert not check_permission(recruiter_user, UserRole.ADMIN)
        assert check_permission(recruiter_user, UserRole.RECRUITER)
        assert not check_permission(recruiter_user, UserRole.HIRING_MANAGER)
    
    def test_hiring_manager_limited_permissions(self):
        """Test that hiring manager has limited permissions"""
        hm_user = User(
            id=uuid.uuid4(),
            username="hm",
            email="hm@test.com",
            password_hash="hash",
            role=UserRole.HIRING_MANAGER
        )
        
        # Hiring manager should only have hiring manager permissions
        assert not check_permission(hm_user, UserRole.ADMIN)
        assert not check_permission(hm_user, UserRole.RECRUITER)
        assert check_permission(hm_user, UserRole.HIRING_MANAGER)


class TestTokenExtractionEdgeCases:
    """Unit tests for token extraction edge cases"""
    
    def test_extract_user_id_from_valid_token(self):
        """Test extracting user ID from valid token"""
        auth_service = AuthService()
        user_id = str(uuid.uuid4())
        
        token = auth_service.create_access_token(
            user_id=user_id,
            username="test",
            role=UserRole.RECRUITER
        )
        
        extracted_id = auth_service.get_user_id_from_token(token)
        assert extracted_id == user_id
    
    def test_extract_user_id_from_invalid_token(self):
        """Test extracting user ID from invalid token"""
        auth_service = AuthService()
        
        extracted_id = auth_service.get_user_id_from_token("invalid.token")
        assert extracted_id is None
    
    def test_extract_role_from_valid_token(self):
        """Test extracting role from valid token"""
        auth_service = AuthService()
        
        token = auth_service.create_access_token(
            user_id=str(uuid.uuid4()),
            username="test",
            role=UserRole.ADMIN
        )
        
        extracted_role = auth_service.get_user_role_from_token(token)
        assert extracted_role == UserRole.ADMIN
    
    def test_extract_role_from_invalid_token(self):
        """Test extracting role from invalid token"""
        auth_service = AuthService()
        
        extracted_role = auth_service.get_user_role_from_token("invalid.token")
        assert extracted_role is None
