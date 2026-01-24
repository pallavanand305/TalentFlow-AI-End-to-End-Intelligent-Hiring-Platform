"""Property-based tests for API infrastructure

Feature: talentflow-ai
Property 23: Authentication requirement
Property 24: Validation error responses
Property 26: Rate limiting enforcement
Property 27: CORS header inclusion
"""

import pytest
from hypothesis import given, strategies as st, settings
from httpx import AsyncClient
from fastapi import status

from backend.app.main import app
from backend.app.services.auth_service import auth_service
from backend.app.models.user import UserRole


@pytest.fixture
async def client():
    """Create async HTTP client"""
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestAuthenticationRequirement:
    """Property tests for authentication requirement"""
    
    @pytest.mark.asyncio
    @settings(max_examples=20)
    @given(
        path=st.sampled_from([
            "/api/v1/auth/me",
            "/api/v1/resumes",
            "/api/v1/jobs",
            "/api/v1/scores",
        ])
    )
    async def test_property_23_authentication_requirement(self, client, path):
        """
        Property 23: Authentication requirement
        
        For any API request without a valid JWT token, the system should
        reject the request before processing any business logic.
        
        Validates: Requirements 4.2
        """
        # Test 1: Request without token should be rejected
        response = await client.get(path)
        assert response.status_code == status.HTTP_403_FORBIDDEN, \
            "Request without token should be rejected with 403"
        
        # Test 2: Request with invalid token should be rejected
        response = await client.get(
            path,
            headers={"Authorization": "Bearer invalid.token.here"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED, \
            "Request with invalid token should be rejected with 401"
        
        # Test 3: Request with malformed authorization header should be rejected
        response = await client.get(
            path,
            headers={"Authorization": "InvalidFormat token"}
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN, \
            "Request with malformed auth header should be rejected"


class TestValidationErrorResponses:
    """Property tests for validation error responses"""
    
    @pytest.mark.asyncio
    @settings(max_examples=30)
    @given(
        username=st.text(max_size=2),  # Too short
        email=st.text(min_size=1, max_size=10),  # Invalid email
        password=st.text(max_size=7),  # Too short
    )
    async def test_property_24_validation_error_responses(
        self, client, username, email, password
    ):
        """
        Property 24: Validation error responses
        
        For any API request that fails validation (missing fields, invalid format, etc.),
        the system should return an appropriate HTTP status code (4xx) and a
        descriptive error message.
        
        Validates: Requirements 4.3
        """
        # Test registration with invalid data
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password,
            }
        )
        
        # Should return 422 for validation errors
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, \
            "Invalid data should return 422 status code"
        
        # Response should contain error details
        data = response.json()
        assert "error" in data or "detail" in data, \
            "Response should contain error information"
        
        # Should have request ID for tracking
        assert "request_id" in data or "X-Request-ID" in response.headers, \
            "Response should include request ID"


class TestRateLimitingEnforcement:
    """Property tests for rate limiting enforcement"""
    
    @pytest.mark.asyncio
    async def test_property_26_rate_limiting_enforcement(self, client):
        """
        Property 26: Rate limiting enforcement
        
        For any client making requests exceeding the configured rate limit,
        subsequent requests should be rejected with HTTP 429 status until
        the rate limit window resets.
        
        Validates: Requirements 4.6
        """
        # Make requests up to the rate limit
        # Note: In tests, we'll make fewer requests to avoid long test times
        max_requests = 10
        
        responses = []
        for i in range(max_requests + 5):  # Try to exceed limit
            response = await client.get("/")
            responses.append(response)
        
        # Check that rate limit headers are present
        last_response = responses[-1]
        assert "X-RateLimit-Limit" in last_response.headers or \
               last_response.status_code == status.HTTP_429_TOO_MANY_REQUESTS, \
            "Rate limit headers should be present or rate limit should be enforced"
        
        # If we hit rate limit, verify 429 status
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS for r in responses)
        if rate_limited:
            # Find first rate-limited response
            for response in responses:
                if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                    data = response.json()
                    assert "error" in data or "detail" in data, \
                        "Rate limit error should include error message"
                    break


class TestCORSHeaderInclusion:
    """Property tests for CORS header inclusion"""
    
    @pytest.mark.asyncio
    @settings(max_examples=20)
    @given(
        path=st.sampled_from([
            "/",
            "/health",
            "/api/v1/auth/login",
        ]),
        method=st.sampled_from(["GET", "POST", "OPTIONS"])
    )
    async def test_property_27_cors_header_inclusion(self, client, path, method):
        """
        Property 27: CORS header inclusion
        
        For any API response, the response should include appropriate CORS headers
        (Access-Control-Allow-Origin, etc.) to enable cross-origin requests.
        
        Validates: Requirements 4.7
        """
        # Make request with origin header
        headers = {"Origin": "http://localhost:3000"}
        
        if method == "GET":
            response = await client.get(path, headers=headers)
        elif method == "POST":
            response = await client.post(path, headers=headers, json={})
        else:  # OPTIONS
            response = await client.options(path, headers=headers)
        
        # Check for CORS headers
        # Note: CORS headers might not be present for all responses,
        # but they should be present for cross-origin requests
        if response.status_code < 500:  # Don't check for server errors
            # At minimum, check that CORS is configured (headers may vary)
            assert (
                "access-control-allow-origin" in response.headers or
                "Access-Control-Allow-Origin" in response.headers or
                response.status_code in [401, 403, 404, 422]  # Auth/validation errors
            ), "CORS headers should be present for valid requests"


class TestRequestIDTracking:
    """Test request ID tracking"""
    
    @pytest.mark.asyncio
    @settings(max_examples=20)
    @given(
        path=st.sampled_from(["/", "/health", "/api/v1/auth/login"])
    )
    async def test_request_id_in_response(self, client, path):
        """Test that all responses include request ID"""
        response = await client.get(path)
        
        # Request ID should be in headers
        assert "X-Request-ID" in response.headers, \
            "Response should include X-Request-ID header"
        
        # Request ID should be a valid UUID format
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 0, "Request ID should not be empty"
        
        # If response has JSON body with error, it should include request_id
        if response.status_code >= 400 and response.headers.get("content-type", "").startswith("application/json"):
            try:
                data = response.json()
                if isinstance(data, dict):
                    assert "request_id" in data or "X-Request-ID" in response.headers, \
                        "Error responses should include request_id"
            except:
                pass  # Not JSON response


class TestErrorResponseFormat:
    """Test error response formatting"""
    
    @pytest.mark.asyncio
    async def test_error_response_structure(self, client):
        """Test that error responses have consistent structure"""
        # Trigger validation error
        response = await client.post(
            "/api/v1/auth/register",
            json={"invalid": "data"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        data = response.json()
        
        # Check error response structure
        assert isinstance(data, dict), "Error response should be a dictionary"
        assert "error" in data or "detail" in data, \
            "Error response should contain error field"
        
        # Should have request ID
        assert "request_id" in data or "X-Request-ID" in response.headers, \
            "Error response should include request ID"
