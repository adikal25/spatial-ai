"""
Basic tests for the API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "version" in data
    assert "models_loaded" in data


def test_ask_endpoint_missing_fields():
    """Test ask endpoint with missing required fields."""
    response = client.post("/ask", json={})
    assert response.status_code == 422  # Validation error


def test_ask_endpoint_invalid_base64():
    """Test ask endpoint with invalid base64 image."""
    response = client.post("/ask", json={
        "image": "not_valid_base64",
        "question": "What do you see?"
    })
    # Should return 500 or validation error
    assert response.status_code in [422, 500]


@pytest.mark.skip(reason="Requires API keys and takes time")
def test_ask_endpoint_valid_request():
    """Test ask endpoint with valid request (requires API keys)."""
    # This test requires actual API keys and a valid base64 image
    # Skip by default to avoid API costs during testing
    pass
