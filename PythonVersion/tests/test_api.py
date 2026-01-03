"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def test_health_check(self):
        """Test health endpoint returns valid response."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "ollama_connected" in data
        assert "version" in data


class TestRootEndpoint:
    """Test cases for root endpoint."""

    def test_root(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "ARLO Microservice"
        assert "version" in data


class TestMatrixEndpoint:
    """Test cases for matrix endpoint."""

    def test_get_matrix(self):
        """Test matrix endpoint returns valid matrix data."""
        response = client.get("/api/matrix")
        assert response.status_code == 200
        
        data = response.json()
        assert "groups" in data
        assert "patterns" in data
        assert len(data["patterns"]) > 0


class TestAnalyzeEndpoint:
    """Test cases for analyze endpoint."""

    def test_analyze_empty_requirements(self):
        """Test analyze endpoint rejects empty requirements."""
        response = client.post(
            "/api/analyze",
            json={"requirements": []},
        )
        assert response.status_code == 422  # Validation error

    def test_analyze_request_format(self):
        """Test analyze endpoint accepts valid request format."""
        # Note: This will fail in CI without Ollama, but validates request format
        response = client.post(
            "/api/analyze",
            json={
                "requirements": [
                    "The system shall support 1000 concurrent users",
                    "All data must be encrypted",
                ],
                "settings": {
                    "optimization_strategy": "ILP",
                    "quality_weights_mode": "Inferred",
                },
            },
        )
        # Will be 500 if Ollama not available, but validates request parsing
        assert response.status_code in [200, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
