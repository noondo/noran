import pytest
from fastapi.testclient import TestClient

from production_app.api import app


# إعداد بيئة الاختبار لتشغيل أحداث البداية (Startup Events)
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Test if the health endpoint returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_input(client):
    """Test the predict endpoint with valid features."""
    payload = {
        "mean_perimeter": 122.8,
        "mean_concave_points": 0.1471,
        "worst_radius": 25.38,
        "worst_perimeter": 184.6,
        "worst_concave_points": 0.2654,
    }
    response = client.post("/predict", json=payload)

    # Verify the response is successful and contains the expected output keys
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "label" in data
    assert "confidence_score" in data


def test_predict_invalid_input(client):
    """Test the predict endpoint with missing data to ensure validation works."""
    payload = {
        "mean_perimeter": 122.8
        # Missing the other 4 required features intentionally
    }
    response = client.post("/predict", json=payload)

    # Pydantic should catch the missing fields and return a 422 Unprocessable Entity
    assert response.status_code == 422
