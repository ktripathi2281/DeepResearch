"""Health endpoint tests (no DB required)."""

from fastapi.testclient import TestClient

from deepresearch.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "deepresearch"
    assert "x-request-id" in response.headers
