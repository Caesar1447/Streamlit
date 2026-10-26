from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint_is_explicitly_nonclinical():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["clinical_use"] is False


def test_predict_no_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("FDE_MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    response = client.post("/predict", json={"features": {"Age": 41}})
    assert response.status_code == 503
