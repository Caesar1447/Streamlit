"""Regression tests for the FDE API's privacy-conscious request monitoring.

Place this file at tests/test_monitoring.py in the Streamlit-fde-v2 project.
Run: python -m pytest -q tests/test_monitoring.py
"""

import json
import re
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import monitoring


@pytest.fixture
def events(monkeypatch):
    """Capture structured events without writing extra test data to stdout."""
    captured = []
    monkeypatch.setattr(monitoring._LOG, "info", lambda line: captured.append(json.loads(line)))
    return captured


def make_test_app():
    app = FastAPI()
    monitoring.install_request_monitoring(app)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict", status_code=202)
    def predict():
        return {"accepted": True}

    @app.get("/fails")
    def fails():
        raise ValueError("private medical information must not reach logs")

    return app


def test_health_request_has_timing_and_request_id_headers(events):
    with TestClient(make_test_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    assert float(response.headers["X-Response-Time-Ms"]) >= 0

    assert len(events) == 1
    event = events[0]
    assert event["event"] == "http_request"
    assert event["method"] == "GET"
    assert event["endpoint"] == "/health"
    assert event["status_code"] == 200
    assert event["request_id"] == request_id
    assert event["duration_ms"] >= 0
    timestamp = datetime.fromisoformat(event["timestamp"])
    assert timestamp.utcoffset() == timedelta(0)


def test_request_body_headers_and_query_string_are_not_logged(events):
    secret = "DO_NOT_LOG_PRIVATE_TEST_VALUE_7788"
    with TestClient(make_test_app()) as client:
        response = client.post(
            f"/predict?patient={secret}",
            json={"confidential": secret},
            headers={"X-Private-Test": secret},
        )

    assert response.status_code == 202
    assert len(events) == 1
    event = events[0]
    assert event["endpoint"] == "/predict"
    assert event["method"] == "POST"
    assert event["status_code"] == 202
    assert secret not in json.dumps(event)
    assert "patient=" not in json.dumps(event)


def test_unknown_endpoint_is_redacted_even_for_404(events):
    with TestClient(make_test_app()) as client:
        response = client.get("/private/Patient_ABC_123")

    assert response.status_code == 404
    assert len(events) == 1
    assert events[0]["endpoint"] == "[redacted]"
    assert events[0]["status_code"] == 404
    assert "Patient_ABC_123" not in json.dumps(events[0])


def test_exception_logs_only_exception_type_not_message(events):
    with TestClient(make_test_app(), raise_server_exceptions=False) as client:
        response = client.get("/fails")

    assert response.status_code == 500
    assert len(events) == 1
    event = events[0]
    assert event["endpoint"] == "[redacted]"
    assert event["status_code"] == 500
    assert event["error_type"] == "ValueError"
    assert "private medical information" not in json.dumps(event)


def test_request_ids_are_unique(events):
    with TestClient(make_test_app()) as client:
        first = client.get("/health")
        second = client.get("/health")

    assert len(events) == 2
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
    assert events[0]["request_id"] == first.headers["X-Request-ID"]
    assert events[1]["request_id"] == second.headers["X-Request-ID"]
