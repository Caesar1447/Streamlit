"""API contract tests for fixed, read-only UCI historical research endpoints."""
from fastapi.testclient import TestClient

from api.main import app, _evaluate_uci_cached

client = TestClient(app)


def test_summary_contains_correct_public_dataset_totals():
    response = client.get("/benchmarks/uci/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["sample_count"] == 569
    assert body["feature_count"] == 30
    assert body["benign_count"] == 357
    assert body["malignant_count"] == 212
    assert body["positive_class"] == "malignant"
    assert body["research_only"] is True
    assert "clinical" in body["disclaimer"].lower()


def test_uci_evaluation_contract_and_confusion_total():
    _evaluate_uci_cached.cache_clear()
    response = client.get("/benchmarks/uci/evaluation")
    assert response.status_code == 200
    body = response.json()
    assert body["research_only"] is True
    assert body["runtime_scikit_learn_version"]
    assert body["runtime_python_version"]
    assert body["training_samples"] == 455
    assert body["held_out_samples"] == 114
    assert body["training_malignant"] == 170
    assert body["held_out_malignant"] == 42
    assert sum(body["confusion_counts"].values()) == 114
    assert set(body["held_out_scores"]) == {
        "accuracy", "balanced_accuracy", "malignant_precision",
        "malignant_recall", "benign_specificity", "malignant_f1", "roc_auc", "pr_auc"
    }
    assert set(body["cross_validation_scores"]) == {
        "balanced_accuracy", "roc_auc", "recall"
    }
    assert len(body["top_impurity_based_features"]) == 10
    assert 0 <= body["held_out_scores"]["malignant_recall"] <= 1


def test_endpoints_are_read_only_and_no_uci_individual_predict():
    assert client.post("/benchmarks/uci/evaluation", json={}).status_code == 405
    assert client.post("/benchmarks/uci/summary", json={}).status_code == 405
    assert client.post("/benchmarks/uci/predict", json={"features": {}}).status_code == 404


def test_openapi_documents_read_only_benchmark_api():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert list(paths["/benchmarks/uci/summary"]) == ["get"]
    assert list(paths["/benchmarks/uci/evaluation"]) == ["get"]
    assert "/predict" in paths  # Keep prior synthetic API intact.
