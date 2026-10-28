"""Local-only demonstration API for the breast cancer analytics portfolio.

Synthetic demo predictions are not clinical predictions. The UCI endpoints return
only aggregate information from a public historical research dataset; there is
NO per-person prediction endpoint for the UCI benchmark and no patient upload.

Do not deploy this unauthenticated demonstration API to the public internet.
"""
from __future__ import annotations

import os
import platform
from api.monitoring import install_request_monitoring
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import sklearn

from fde.artifacts import load_bundle, predict_features
from fde.uci_diagnostic import (
    DiagnosticBenchmarkResult,
    evaluate_uci_diagnostic,
    load_uci_diagnostic,
)

app = FastAPI(
    title="Breast Cancer Analytics — Educational Portfolio API",
    version="0.2.0",
    description=(
        "Local educational API. Synthetic model predictions are not clinical advice. "
        "UCI benchmark endpoints provide aggregate, historical research results "
        "only. Do not submit personal or identifiable medical data."
    ),
)
install_request_monitoring(app)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "demo_model.joblib"
UCI_SOURCE = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
UCI_DOI = "10.24432/C5DW2B"
RESEARCH_DISCLAIMER = (
    "Historical research benchmark for education only; not validated for clinical "
    "diagnosis, screening, or assessment of personal cancer risk."
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, str | float | int | bool | None] = Field(
        description="Exact feature names from /model/info; synthetic inputs only."
    )


class UCISummary(BaseModel):
    dataset: str
    source: str
    doi: str
    sample_count: int
    feature_count: int
    benign_count: int
    malignant_count: int
    positive_class: str
    research_only: bool
    disclaimer: str


class CVScore(BaseModel):
    mean: float
    standard_deviation: float


class ConfusionCounts(BaseModel):
    true_benign: int
    false_malignant: int
    missed_malignant: int
    true_malignant: int


class ImportantFeature(BaseModel):
    feature: str
    importance: float


class UCIEvaluation(BaseModel):
    dataset: str
    doi: str
    research_only: bool
    disclaimer: str
    training_samples: int
    held_out_samples: int
    training_malignant: int
    held_out_malignant: int
    split_method: str
    cross_validation_method: str
    runtime_python_version: str
    runtime_scikit_learn_version: str
    held_out_scores: dict[str, float]
    cross_validation_scores: dict[str, CVScore]
    majority_class_baseline: float
    confusion_counts: ConfusionCounts
    top_impurity_based_features: list[ImportantFeature]
    interpretation_warning: str


@lru_cache(maxsize=2)
def _load(path_str: str) -> dict:
    return load_bundle(path_str)


@lru_cache(maxsize=1)
def _evaluate_uci_cached() -> DiagnosticBenchmarkResult:
    """A reproducible, compute-once benchmark; no arbitrary user inputs."""
    return evaluate_uci_diagnostic()


def model_path() -> Path:
    return Path(os.getenv("FDE_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


@app.get("/health")
def health() -> dict:
    path = model_path()
    return {"status": "ok", "model_available": path.is_file(), "clinical_use": False}


@app.get("/model/info")
def model_info() -> dict:
    path = model_path()
    if not path.is_file():
        raise HTTPException(503, "Model not trained; run scripts/train_model.py first")
    bundle = _load(str(path.resolve()))
    return {
        "version": bundle["version"],
        "features": bundle["features"],
        "synthetic_demo_only": True,
        "metrics": bundle["metrics"],
    }


@app.post("/predict")
def predict(body: PredictRequest) -> dict:
    """Existing synthetic model endpoint; never use for real patient input."""
    path = model_path()
    if not path.is_file():
        raise HTTPException(503, "Model not trained; run scripts/train_model.py first")
    try:
        return predict_features(_load(str(path.resolve())), body.features)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/benchmarks/uci/summary", response_model=UCISummary)
def uci_summary() -> UCISummary:
    """Dataset-level metadata only; no patient-level records or uploaded data."""
    features, target = load_uci_diagnostic()
    return UCISummary(
        dataset="Breast Cancer Wisconsin (Diagnostic)",
        source=UCI_SOURCE,
        doi=UCI_DOI,
        sample_count=len(target),
        feature_count=features.shape[1],
        benign_count=int((target == 0).sum()),
        malignant_count=int((target == 1).sum()),
        positive_class="malignant",
        research_only=True,
        disclaimer=RESEARCH_DISCLAIMER,
    )


@app.get("/benchmarks/uci/evaluation", response_model=UCIEvaluation)
def uci_evaluation() -> UCIEvaluation:
    """Aggregate scores for the fixed, local, research-only benchmark."""
    result = _evaluate_uci_cached()
    tn, fp, fn, tp = (int(value) for value in result.confusion.ravel())
    metric_names = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "malignant_precision": "Precision (malignant)",
        "malignant_recall": "Sensitivity / recall",
        "benign_specificity": "Specificity",
        "malignant_f1": "F1 (malignant)",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    return UCIEvaluation(
        dataset="Breast Cancer Wisconsin (Diagnostic)",
        doi=UCI_DOI,
        research_only=True,
        disclaimer=RESEARCH_DISCLAIMER,
        training_samples=result.train_rows,
        held_out_samples=result.test_rows,
        training_malignant=result.train_malignant,
        held_out_malignant=result.test_malignant,
        split_method="80/20 stratified train/test split; random_state=42",
        cross_validation_method="5-fold stratified CV on training partition only",
        runtime_python_version=platform.python_version(),
        runtime_scikit_learn_version=sklearn.__version__,
        held_out_scores={key: result.metrics[value] for key, value in metric_names.items()},
        cross_validation_scores={
            key: CVScore(mean=mean, standard_deviation=std)
            for key, (mean, std) in result.cv_metrics.items()
        },
        majority_class_baseline=result.majority_baseline,
        confusion_counts=ConfusionCounts(
            true_benign=tn,
            false_malignant=fp,
            missed_malignant=fn,
            true_malignant=tp,
        ),
        top_impurity_based_features=[
            ImportantFeature(feature=str(row.Feature), importance=float(row.Importance))
            for row in result.importance.head(10).itertuples(index=False)
        ],
        interpretation_warning=(
            "Scores reflect internal validation on a small historical dataset. "
            "Feature importance is impurity-based, may be biased by correlated "
            "features, and does not establish causality or clinical utility."
        ),
    )
