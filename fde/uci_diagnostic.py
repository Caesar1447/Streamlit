"""UCI Breast Cancer Wisconsin (Diagnostic): separate, nonclinical benchmark.

This is the public WDBC dataset distributed with scikit-learn, originally
published by Wolberg et al. and hosted by the UCI Machine Learning Repository.
The target concerns benign/malignant breast-mass FNA characteristics; it is
NOT a personal cancer risk model and NOT intended for clinical use.
Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
DOI: 10.24432/C5DW2B; dataset license: CC BY 4.0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


@dataclass
class DiagnosticBenchmarkResult:
    metrics: dict[str, float]
    confusion: np.ndarray
    train_rows: int
    test_rows: int
    train_malignant: int
    test_malignant: int
    cv_metrics: dict[str, tuple[float, float]]
    importance: pd.DataFrame
    majority_baseline: float


def load_uci_diagnostic() -> tuple[pd.DataFrame, pd.Series]:
    """Load the UCI WDBC features (30 measurements) without patient identifiers.

    Malignant is explicitly assigned positive class 1; benign is class 0.
    scikit-learn's native target encoding is reversed (0=malignant, 1=benign).
    """
    bunch = load_breast_cancer(as_frame=True)
    features = bunch.data.copy()
    target = pd.Series((bunch.target == 0).astype(int), index=features.index, name="Malignant")
    if features.shape != (569, 30) or len(target) != 569:
        raise ValueError("Unexpected Wisconsin Diagnostic dataset shape")
    return features, target


def evaluate_uci_diagnostic(
    features: pd.DataFrame | None = None,
    target: pd.Series | None = None,
) -> DiagnosticBenchmarkResult:
    """Use a stratified, untouched hold-out with 5-fold CV on training only.

    All preprocessing is in the pipeline and fitted inside each CV fold.
    This evaluation is for software demonstration, not clinical validation.
    """
    if features is None and target is None:
        features, target = load_uci_diagnostic()
    if features is None or target is None:
        raise ValueError("Provide both features and target or neither")
    if len(features) != len(target) or features.empty:
        raise ValueError("Features and target must be nonempty and have matching row counts")
    if features.isna().all(axis=None):
        raise ValueError("No usable measurements")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
        )),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring={"balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc", "recall": "recall"},
        n_jobs=1, error_score="raise",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    metrics = {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Balanced accuracy": float(balanced_accuracy_score(y_test, pred)),
        "Precision (malignant)": float(precision_score(y_test, pred, zero_division=0)),
        "Sensitivity / recall": float(recall_score(y_test, pred, zero_division=0)),
        "Specificity": float(recall_score(y_test, pred, pos_label=0, zero_division=0)),
        "F1 (malignant)": float(f1_score(y_test, pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, proba)),
        "PR-AUC": float(average_precision_score(y_test, proba)),
    }
    importance = pd.DataFrame({
        "Feature": features.columns,
        "Importance": model.named_steps["classifier"].feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    cv_metrics = {
        k[5:]: (float(v.mean()), float(v.std()))
        for k, v in scores.items() if k.startswith("test_")
    }
    return DiagnosticBenchmarkResult(
        metrics=metrics,
        confusion=confusion_matrix(y_test, pred, labels=[0, 1]),
        train_rows=len(X_train), test_rows=len(X_test),
        train_malignant=int((y_train == 1).sum()), test_malignant=int((y_test == 1).sum()),
        cv_metrics=cv_metrics, importance=importance,
        majority_baseline=float(max(y_test.value_counts(normalize=True))),
    )
