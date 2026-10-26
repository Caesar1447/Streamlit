"""Leakage-conscious training, evaluation and demonstration anomaly detection.

This project models synthetic labels, not actual cancer diagnosis or clinical risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Conservative, prespecified predictors available prior to the simulated outcome.
# Demographic attributes are excluded by default (may be audited separately).
SAFE_FEATURES = (
    "Age", "BRCA_Status", "Family_History", "Smoking_Status",
    "Alcohol_Consumption", "BMI", "Physical_Activity", "Menopause_Status",
    "Hormone_Therapy", "Genetic_Mutation", "Lifestyle_Risk",
)
TARGET = "Diagnosis"
GROUP = "Participant_ID"
LABELS = ("Negative", "Positive")


@dataclass
class ModelResult:
    pipeline: Pipeline
    metrics: dict[str, float]
    confusion: np.ndarray
    features: list[str]
    test_rows: int
    test_positive: int
    x_test: pd.DataFrame
    y_test: pd.Series
    group_split: bool
    classification_note: str


def select_features(df: pd.DataFrame) -> list[str]:
    """Never select targets, identifiers, survival outcomes or arbitrary dataset columns."""
    return [field for field in SAFE_FEATURES if field in df.columns]


def checked_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if TARGET not in df:
        raise ValueError("Dataset needs a 'Diagnosis' column with Positive/Negative labels.")
    features = select_features(df)
    if not features:
        raise ValueError("No approved baseline predictors found. Add Age or another listed pre-outcome feature.")
    labels = df[TARGET].astype("string").str.strip().str.title()
    keep = labels.isin(LABELS)
    x = df.loc[keep, features].copy()
    y = labels.loc[keep].astype(str)
    if len(y) < 40 or y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError("At least 40 records and five examples of each Positive/Negative class are required.")
    if len(y) < len(df):
        # Invalid/missing diagnoses are excluded rather than inferred.
        pass
    for name in x.columns:
        if name in ("Age", "BMI"):
            x[name] = pd.to_numeric(x[name], errors="coerce")
        else:
            x[name] = x[name].astype("string").fillna("Missing").astype(str)
    # All-empty feature columns cannot be learned meaningfully.
    features = [f for f in features if x[f].notna().any()]
    if not features:
        raise ValueError("All approved predictors are empty.")
    return x[features], y, features


def build_pipeline(x: pd.DataFrame) -> Pipeline:
    nums = [c for c in x if pd.api.types.is_numeric_dtype(x[c])]
    cats = [c for c in x if c not in nums]
    transformers = []
    if nums:
        transformers.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), nums))
    if cats:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cats))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preprocess", preprocessor),
        ("forest", RandomForestClassifier(
            n_estimators=160, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])


def train_evaluate(df: pd.DataFrame) -> ModelResult:
    x, y, features = checked_data(df)
    # Repeated Participant_IDs, if any, cannot be divided between training/test.
    use_groups = bool(GROUP in df and df.loc[x.index, GROUP].duplicated().any())
    if use_groups:
        groups = df.loc[x.index, GROUP].fillna("MISSING-ID").astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr, te = next(splitter.split(x, y, groups=groups))
        x_train, x_test = x.iloc[tr], x.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=42, stratify=y
        )
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Train/test split lacks both classes. Supply a larger, more balanced dataset.")
    pipeline = build_pipeline(x_train)
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probability = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index("Positive")]
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=list(LABELS)).ravel()
    metrics = {
        "Accuracy": float(accuracy_score(y_test, predicted)),
        "Balanced accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "Precision (positive)": float(precision_score(y_test, predicted, pos_label="Positive", zero_division=0)),
        "Sensitivity / recall": float(recall_score(y_test, predicted, pos_label="Positive", zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "F1 (positive)": float(f1_score(y_test, predicted, pos_label="Positive", zero_division=0)),
        "ROC-AUC": float(roc_auc_score((y_test == "Positive").astype(int), probability)),
        "Majority-class baseline": float(y_test.value_counts(normalize=True).max()),
    }
    return ModelResult(
        pipeline=pipeline, metrics=metrics,
        confusion=np.array([[tn, fp], [fn, tp]]), features=features,
        test_rows=len(y_test), test_positive=int((y_test == "Positive").sum()),
        x_test=x_test, y_test=y_test, group_split=use_groups,
        classification_note="Synthetic-label discrimination only; not validated for clinical use.",
    )


def feature_importance(result: ModelResult, max_rows: int = 250) -> pd.DataFrame:
    """Held-out permutation importance, not causal explanation."""
    x = result.x_test.iloc[:max_rows]
    y = result.y_test.iloc[:max_rows]
    scored = permutation_importance(result.pipeline, x, y, n_repeats=3,
                                    random_state=42, scoring="balanced_accuracy", n_jobs=1)
    return pd.DataFrame({
        "Feature": result.features,
        "Importance": scored.importances_mean,
        "Std dev": scored.importances_std,
    }).sort_values("Importance", ascending=False)


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> tuple[pd.DataFrame, str]:
    """Unsupervised unusual-feature detector; never equate results to medical abnormalities."""
    if not (0.01 <= contamination <= 0.20):
        raise ValueError("Anomaly fraction must be between 0.01 and 0.20.")
    x, _, _ = checked_data(df)
    if len(x) < 40:
        raise ValueError("Not enough rows for demonstration anomaly detection.")
    pipeline = build_pipeline(x)
    transformed = pipeline.named_steps["preprocess"].fit_transform(x)
    forest = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    marks = forest.fit_predict(transformed)
    result = df.loc[x.index].copy()
    result["Anomaly_Score"] = forest.decision_function(transformed)
    result["Flagged_Unusual"] = marks == -1
    return result[result["Flagged_Unusual"]].copy(), (
        "Unsupervised flags reflect uncommon synthetic feature patterns; "
        "they are not verified medical abnormalities."
    )
