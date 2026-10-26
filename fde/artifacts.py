"""Save/load trusted demonstration model bundles; joblib is unsafe for untrusted input."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from .modeling import ModelResult, LABELS


def write_bundle(result: ModelResult, path: str | Path) -> Path:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result.pipeline,
        "features": result.features,
        "version": "demo-v1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": result.metrics,
        "labels": list(LABELS),
        "synthetic_demo_only": True,
    }
    joblib.dump(bundle, location)
    return location


def load_bundle(path: str | Path) -> dict:
    """Only load a locally generated trusted artifact, never arbitrary uploaded .joblib files."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("synthetic_demo_only") is not True:
        raise ValueError("Invalid demonstration model artifact")
    return bundle


def predict_features(bundle: dict, features: dict) -> dict:
    expected = bundle["features"]
    extras = set(features) - set(expected)
    if extras:
        raise ValueError(f"Unknown features: {', '.join(sorted(extras))}")
    missing = set(expected) - set(features)
    if missing:
        raise ValueError(f"Missing features: {', '.join(sorted(missing))}")
    if not all(v is None or isinstance(v, (int, float, str, bool)) for v in features.values()):
        raise ValueError("Features must be primitive JSON values")
    row = pd.DataFrame([{name: features[name] for name in expected}])
    for name in ("Age", "BMI"):
        if name in row:
            row[name] = pd.to_numeric(row[name], errors="coerce")
    proba = bundle["model"].predict_proba(row)[0]
    by_label = {label: float(prob) for label, prob in zip(bundle["model"].classes_, proba)}
    return {
        "synthetic_positive_probability": by_label.get("Positive"),
        "predicted_synthetic_label": max(by_label, key=by_label.get),
        "disclaimer": "Research demonstration only. Not a medical diagnosis, screening, or risk estimate.",
        "model_version": bundle["version"],
    }
