"""Demonstration-only API. Add authentication and privacy review before public exposure."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from fde.artifacts import load_bundle, predict_features

app = FastAPI(title="Synthetic Breast Cancer Analytics Demo API", version="0.1.0",
              description="Educational model only. Do not send identifiable patient information.")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "demo_model.joblib"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, str | float | int | bool | None] = Field(
        description="Use exact feature names from /model/info; synthetic inputs only."
    )


@lru_cache(maxsize=2)
def _load(path_str: str) -> dict:
    return load_bundle(path_str)


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
    return {"version": bundle["version"], "features": bundle["features"],
            "synthetic_demo_only": True, "metrics": bundle["metrics"]}


@app.post("/predict")
def predict(body: PredictRequest) -> dict:
    path = model_path()
    if not path.is_file():
        raise HTTPException(503, "Model not trained; run scripts/train_model.py first")
    try:
        return predict_features(_load(str(path.resolve())), body.features)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
