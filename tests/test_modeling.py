import numpy as np
import pandas as pd
import pytest

from scripts.generate_demo import generate_synthetic
from fde.modeling import checked_data, select_features, train_evaluate, detect_anomalies
from fde.artifacts import write_bundle, load_bundle, predict_features


def test_feature_allowlist_rejects_leakage_columns():
    df = generate_synthetic(n=160)
    df["Perfect_Leakage"] = df["Diagnosis"]
    selected = select_features(df)
    assert "Diagnosis" not in selected
    assert "Survival_Years" not in selected
    assert "Participant_ID" not in selected
    assert "Gender" not in selected
    assert "Ethnicity" not in selected
    assert "Perfect_Leakage" not in selected
    assert "Age" in selected


def test_training_generates_heldout_metrics_and_artifact(tmp_path):
    df = generate_synthetic(n=400)
    result = train_evaluate(df)
    assert result.test_rows == 80
    assert result.confusion.shape == (2, 2)
    assert result.confusion.sum() == 80
    assert 0 <= result.metrics["ROC-AUC"] <= 1
    artifact = tmp_path / "demo.joblib"
    write_bundle(result, artifact)
    predicted = predict_features(load_bundle(artifact), df.loc[0, result.features].to_dict())
    assert 0 <= predicted["synthetic_positive_probability"] <= 1
    assert "Not a medical diagnosis" in predicted["disclaimer"]


def test_unknown_prediction_features_rejected(tmp_path):
    df = generate_synthetic(200)
    result = train_evaluate(df)
    path = write_bundle(result, tmp_path / "model.joblib")
    sample = df.loc[0, result.features].to_dict()
    sample["Survival_Years"] = 3
    with pytest.raises(ValueError, match="Unknown features"):
        predict_features(load_bundle(path), sample)


def test_bad_dataset_rejected():
    df = generate_synthetic(n=70)
    df = df.drop(columns=["Diagnosis"])
    with pytest.raises(ValueError, match="Diagnosis"):
        checked_data(df)


def test_anomaly_detection_flags_fraction_not_survival_cutoff():
    df = generate_synthetic(n=150)
    df["Survival_Years"] = 2.0  # If flagging survival tail, nothing would be detected.
    flagged, note = detect_anomalies(df, contamination=0.05)
    assert 3 <= len(flagged) <= 12
    assert "NOT" not in note  # nonclinical wording without relying on outcomes
    assert flagged["Flagged_Unusual"].all()


def test_group_split_keeps_participant_together():
    df = generate_synthetic(n=400)
    df.loc[1, "Participant_ID"] = df.loc[0, "Participant_ID"]
    result = train_evaluate(df)
    assert result.group_split is True
