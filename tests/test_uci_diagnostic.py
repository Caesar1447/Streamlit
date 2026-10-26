"""Offline tests of the separate UCI WDBC historical research benchmark."""
import numpy as np

from fde.uci_diagnostic import load_uci_diagnostic, evaluate_uci_diagnostic


def test_public_wdbc_schema_and_positive_class():
    x, y = load_uci_diagnostic()
    assert x.shape == (569, 30)
    assert not x.isna().any().any()
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 212  # malignant in the original WDBC data
    assert "Participant_ID" not in x.columns
    assert "Survival_Years" not in x.columns
    assert "Diagnosis" not in x.columns


def test_stratified_evaluation_is_reproducible():
    result = evaluate_uci_diagnostic()
    assert result.train_rows + result.test_rows == 569
    assert result.confusion.shape == (2, 2)
    assert result.confusion.sum() == result.test_rows
    assert 0 < result.test_malignant < result.test_rows
    assert set(result.cv_metrics) == {"balanced_accuracy", "roc_auc", "recall"}
    assert set(result.metrics) == {
        "Accuracy", "Balanced accuracy", "Precision (malignant)",
        "Sensitivity / recall", "Specificity", "F1 (malignant)", "ROC-AUC", "PR-AUC"
    }
    assert all(np.isfinite(v) and 0 <= v <= 1 for v in result.metrics.values())
    assert len(result.importance) == 30
