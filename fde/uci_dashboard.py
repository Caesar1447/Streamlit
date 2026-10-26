"""Independent UCI WDBC dataset demonstration page for the FDE Streamlit app."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from fde.uci_diagnostic import evaluate_uci_diagnostic, load_uci_diagnostic


def render_uci_benchmark() -> None:
    st.title("🔬 Breast Cancer Wisconsin (Diagnostic) — Research Benchmark")
    st.caption(
        "Historical breast-mass FNA measurements • 569 records • 30 numeric features "
        "• Malignant vs benign sample classification"
    )
    st.warning(
        "Educational research benchmark only. Not a medical device, clinical diagnosis, "
        "screening test, or estimate of a person's future breast cancer risk. "
        "Do not upload or enter patient information."
    )
    st.markdown(
        "**Source:** [UCI Breast Cancer Wisconsin (Diagnostic)]"
        "(https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) "
        "(DOI: 10.24432/C5DW2B, CC BY 4.0). "
        "Distributed offline using scikit-learn's included copy of the dataset."
    )
    st.info(
        "This mode uses only the historical, published WDBC data. It never combines "
        "these features or labels with the synthetic demographic dataset."
    )
    X, y = load_uci_diagnostic()
    benign_count = int((y == 0).sum())
    malignant_count = int((y == 1).sum())
    a, b, c, d = st.columns(4)
    a.metric("Sample records", len(X))
    b.metric("Measured features", X.shape[1])
    c.metric("Benign", benign_count)
    d.metric("Malignant", malignant_count)
    st.subheader("Research dataset overview")
    st.caption(
        "Features describe cell nuclei from digitized fine-needle-aspirate images "
        "of breast masses. No demographic, BRCA, survival, or time-series claims are made."
    )
    with st.expander("Preview de-identified measurements", expanded=False):
        st.dataframe(X.head(15), use_container_width=True)
    with st.expander("Explore measurement distributions", expanded=False):
        feature = st.selectbox("Select a numeric feature", list(X.columns))
        chart = pd.DataFrame({feature: X[feature], "Diagnosis": y.map({0: "Benign", 1: "Malignant"})})
        st.bar_chart(chart.groupby("Diagnosis")[feature].mean())
        st.caption("Group averages for exploration only; not a diagnostic threshold.")

    st.subheader("Held-out model evaluation")
    st.caption(
        "80/20 stratified split (fixed random seed). Five-fold stratified cross-validation "
        "uses the training partition only. Hold-out records are not used to fit preprocessing "
        "or train the model. Positive class = malignant."
    )
    if st.button("Run UCI benchmark evaluation", type="primary", key="run_uci_eval"):
        with st.spinner("Evaluating the research benchmark..."):
            st.session_state["uci_benchmark_result"] = evaluate_uci_diagnostic(X, y)
    result = st.session_state.get("uci_benchmark_result")
    if result is None:
        st.info("Run UCI benchmark evaluation to view held-out metrics and cross-validation results.")
        return

    a, b, c, d = st.columns(4)
    for container, label in zip(
        (a, b, c, d), ("Accuracy", "Sensitivity / recall", "Specificity", "ROC-AUC")
    ):
        container.metric(label, f"{result.metrics[label]:.1%}")
    st.caption(
        f"Training: {result.train_rows} samples ({result.train_malignant} malignant) • "
        f"Untouched test: {result.test_rows} samples ({result.test_malignant} malignant) • "
        f"Majority-class baseline on test: {result.majority_baseline:.1%}"
    )
    st.dataframe(
        pd.DataFrame(result.metrics.items(), columns=["Metric", "Test score"]),
        use_container_width=True, hide_index=True,
    )
    st.markdown("**Confusion matrix** (actual rows × predicted columns)")
    st.dataframe(pd.DataFrame(
        result.confusion, index=["Actual benign", "Actual malignant"],
        columns=["Predicted benign", "Predicted malignant"],
    ), use_container_width=True)
    with st.expander("Training-only 5-fold cross-validation"):
        st.dataframe(pd.DataFrame([
            {"Metric": k, "Mean": v[0], "Std. dev.": v[1]}
            for k, v in result.cv_metrics.items()
        ]), use_container_width=True, hide_index=True)
    with st.expander("Random Forest feature importance"):
        st.caption(
            "Impurity-based importance is a model diagnostic, not an explanation of "
            "causation or a clinically meaningful feature contribution."
        )
        st.dataframe(result.importance.head(15), use_container_width=True, hide_index=True)
        st.bar_chart(result.importance.head(15).set_index("Feature")["Importance"])
    st.success("UCI benchmark evaluation complete for historical research data.")
    st.caption(
        "These results are an internal historical-dataset benchmark only. "
        "They do not establish generalization to hospitals, prospective patients, or contemporary populations."
    )
