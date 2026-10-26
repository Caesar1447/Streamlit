"""Synthetic Breast Cancer Analytics — Streamlit portfolio application.

Educational demonstration only: NOT for patient care, diagnosis or risk assessment.
"""
from __future__ import annotations
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from fde.data import read_dataset, describe_source, numeric_fields
from fde.modeling import train_evaluate, feature_importance, detect_anomalies


st.set_page_config(page_title="Breast Cancer Analytics | FDE Portfolio", page_icon="🔬", layout="wide")
st.title("🔬 Synthetic Breast Cancer Analytics")
st.caption("End-to-end data engineering and ML demonstration • Synthetic data only • Not a clinical product")
st.warning(
    "Educational research portfolio only. Predictions are about synthetic labels, not actual breast "
    "cancer diagnoses or personal medical risk. Do not upload real patient or identifying information."
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", [
    "📊 KPI Dashboard", "📈 Data Exploration", "🚀 Predictive Modeling & Anomaly Detection",
])
st.sidebar.divider()
st.sidebar.caption("Portfolio build 0.1 · Evaluation, testing, API, Docker and CI examples included in the repository.")

uploaded = st.file_uploader("Upload a SYNTHETIC dataset (CSV or Excel)", type=["csv", "xlsx"],
                            help="Never upload identifiable or sensitive real patient records.")
content = uploaded.getvalue() if uploaded is not None else None
try:
    df = read_dataset(content, uploaded.name if uploaded else None)
except (ValueError, OSError, KeyError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(f"Cannot read dataset: {exc}")
    st.stop()

st.info(f"**Dataset source:** {describe_source(content)} · **Rows:** {len(df):,} · **Columns:** {len(df.columns)}")
with st.expander("Preview dataset", expanded=page == "📊 KPI Dashboard"):
    st.dataframe(df.head(15), use_container_width=True)


if page == "📊 KPI Dashboard":
    st.header("KPI Dashboard")
    columns = st.columns(3)
    columns[0].metric("Records", f"{len(df):,}")
    columns[1].metric("Variables", str(len(df.columns)))
    columns[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.markdown("### Demographic and categorical distribution")
    categories = [k for k in df.select_dtypes(include=["object", "string", "category"]).columns
                  if k not in {"Participant_ID", "date"}]
    if categories:
        col = st.selectbox("Choose category", categories)
        counts = df[col].astype("string").fillna("Missing").value_counts().head(15)
        st.bar_chart(counts)
    else:
        st.info("No suitable categorical columns in this dataset.")

    if "date" in df:
        st.markdown("### Trends over time")
        parsed = pd.to_datetime(df["date"], errors="coerce")
        valid = df.loc[parsed.notna()].copy()
        valid["date"] = parsed.loc[parsed.notna()]
        nums = [col for col in numeric_fields(valid)
                if col not in {"Participant_ID", "Diagnosis", "Event_Observed"}]
        if len(valid) and nums:
            choice = st.multiselect("Numeric indicators", nums, default=nums[:1])
            if choice:
                trend = valid.set_index("date").sort_index().resample("ME")[choice].mean()
                st.line_chart(trend)
                st.download_button("Download trend CSV", trend.to_csv().encode(),
                                   "monthly_kpi_trends.csv", mime="text/csv")
        else:
            st.info("Trend chart needs valid dates and numeric indicators.")
    else:
        st.caption("No date field supplied; time-series analysis is hidden rather than fabricated.")

elif page == "📈 Data Exploration":
    st.header("Data Exploration")
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose().fillna(""), use_container_width=True)
    numeric = numeric_fields(df)
    if numeric:
        with st.expander("Correlation heatmap", expanded=True):
            # Prevent a giant unreadable chart on wide schemas.
            selected = st.multiselect("Columns for correlation", numeric,
                                      default=numeric[:min(12, len(numeric))])
            if len(selected) >= 2:
                fig, ax = plt.subplots(figsize=(max(7, len(selected) * .75), 6))
                sns.heatmap(df[selected].corr(numeric_only=True), ax=ax,
                            cmap="coolwarm", center=0, annot=len(selected) <= 9, fmt=".2f")
                st.pyplot(fig)
                plt.close(fig)
        a, b = st.columns(2)
        with a:
            st.subheader("Histogram")
            chosen = st.selectbox("Numeric variable", numeric, key="hist")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(pd.to_numeric(df[chosen], errors="coerce").dropna(), bins=24, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        with b:
            st.subheader("Box plot")
            chosen_box = st.selectbox("Numeric variable", numeric, key="box")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(y=pd.to_numeric(df[chosen_box], errors="coerce"), ax=ax)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("No numeric fields are available.")

    st.subheader("Survival analysis (only with genuine event/censoring information)")
    if {"Survival_Years", "Event_Observed", "Diagnosis"}.issubset(df.columns):
        from lifelines import KaplanMeierFitter
        candidate = df.loc[df["Diagnosis"].astype(str).str.strip().str.title() == "Positive"].copy()
        candidate["Survival_Years"] = pd.to_numeric(candidate["Survival_Years"], errors="coerce")
        candidate["Event_Observed"] = pd.to_numeric(candidate["Event_Observed"], errors="coerce")
        candidate = candidate.dropna(subset=["Survival_Years", "Event_Observed"])
        candidate = candidate.loc[(candidate["Survival_Years"] >= 0) &
                                  (candidate["Event_Observed"].isin([0, 1]))]
        if len(candidate) >= 10:
            st.caption("Event_Observed: 1 = event observed, 0 = right-censored. Synthetic example only.")
            kmf = KaplanMeierFitter()
            kmf.fit(candidate["Survival_Years"], event_observed=candidate["Event_Observed"])
            fig, ax = plt.subplots(figsize=(8, 4))
            kmf.plot_survival_function(ax=ax)
            ax.set(xlabel="Synthetic follow-up time (years)", ylabel="Estimated survival",
                   title="Demonstration Kaplan–Meier curve")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Insufficient valid event and censoring information.")
    else:
        st.info("Survival analysis is disabled until both Survival_Years and Event_Observed are present, "
                "with valid censoring semantics. Diagnosis alone does not establish an event.")

    report = io.BytesIO()
    with pd.ExcelWriter(report, engine="xlsxwriter") as writer:
        df.describe(include="all").transpose().to_excel(writer, sheet_name="Summary")
        if len(numeric) >= 2:
            df[numeric].corr().to_excel(writer, sheet_name="Correlations")
    st.download_button("Download analysis workbook", report.getvalue(), "synthetic_analysis.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.header("Predictive Modeling & Anomaly Detection")
    st.caption("The classifier is trained on the supplied synthetic data only. "
               "Sensitive demographics and post-outcome variables are excluded by default.")
    st.subheader("Model validation")
    if st.button("Train and evaluate Random Forest", type="primary"):
        with st.spinner("Preparing a leakage-conscious train/test evaluation..."):
            try:
                st.session_state["result"] = train_evaluate(df)
                st.session_state["source_id"] = (uploaded.name if uploaded else "default", len(df), tuple(df.columns))
            except ValueError as exc:
                st.session_state.pop("result", None)
                st.error(f"Model evaluation could not complete: {exc}")

    current_id = (uploaded.name if uploaded else "default", len(df), tuple(df.columns))
    result = st.session_state.get("result") if st.session_state.get("source_id") == current_id else None
    if result is not None:
        st.caption(f"Held-out test rows: {result.test_rows:,} · Positive: {result.test_positive:,} · "
                   f"Group-aware splitting: {'Yes' if result.group_split else 'Not needed'}")
        st.write("**Included pre-outcome predictors:** " + ", ".join(result.features))
        cards = st.columns(4)
        for col, title in zip(cards, ["Accuracy", "Sensitivity / recall", "Specificity", "ROC-AUC"]):
            col.metric(title, f"{result.metrics[title]:.1%}")
        metric_df = pd.DataFrame(result.metrics.items(), columns=["Metric", "Held-out score"])
        st.dataframe(metric_df.style.format({"Held-out score": "{:.3f}"}), hide_index=True,
                     use_container_width=True)
        st.caption("A high synthetic-label score may reflect how the fake labels were generated. "
                   "It does not indicate medical diagnostic accuracy or generalization to real patients.")
        st.write("**Confusion matrix** (rows = actual synthetic label, columns = predicted synthetic label)")
        st.dataframe(pd.DataFrame(result.confusion,
                                 index=["Actual negative", "Actual positive"],
                                 columns=["Predicted negative", "Predicted positive"]))
        with st.expander("Explainability: held-out permutation importance"):
            st.caption("Importance is the model's predictive reliance on each feature in held-out synthetic data; "
                       "not proof of causation or clinical validity.")
            if st.button("Calculate permutation importance"):
                importance = feature_importance(result)
                st.dataframe(importance, hide_index=True, use_container_width=True)
                st.bar_chart(importance.set_index("Feature")["Importance"])
        st.success("Evaluation complete for the selected synthetic dataset.")
    else:
        st.info("Select **Train and evaluate Random Forest** to run a held-out assessment.")

    st.divider()
    st.subheader("Unsupervised anomaly detection")
    st.caption("Isolation Forest flags unusual combinations of approved pre-outcome features. "
               "These flags are NOT medical abnormalities.")
    fraction = st.slider("Expected fraction of unusual synthetic records", 0.01, 0.20, 0.05, 0.01)
    if st.button("Identify unusual records"):
        try:
            with st.spinner("Fitting unsupervised detector..."):
                anomalies, interpretation = detect_anomalies(df, fraction)
            st.metric("Records flagged", f"{len(anomalies):,}")
            st.info(interpretation)
            st.dataframe(anomalies.head(200), use_container_width=True)
            st.download_button("Download flags as CSV", anomalies.to_csv(index=False).encode(),
                               "synthetic_anomaly_flags.csv", mime="text/csv")
        except ValueError as exc:
            st.error(str(exc))

st.divider()
st.caption("Research/portfolio demonstration. Do not make clinical decisions from this app or upload patient information.")
