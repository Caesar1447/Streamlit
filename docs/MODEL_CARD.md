# Model card — synthetic label classification demo

- **Name:** Random Forest synthetic-label classifier
- **Version:** demo-v1
- **Intended use:** Demonstrate an end-to-end ML engineering workflow for portfolio review.
- **Not intended for:** Patient care, cancer diagnosis, cancer screening, individual health risk prediction, insurance or other high-stakes decisions.
- **Training data:** Reproducible synthetic demo CSV, or user-supplied synthetic records (never identifiable real patient data).
- **Label:** Artificial `Diagnosis` values (`Positive` / `Negative`) used solely as a programming exercise.
- **Selection:** Conservative pre-outcome allowlist; no survival outcomes, identifiers, dates or sensitive demographic attributes.
- **Preprocessing:** Numeric median imputation; categorical imputation + one-hot encoding with unknown-category tolerance, fitted only on training data.
- **Estimator:** RandomForestClassifier, fixed random seed. Single 20% holdout, stratified unless repeated participant IDs need grouping.
- **Metrics:** Accuracy, balanced accuracy, precision, sensitivity, specificity, F1, ROC-AUC and majority baseline; metric values depend on the training dataset and are *not* medical validation.
- **Interpretability:** Held-out permutation importance, not causal evidence.
- **Known limitations:** Potential synthetic generator artifacts; no external validation, cohort drift monitoring, fairness evaluation, calibration review or confidence intervals.
- **Security:** No patient data; inference API is a local unauthenticated demonstration only; joblib artifacts must be trusted.
