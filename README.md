# Breast Cancer Analytics & Machine Learning Platform

**An educational data analytics and machine learning application demonstrating end-to-end software engineering, model evaluation, and deployment practices.**

> **Research and portfolio demonstration only.** This application is not a medical device and must not be used for diagnosis, clinical decision-making, or assessment of an individual's cancer risk. The default dashboard data is synthetic. A separate UCI historical research benchmark is also available. Performance in either mode does not establish clinical validity.

## Overview

The platform combines an interactive Streamlit dashboard with modular Python data-processing and machine-learning components. It is designed to demonstrate practical data ingestion, exploratory analysis, classification workflows, anomaly detection, reproducibility, and engineering documentation.

The **Synthetic demo** dataset mode includes three sections:

- **KPI Dashboard:** Explore record counts, missing values, categorical distributions, and time trends when relevant fields are available.
- **Data Exploration:** Review descriptive statistics and visualize relationships in the dataset.
- **Predictive Modeling & Anomaly Detection:** Train and evaluate a Random Forest classifier on appropriately structured **synthetic** data, inspect evaluation metrics and feature importance, and flag unusual feature patterns using Isolation Forest.

The **UCI Wisconsin Diagnostic benchmark** mode independently evaluates benign/malignant classification on 569 historical breast-mass aspirate samples with 30 measurements. It does not share features, labels, or model artifacts with the synthetic demo. See [the original UCI repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (DOI: 10.24432/C5DW2B; CC BY 4.0). The data is included with scikit-learn, requiring no runtime dataset download.

The project also includes a local demonstration API, automated tests, and container configuration. These are engineering examples, not evidence of clinical approval or production security.

## Technology

- **Application:** Python, Streamlit
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **API (local demonstration):** FastAPI
- **Engineering:** pytest, Docker, Git

## Data and model methodology

- The default demo data supplied with the project is **synthetically generated** for demonstration.
- The dashboard accepts CSV and Excel uploads intended for **synthetic demonstration datasets only**. Do not upload identifiable patient records or sensitive health information.
- For the synthetic classification task, selected predictors are declared in advance. Identifiers and outcome-related fields are excluded from model inputs.
- Preprocessing is fitted using the training portion of the data to reduce leakage risk. Where repeated participant identifiers exist, the evaluation workflow can use group-aware splitting.
- The application reports multiple holdout evaluation metrics, including accuracy, precision, sensitivity, specificity, F1, and ROC-AUC when applicable.
- Permutation importance illustrates model associations; it does not establish causal or clinical explanations.
- Survival estimates are shown only when the required event/censoring information is available.

**UCI benchmark methodology:** 80/20 stratified held-out test set; 5-fold stratified cross-validation on the training partition only; separate Random Forest pipeline. The positive class is malignant and the feature set contains 30 numerical cell-nucleus measurements. No individual diagnostic tool is exposed.

**Interpretation:** Scores on synthetic labels or historical breast-mass measurements demonstrate statistical and engineering methods; neither establishes clinical validity or future cancer risk.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the virtual environment:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

Install dependencies and launch the dashboard:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
streamlit run Dashboard.py
```

The dashboard should open locally at `http://localhost:8501`.

### Optional local API demonstration

Train a demonstration model using the bundled synthetic dataset, then start the FastAPI service:

```bash
python -m scripts.train_model
uvicorn api.main:app --reload
```

API documentation is available locally at `http://127.0.0.1:8000/docs`. The API includes `/health`, `/model/info`, and `/predict`. It is **not authenticated or approved for public exposure**. Use synthetic example inputs only, and load model artifacts solely from trusted sources.

### Optional Docker demonstration

After training the local model, run:

```bash
docker compose up --build
```

The provided Compose configuration binds the example services to the local machine. Deployment to an internet-facing environment would require a separate security and privacy review.

## Repository layout

```text
Dashboard.py          Streamlit application
fde/                  Synthetic modeling, UCI diagnostic benchmark, artifact utilities
api/                  Local FastAPI demonstration
scripts/              Synthetic data generation and model training
demo_data/            Synthetic example dataset
tests/                Automated tests
docs/                 Architecture, model, and security documentation
requirements*.txt     Python dependencies
Dockerfile*           Container configuration
compose.yaml          Local multi-service configuration
```

## Current status and roadmap

**Current status:** Development-stage educational portfolio application offering synthetic demonstrations and a separate historical UCI Diagnostic research benchmark. Existing automated tests and Docker configuration support engineering demonstrations, but clinical validation and production security have not been established.

**Planned enhancements:**

- Extend validation with repeated cross-validation, calibration assessment, and dataset-level documentation.
- Improve model explanation, monitoring, continuous integration, and secure deployment practices.

The UCI dataset is integrated only as a **historical research benchmark**. A classifier trained on breast-mass measurements must not be confused with a model predicting the future risk of developing breast cancer.

**Attribution:** Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993), *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository, DOI: 10.24432/C5DW2B, CC BY 4.0.

## Responsible use

This is a research, learning, and engineering portfolio project. It is not intended for patient care. Do not submit personally identifiable health information to the public demonstration. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and [`docs/SECURITY.md`](docs/SECURITY.md) for additional assumptions and limitations.
