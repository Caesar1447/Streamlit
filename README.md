# Breast Cancer Analytics | Data Engineering & Machine Learning Portfolio

**An end-to-end educational portfolio demonstrating synthetic-data quality remediation, interactive analytics, machine-learning evaluation, API engineering, Docker, and continuous integration.**

[![Portfolio CI](https://github.com/Caesar1447/Streamlit/actions/workflows/ci.yml/badge.svg?branch=fde-v2)](https://github.com/Caesar1447/Streamlit/actions/workflows/ci.yml)

**[Live Streamlit application](https://breast-cancer-ai-fde-v2.streamlit.app/)** · **[Source code (`fde-v2`)](https://github.com/Caesar1447/Streamlit/tree/fde-v2)** · **[Data-quality report](docs/PORTFOLIO_DATA_QUALITY.md)** · **[Model card](docs/MODEL_CARD.md)**

> **Research and portfolio demonstration only.** This is **not** a medical device, clinical diagnostic service, screening tool, or predictor of an individual's future breast cancer risk. Do not submit actual patient information. The synthetic demonstration and the historical UCI benchmark are separate, and neither provides evidence of clinical readiness.

## Overview

This project extends a graduate breast-cancer analytics capstone into a modular, testable application. It brings together data ingestion, data-quality checks, statistical exploration, supervised classification, anomaly detection, model evaluation, and deployment-oriented engineering practices. The goal is **transparent methodology and reproducible software**, not a claim of medical efficacy.

| Component | What the project demonstrates |
| --- | --- |
| **Streamlit application** | KPI dashboards, descriptive statistics, charts, synthetic CSV/XLSX upload, and distinct dataset modes |
| **Synthetic data quality** | Audited 12,000-row dataset, explicit correction rules, ISO-formatted dates, and automated structural checks |
| **Dynamic Excel dashboard** | Formula-driven KPI calculations with a structured Excel Table, tested by changing and restoring a diagnosis label |
| **Historical research benchmark** | Random Forest classification on the separate UCI Wisconsin Diagnostic dataset, with a held-out test set and training-only cross-validation |
| **Local API** | FastAPI health, model metadata, synthetic prediction, and UCI benchmark endpoints; JSON request monitoring and cached benchmark evaluation |
| **Engineering and operations** | pytest, Docker Compose, local container health checks, and GitHub Actions continuous integration |

## Explore the application

The **[deployed Streamlit dashboard](https://breast-cancer-ai-fde-v2.streamlit.app/)** offers two independent modes:

### Mode 1 — Reviewed synthetic data

**Default source:** [`demo_data/Breast_Cancer_Synthetic_Reviewed.csv`](demo_data/Breast_Cancer_Synthetic_Reviewed.csv)

- **12,000 synthetic records**, **27 columns**; **72 artificial positive labels** and **11,928 artificial negative labels**.
- KPI summaries, distribution views, data exploration, Random Forest model evaluation, and Isolation Forest detection of unusual *synthetic feature patterns*.
- The histogram and box plot default to **Age**, rather than the arbitrary participant identifier.
- CSV and XLSX uploads are supported for **synthetic or suitable non-identifying educational data only**.

The original synthetic workbook had internally inconsistent combinations of negative diagnoses with cancer-specific staging, treatment, and recurrence information. The reviewed CSV applies documented non-applicability rules to those fields, normalizes Excel serial dates to `YYYY-MM-DD`, and preserves all 12,000 records. An independent validation script checks the expected schema, labels, unique IDs, valid dates, and specified field rules.

**Critical limitation:** Cleaning contradictory values improves *internal consistency*, not medical realism. Ages are restricted to **40–44**, the artificial positive-label proportion is only **0.6%**, and the data-generation process has not been shown to reproduce clinical distributions. This dataset is unsuitable for risk estimation, population prevalence claims, or clinical inference.

**Synthetic model example (held-out test):**

| Metric | Result |
| --- | ---: |
| Accuracy | 89.2% |
| Positive-label recall | 14.3% |
| Positive-label precision | 0.8% |
| ROC-AUC | 59.2% |
| All-negative majority-class baseline accuracy | ~99.4% |

The synthetic classifier performs **worse than the majority-class accuracy baseline** and misses most artificial positive labels. These results are disclosed intentionally as an illustration of class imbalance, misleading accuracy, and the importance of reporting multiple metrics. They are **not** clinical prediction results.

For the correction rules, provenance, validation scope, and limitations, see [`docs/PORTFOLIO_DATA_QUALITY.md`](docs/PORTFOLIO_DATA_QUALITY.md).

### Mode 2 — UCI Wisconsin Diagnostic research benchmark

The separate [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) contains **569 historical breast-mass fine-needle-aspirate samples**, **30 numerical cell-nucleus measurements**, and diagnosis labels (**357 benign**, **212 malignant**). It does **not** contain the synthetic dataset's demographic or BRCA information.

**Evaluation protocol:** A fixed-seed **80/20 stratified split** produces **455 training** and **114 untouched held-out test samples**. **Five-fold stratified cross-validation** is confined to the training partition. The positive class is **malignant**. Preprocessing and model training do not use the held-out test partition.

#### Verified held-out Random Forest performance

| Metric | Result |
| --- | ---: |
| Accuracy | **97.37%** (111/114) |
| Malignant sensitivity / recall | **92.86%** (39/42) |
| Specificity | **100.00%** (72/72) |
| Malignant precision | **100.00%** |
| Malignant F1 | **96.30%** |
| Balanced accuracy | **96.43%** |
| ROC-AUC | **99.77%** |
| Precision–recall AUC | **99.62%** |
| Held-out majority-class accuracy baseline | **63.2%** |

**Confusion matrix — actual rows, predicted columns:**

| Actual label | Predicted benign | Predicted malignant |
| --- | ---: | ---: |
| Benign | 72 | 0 |
| Malignant | **3** | 39 |

**Interpretation:** Three malignant samples were incorrectly classified as benign. The reported performance is for one historical dataset and a particular fixed held-out split. It does **not** establish generalization to current patients, new hospitals, changing populations, or prospective clinical practice. The UCI evaluation never trains on or mixes in the synthetic records.

## Dynamic Excel KPI workbook

[`Breast_Cancer_Dynamic_KPI_Portfolio.xlsx`](Breast_Cancer_Dynamic_KPI_Portfolio.xlsx) is a **separate Excel analytics artifact** with **12,000 synthetic records and 14 selected fields**, not a replacement for the full reviewed 27-column CSV.

- Uses an Excel Table and formulas for record counts, positive/negative-label summaries, percentages, age statistics, selected categorical breakdowns, and structural-quality indicators.
- Contains **Dashboard**, **Dataset**, **Quality Checks**, and **Guide** sheets.
- Dynamic recalculation was **manually confirmed in Microsoft Excel** by changing a test label and restoring it.
- New records must be included in the expanding Excel Table, with automatic calculation enabled.

The Excel workbook and Streamlit app are distinct deliverables; editing the workbook does **not** automatically change the deployed Streamlit dataset. The historical `Breast Cancer_Analysis_Dashboard.xlsx` file is a legacy artifact with static KPI snapshots.

## Architecture and repository organization

```text
Streamlit application (Dashboard.py)
  |-- Synthetic mode
  |     |-- demo_data/Breast_Cancer_Synthetic_Reviewed.csv
  |     |-- KPI + exploratory views
  |     `-- Synthetic classification / anomaly detection
  |
  `-- UCI Wisconsin Diagnostic research mode
        |-- scikit-learn's bundled historical WDBC data
        `-- Random Forest / holdout metrics / training-only CV

Shared data and modeling modules: fde/
Tests and quality checks: tests/ + scripts/validate_public_dataset.py

Separate local demonstration API: api/
  |-- FastAPI endpoints and cached UCI evaluation
  `-- Structured request monitoring

Local deployment: Dockerfile + Dockerfile.api + compose.yaml
Continuous integration: .github/workflows/ci.yml
```

| Repository path | Purpose |
| --- | --- |
| `Dashboard.py` | Streamlit dashboard and mode selection |
| `fde/` | Data loading, model evaluation, anomaly detection, and benchmark logic |
| `api/` | Local FastAPI application and request monitoring |
| `demo_data/` | Reviewed synthetic CSV and original bundled fallback CSV |
| `scripts/` | Demo-data generation, model training, and dataset validation |
| `tests/` | Automated tests |
| `docs/` | API, architecture, model card, security guidance, and data-quality report |
| `.github/workflows/ci.yml` | CI pipeline |
| `Dockerfile`, `Dockerfile.api`, `compose.yaml` | Local container configuration |

The data-loading module prioritizes a user-uploaded demonstration file when supplied, then the **reviewed synthetic CSV**, with the legacy Excel workbook and bundled example CSV retained as fallbacks. The UCI benchmark is selected independently from the sidebar.

## Local API demonstration

The FastAPI service is **separate from the publicly hosted Streamlit interface** and is intended for local engineering demonstrations only.

| Endpoint | Description |
| --- | --- |
| `GET /health` | Service health |
| `GET /model/info` | Synthetic model metadata |
| `POST /predict` | Prediction with the locally trained synthetic demonstration model |
| `GET /benchmarks/uci/summary` | Published research benchmark summary |
| `GET /benchmarks/uci/evaluation` | Reproducible benchmark evaluation |

Request middleware emits structured operational logs (request ID, method, recognized route, status, and duration) without intentionally including request bodies or query strings. UCI evaluation caching avoids recomputing the full benchmark on every request.

**Security boundary:** The demo API does **not** provide authentication or a complete public-service security model. Keep it **localhost-only** and do not submit patient information. See [`docs/SECURITY.md`](docs/SECURITY.md) for guidance.

## Run the project locally

**Recommended:** Python **3.11**, matching the GitHub Actions test environment.

### Streamlit dashboard

```bash
# Run from the repository root
python -m venv .venv
```

Activate your environment using **Windows PowerShell**:

```powershell
.venv\Scripts\Activate.ps1
```

Or on **macOS/Linux**:

```bash
source .venv/bin/activate
```

Then install dependencies and launch:

```bash
python -m pip install -r requirements-dev.txt
python -m streamlit run Dashboard.py
```

The dashboard typically opens at `http://localhost:8501`.

### Automated checks

```bash
python -m compileall -q Dashboard.py fde api scripts tests
python -m pytest -q
python scripts/validate_public_dataset.py demo_data/Breast_Cancer_Synthetic_Reviewed.csv
```

The [`portfolio-ci` GitHub Actions workflow](.github/workflows/ci.yml) runs syntax verification, pytest, and reviewed-CSV validation on pushes and pull requests. The configured workflow has passed following the dataset integration and exploration-chart changes. CI success is evidence that **those configured checks** passed, not a comprehensive security audit.

### Optional local FastAPI service

First train a synthetic demonstration model, then start the API:

```bash
python -m scripts.train_model
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Swagger UI: `http://127.0.0.1:8000/docs`.

### Optional local Docker Compose setup

After preparing any model artifacts required by the demo API:

```bash
docker compose up --build -d
docker compose ps
```

The supplied Compose ports bind to **`127.0.0.1`**, not to a public network interface. The dashboard is at `http://127.0.0.1:8501` and the API docs at `http://127.0.0.1:8000/docs`.

```bash
docker compose down
```

## Limitations and responsible use

1. **Not a clinical or patient-facing service:** No claims of clinical approval, screening utility, patient-level cancer risk, prognosis, or treatment guidance.
2. **Synthetic dataset limits:** Corrected structural inconsistencies do not establish realistic disease relationships, population distributions, or fairness.
3. **Imbalanced synthetic modeling:** Low positive recall and precision are reported; the model should not be described as a successful disease predictor.
4. **Historical benchmark limits:** Strong UCI holdout scores do not substitute for prospective external validation; **three malignant cases were missed**.
5. **No personal health uploads:** Public demonstration uploads are intended for synthetic or appropriately non-identifying educational material only.
6. **Engineering maturity:** Local API, tests, health checks, and monitoring are learning/portfolio components—not audited production security or medical-device controls.

## Next steps

- Improve synthetic-data provenance and realism through independently specified generation rules and external distribution checks.
- Add automated tests explicitly covering the Streamlit default data source and chart-selection behavior.
- Expand research evaluation to calibration, robustness, and independent external datasets where legally and ethically appropriate.
- Align cloud-deployment and CI Python environments and complete repository-history/legacy-workbook metadata hygiene.
- Harden authentication, access controls, and deployment security **before** considering any public API deployment.

## Dataset attribution

Wolberg, W.; Mangasarian, O.; Street, N.; and Street, W. (1993), **Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository, DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Distributed under **CC BY 4.0**. The UCI mode uses scikit-learn's bundled copy of the historical dataset.
