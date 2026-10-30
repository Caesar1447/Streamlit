# Synthetic Breast Cancer Data — Quality Audit and Limitations

**Portfolio release:** `fde-v2` · October 2026  
**Released dataset:** `demo_data/Breast_Cancer_Synthetic_Reviewed.csv`  
**Scope:** 12,000 artificial records, 27 fields  
**Intended use:** educational data engineering, analytics and demonstration **only**; not clinical, epidemiological or patient-care use.

## Data provenance and separation

The reviewed cohort derives from a historical **synthetic** breast cancer analysis workbook. It is not a real patient cohort. The original workbook predates the FDE v2 work; its medical values were artificially generated and have **not** been independently validated for clinical plausibility, demographic realism, subgroup fairness, or causal interpretation.

There are two repository-level synthetic data files with different roles:

- `demo_data/Breast_Cancer_Synthetic_Reviewed.csv`: **the current default** dataset for the Streamlit application's **Synthetic demo** mode, following the October 2026 `fde/data.py` update. This is the corrected 12,000-row dataset analyzed here.
- `demo_data/synthetic_demo.csv`: the **original bundled fallback example**, used only when the preferred reviewed CSV and the historical Excel workbook are unavailable and no uploaded dataset is supplied.

The loader accepts a user-supplied demonstration CSV/XLSX first; otherwise it selects the reviewed CSV, followed by the historical Excel workbook, then the bundled fallback example. The UCI mode is selected separately and does not use these synthetic files.

The **UCI Breast Cancer Wisconsin (Diagnostic)** dataset is a *separate real, historical research benchmark* with 569 samples and 30 features. Its modeling metrics must not be attributed to either synthetic dataset. See the [official dataset page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## Why we performed the audit

The initial artificial dataset included contradictory combinations. For example, many records labeled `Diagnosis = Negative` also listed a cancer stage, a cancer-specific treatment, a recurrence event or diagnosis year. These are inconsistencies in an invented dataset, **not observations from real patients**.

### Correction rules

For each of **11,928 synthetic `Negative`** records:

| Field | Public-release rule |
|---|---|
| `Cancer_Stage` | `Not applicable` |
| `Treatment_Type` | `Not applicable` |
| `Recurrence` | `Not applicable` |
| `Follow_Up_Adherence` | `Not applicable` |
| `Survival_Years` | Blank (structural non-applicability) |
| `Diagnosis_Year` | Blank (structural non-applicability) |

The **72 synthetic `Positive`** records retain their disease-specific fields unchanged. We did **not** invent more positive cases, relabel cases to improve a model score, or claim that the positive records are clinically realistic.

**Date normalization:** `date` values in the source were represented as Excel serial numbers, sometimes with fractional time-of-day values. The public CSV converts these to ISO calendar dates (`YYYY-MM-DD`) using Excel's 1900 date system (origin `1899-12-30`). Fractional time information was intentionally discarded; this is a documented transformation, not preservation of the original timestamps.

## Reproducible validation

Run from the repository root with Python 3, with no third-party packages required:

```bash
python scripts/validate_public_dataset.py demo_data/Breast_Cancer_Synthetic_Reviewed.csv
```

The script checks field names and order, 12,000 rows, unique participant IDs, expected artificial class counts, ISO date validity, and structural non-applicability for negative labels. A passing result verifies **only the specified engineering/data-integrity constraints**, not medically meaningful relationships.

### Verified release snapshot

| Statistic | Audited value |
|---|---:|
| Synthetic records | 12,000 |
| Fields | 27 |
| `Positive` synthetic labels | 72 |
| `Negative` synthetic labels | 11,928 |
| Artificial positive-label fraction | 0.60% |
| Age range | 40–44 years |
| Negative-label structural violations after cleaning | 0 (for the six checks above) |
| `date` format | `YYYY-MM-DD` |

## Excel dashboard interpretation

`Breast_Cancer_Dynamic_KPI_Portfolio.xlsx` provides an **Excel Table and formula-driven KPIs**. It includes **all 12,000 records but only 14 selected columns** needed for its dashboard and quality calculations. It is *not* a full 27-column replacement for the reviewed CSV.

A manual Excel check confirmed that changing a diagnosis label recalculates the KPI. After inspection, the changed label should be restored so that the released values remain unchanged. Excel automatic calculation must be enabled; additions need to be inside the expanding table. This demonstrates spreadsheet engineering, not clinical accuracy.

`Breast Cancer_Analysis_Dashboard.xlsx` is the *historical original*. Its KPI snapshots are not automatically recalculating. It is retained as a legacy fallback, not the current default synthetic data source. Older commits can continue to expose its original file contents and document metadata even after a file is replaced or deleted; historical repository hygiene is a separate security issue.

## Limitations and responsible claims

1. **Age distribution is unrealistic as a general cohort:** 40–44 years only.
2. **Artificial label imbalance:** the 0.60% positive fraction must not be interpreted as incidence, prevalence or individual cancer probability.
3. **Artificial correlations:** cleanup of contradictory labels does not prove valid medical or statistical relationships.
4. **No validation against real population distributions:** the source generation protocol and reference clinical distributions are unverified.
5. **No patient-level interpretation:** `BRCA_Status` and other fields are synthetic categories. They should not be used to infer actual genetics, risk, diagnosis or treatment.
6. **No clinical deployment:** dashboards, models and API endpoints are demonstrations and must not guide patient care.

## Portfolio framing

The reviewed dataset demonstrates a **transparent data-quality remediation workflow**: identify contradictions, write precise correction rules, normalize dates, publish structural validation tests, and clearly explain scientific limits. The independent UCI classification benchmark demonstrates model evaluation. These are complementary but separate portfolio accomplishments.
