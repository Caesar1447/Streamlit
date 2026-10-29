#!/usr/bin/env python3
"""Validate the reviewed educational synthetic breast cancer CSV.

Usage:
    python scripts/validate_public_dataset.py demo_data/Breast_Cancer_Synthetic_Reviewed.csv

Runs without third-party dependencies; no data or record content is transmitted.
"""

import argparse
import csv
from collections import Counter
from datetime import date
from pathlib import Path
import re
import sys

EXPECTED_COLUMNS = [
    "Participant_ID", "Age", "Gender", "Ethnicity", "BRCA_Status",
    "Family_History", "Smoking_Status", "Alcohol_Consumption",
    "BMI_Category", "Physical_Activity", "Diet_Quality", "Menarche_Age",
    "Menopause_Status", "Hormone_Therapy", "Breast_Density",
    "Benign_Breast_Disease", "Diabetes", "Cardiovascular_Disease",
    "Other_Cancers", "Diagnosis", "Cancer_Stage", "Treatment_Type",
    "Survival_Years", "Recurrence", "Follow_Up_Adherence",
    "Diagnosis_Year", "date",
]
NON_APPLICABLE = (
    "Cancer_Stage", "Treatment_Type", "Recurrence", "Follow_Up_Adherence"
)
EMPTY_FOR_NEGATIVE = ("Survival_Years", "Diagnosis_Year")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
EXPECTED_TOTAL = 12_000
EXPECTED_LABELS = {"Positive": 72, "Negative": 11_928}


def validate_csv(path: Path) -> int:
    issues = []
    seen_ids = set()
    labels = Counter()
    ages = Counter()
    date_count = 0
    count = 0

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != EXPECTED_COLUMNS:
                print("FAIL — unexpected column names or column order.")
                print(f"Expected: {EXPECTED_COLUMNS}")
                print(f"Found: {reader.fieldnames}")
                return 1
            for line_num, row in enumerate(reader, start=2):
                count += 1
                row_id = row.get("Participant_ID", "")
                if not row_id.isdigit() or int(row_id) < 1:
                    issues.append(f"Line {line_num}: invalid Participant_ID")
                if row_id in seen_ids:
                    issues.append(f"Line {line_num}: duplicate Participant_ID")
                seen_ids.add(row_id)

                age_str = row.get("Age", "")
                if not age_str.isdigit() or not (0 <= int(age_str) <= 120):
                    issues.append(f"Line {line_num}: invalid age")
                else:
                    ages[int(age_str)] += 1

                diagnosis = row.get("Diagnosis", "")
                labels[diagnosis] += 1
                if diagnosis == "Negative":
                    for col in NON_APPLICABLE:
                        if row.get(col) != "Not applicable":
                            issues.append(f"Line {line_num}: {col} must be Not applicable")
                    for col in EMPTY_FOR_NEGATIVE:
                        if row.get(col) != "":
                            issues.append(f"Line {line_num}: {col} must be blank")
                elif diagnosis != "Positive":
                    issues.append(f"Line {line_num}: invalid diagnosis label")

                raw_date = row.get("date", "")
                if not ISO_DATE.fullmatch(raw_date):
                    issues.append(f"Line {line_num}: date is not YYYY-MM-DD")
                else:
                    try:
                        date.fromisoformat(raw_date)
                        date_count += 1
                    except ValueError:
                        issues.append(f"Line {line_num}: invalid calendar date")

                if None in row:  # malformed CSV with extra delimiters
                    issues.append(f"Line {line_num}: unexpected extra columns")
    except (OSError, UnicodeError, csv.Error) as exc:
        print(f"FAIL — cannot read CSV: {exc}")
        return 1

    if count != EXPECTED_TOTAL:
        issues.append(f"Expected {EXPECTED_TOTAL:,} rows, found {count:,}")
    if dict(labels) != EXPECTED_LABELS:
        issues.append(f"Expected labels {EXPECTED_LABELS}, found {dict(labels)}")
    if len(seen_ids) != count:
        issues.append("IDs are not unique")
    if date_count != count:
        issues.append(f"Only {date_count:,}/{count:,} dates are valid ISO dates")

    print(f"Records: {count:,} | Columns: {len(EXPECTED_COLUMNS)}")
    print(f"Diagnosis labels: {dict(labels)}")
    print(f"Unique IDs: {len(seen_ids):,} | ISO dates: {date_count:,}")
    if ages:
        print(f"Age range: {min(ages)}–{max(ages)} (synthetic; not population representative)")
    if issues:
        print(f"FAIL — {len(issues)} issue(s):")
        for issue in issues[:15]:
            print(f"  - {issue}")
        if len(issues) > 15:
            print(f"  ... {len(issues) - 15} more")
        return 1
    print("PASS — schema, row counts, IDs, synthetic labels, dates, and")
    print("       negative-label non-applicability checks all passed.")
    print("NOTE: Passing structural checks does NOT establish clinical validity.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="path to reviewed synthetic CSV")
    sys.exit(validate_csv(parser.parse_args().csv_path))
