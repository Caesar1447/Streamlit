"""Data ingestion and safety-oriented dataset validation for synthetic demonstrations."""
from __future__ import annotations

import io
from pathlib import Path
import pandas as pd

DEFAULT_XLSX = "Breast Cancer_Analysis_Dashboard.xlsx"
DEFAULT_DEMO = Path(__file__).resolve().parents[1] / "demo_data" / "synthetic_demo.csv"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # demonstration limit; hosting config may impose a lower limit
MAX_ROWS = 200_000


def read_dataset(file_bytes: bytes | None = None, filename: str | None = None,
                 repo_root: Path | None = None) -> pd.DataFrame:
    """Read an uploaded CSV/XLSX or the repository dataset, falling back to bundled demo data.

    This app is not approved to accept identifiable patient information.
    """
    root = repo_root or Path(__file__).resolve().parents[1]
    if file_bytes is not None:
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError("Dataset exceeds the 20 MB app-level demonstration limit.")
        suffix = Path(filename or "").suffix.lower()
        bio = io.BytesIO(file_bytes)
        if suffix == ".csv":
            df = pd.read_csv(bio)
        elif suffix == ".xlsx":
            # Original file uses a sheet named Dataset; uploaded files may not.
            xl = pd.ExcelFile(bio)
            sheet = "Dataset" if "Dataset" in xl.sheet_names else xl.sheet_names[0]
            df = pd.read_excel(xl, sheet_name=sheet)
        else:
            raise ValueError("Only .csv and .xlsx files are supported.")
    elif (root / DEFAULT_XLSX).exists():
        df = pd.read_excel(root / DEFAULT_XLSX, sheet_name="Dataset")
    else:
        df = pd.read_csv(root / "demo_data" / "synthetic_demo.csv")

    if df.empty:
        raise ValueError("Dataset is empty.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row limit.")
    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")
    return df


def describe_source(file_bytes: bytes | None, root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parents[1]
    if file_bytes is not None:
        return "Uploaded dataset — use synthetic or fully non-identifiable records only"
    if (root / DEFAULT_XLSX).exists():
        return "Repository Excel dataset (user-confirmed synthetic)"
    return "Bundled demonstration dataset (synthetic)"


def numeric_fields(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()
