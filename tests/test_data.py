import io
import pandas as pd
import pytest
from fde.data import read_dataset


def test_csv_ingest_roundtrip():
    src = pd.DataFrame({"Diagnosis": ["Positive", "Negative"], "Age": [40, 50]})
    loaded = read_dataset(src.to_csv(index=False).encode(), "fake.csv")
    assert loaded.shape == (2, 2)


def test_excel_uses_first_sheet_if_dataset_sheet_absent():
    src = pd.DataFrame({"Diagnosis": ["Positive", "Negative"], "Age": [40, 50]})
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        src.to_excel(writer, index=False, sheet_name="Sheet1")
    loaded = read_dataset(bio.getvalue(), "fake.xlsx")
    assert loaded.loc[0, "Age"] == 40


def test_only_expected_filetypes():
    with pytest.raises(ValueError, match="Only"):
        read_dataset(b"bad-data", "file.json")
