"""Usage: python -m scripts.train_model [optional source CSV/XLSX]"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from fde.data import read_dataset
from fde.modeling import train_evaluate
from fde.artifacts import write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a demonstration model from synthetic data")
    parser.add_argument("--data", type=Path, default=None, help="Optional CSV/XLSX synthetic dataset")
    parser.add_argument("--output", type=Path, default=Path("models/demo_model.joblib"))
    args = parser.parse_args()
    if args.data:
        df = read_dataset(args.data.read_bytes(), args.data.name)
    else:
        df = read_dataset()
    result = train_evaluate(df)
    write_bundle(result, args.output)
    print(json.dumps({"saved": str(args.output), "features": result.features,
                      "test_rows": result.test_rows, "metrics": result.metrics}, indent=2))


if __name__ == "__main__":
    main()
