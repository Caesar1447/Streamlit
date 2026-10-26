"""Generate a reproducible FAKE dataset for local development and automated tests."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic(n: int = 900, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(22, 81, n)
    family = rng.choice(["No History", "One Relative", "Multiple Relatives"], n, p=[.63, .27, .10])
    brca = rng.choice(["Negative", "Positive", "Not Tested"], n, p=[.70, .09, .21])
    smoking = rng.choice(["Never Smoked", "Former Smoker", "Current Smoker"], n, p=[.60,.26,.14])
    activity = rng.choice(["Low", "Moderate", "High"], n)
    alcohol = rng.choice(["None", "Light", "Moderate", "Heavy"], n)
    bmi = np.clip(rng.normal(27, 5, n), 16, 47).round(1)
    # Fake outcome: transparently constructed from an arbitrary noisy process.
    logits = (-1.25 + .018*(age - 45) + .8*(brca=="Positive") +
              .42*(family=="Multiple Relatives") + .3*(smoking=="Current Smoker") +
              .015*(bmi-26) + rng.normal(0, .9, n))
    p = 1 / (1 + np.exp(-logits))
    diagnosis = np.where(rng.uniform(size=n) < p, "Positive", "Negative")
    years = np.round(np.maximum(.1, rng.exponential(4.0, n)), 2)
    event = rng.binomial(1, .55, n)
    df = pd.DataFrame({
        "Participant_ID": np.arange(1, n+1),
        "date": pd.date_range("2022-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Age": age, "Gender": rng.choice(["Female", "Male"], n, p=[.96, .04]),
        "Ethnicity": rng.choice(["Group A", "Group B", "Group C"], n),
        "BRCA_Status": brca, "Family_History": family,
        "Smoking_Status": smoking, "Alcohol_Consumption": alcohol,
        "Physical_Activity": activity, "BMI": bmi,
        "Diagnosis": diagnosis,
        "Survival_Years": years, "Event_Observed": event,
    })
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "demo_data" / "synthetic_demo.csv"
    out.parent.mkdir(exist_ok=True, parents=True)
    generate_synthetic().to_csv(out, index=False)
    print(f"Wrote {out}")
