"""Synthetic clinical dataset generator for patient risk stratification.

Generates 1,500+ realistic patient records with demographic, clinical,
biomarker, and lifestyle features, plus realistic missingness and noise,
sorted into Low, Medium, and High risk groups.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


RISK_LABELS = ["Low", "Medium", "High"]


def generate_patient_dataset(
    n_samples: int = 1500,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a realistic patient risk dataset with >= 1,000 individuals.

    Parameters
    ----------
    n_samples : int, default=1500
        Number of patient records to generate.
    random_state : int, default=42
        Random seed for reproducibility.
    output_path : str | Path | None, optional
        If provided, saves the generated CSV to this path.

    Returns
    -------
    pd.DataFrame
        Generated patient dataset including `risk_group` target column.
    """
    rng = np.random.default_rng(random_state)

    patient_id = [f"PAT-{i:04d}" for i in range(1, n_samples + 1)]
    age = np.clip(rng.normal(50, 15, n_samples).round(0), 18, 90).astype(int)
    sex = rng.choice(["Male", "Female"], size=n_samples, p=[0.49, 0.51])
    bmi = np.clip(rng.normal(27.2, 5.2, n_samples), 16.5, 48.0).round(1)

    # Blood pressure correlated with age & BMI
    systolic_bp = (
        102
        + 0.42 * (age - 30)
        + 0.85 * (bmi - 24)
        + rng.normal(0, 11, n_samples)
    )
    systolic_bp = np.clip(systolic_bp, 88, 210).round(0)

    diastolic_bp = (
        66
        + 0.18 * (age - 30)
        + 0.45 * (bmi - 24)
        + 0.25 * (systolic_bp - 120)
        + rng.normal(0, 7, n_samples)
    )
    diastolic_bp = np.clip(diastolic_bp, 52, 130).round(0)

    # Metabolic & lipid biomarkers
    fasting_glucose = (
        82
        + 0.35 * (age - 35)
        + 1.45 * (bmi - 24)
        + rng.normal(0, 18, n_samples)
    )
    fasting_glucose = np.clip(fasting_glucose, 62, 310).round(1)

    hba1c = (
        4.8
        + 0.018 * (fasting_glucose - 85)
        + rng.normal(0, 0.35, n_samples)
    )
    hba1c = np.clip(hba1c, 4.1, 12.5).round(2)

    total_cholesterol = (
        165
        + 0.55 * (age - 30)
        + 1.3 * (bmi - 24)
        + rng.normal(0, 28, n_samples)
    )
    total_cholesterol = np.clip(total_cholesterol, 115, 340).round(0)

    hdl_cholesterol = (
        58
        - 0.65 * (bmi - 24)
        + rng.normal(0, 10, n_samples)
    )
    hdl_cholesterol = np.clip(hdl_cholesterol, 22, 95).round(0)

    resting_heart_rate = (
        68
        + 0.35 * (bmi - 24)
        + rng.normal(0, 9, n_samples)
    )
    resting_heart_rate = np.clip(resting_heart_rate, 48, 120).round(0)

    smoking_status = rng.choice(
        ["Never", "Former", "Current"], size=n_samples, p=[0.55, 0.25, 0.20]
    )
    physical_activity = rng.choice(
        ["High", "Moderate", "Low"], size=n_samples, p=[0.28, 0.44, 0.28]
    )
    family_history = rng.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    chronic_conditions_count = rng.choice(
        [0, 1, 2, 3, 4], size=n_samples, p=[0.42, 0.28, 0.17, 0.09, 0.04]
    )
    prior_hospitalizations = rng.poisson(
        lam=0.25 + 0.22 * chronic_conditions_count, size=n_samples
    )
    prior_hospitalizations = np.clip(prior_hospitalizations, 0, 6)

    # Latent clinical risk score combining physiological & lifestyle drivers
    smoking_pts = np.where(
        smoking_status == "Current", 1.35, np.where(smoking_status == "Former", 0.45, 0.0)
    )
    activity_pts = np.where(
        physical_activity == "Low", 0.95, np.where(physical_activity == "Moderate", 0.25, -0.35)
    )

    latent_risk = (
        0.045 * (age - 45)
        + 0.095 * (bmi - 26)
        + 0.032 * (systolic_bp - 122)
        + 0.022 * (fasting_glucose - 95)
        + 0.65 * (hba1c - 5.5)
        + 0.012 * (total_cholesterol - 195)
        - 0.028 * (hdl_cholesterol - 50)
        + 0.025 * (resting_heart_rate - 72)
        + smoking_pts
        + activity_pts
        + 0.95 * family_history
        + 0.85 * chronic_conditions_count
        + 0.55 * prior_hospitalizations
        + rng.normal(0, 0.65, n_samples)
    )

    # Sort into Low (~42%), Medium (~36%), High (~22%) risk groups
    q_low, q_high = np.quantile(latent_risk, [0.42, 0.78])
    risk_group = np.where(
        latent_risk >= q_high,
        "High",
        np.where(latent_risk >= q_low, "Medium", "Low"),
    )

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "fasting_glucose": fasting_glucose,
            "hba1c": hba1c,
            "total_cholesterol": total_cholesterol,
            "hdl_cholesterol": hdl_cholesterol,
            "resting_heart_rate": resting_heart_rate,
            "smoking_status": smoking_status,
            "physical_activity": physical_activity,
            "family_history": family_history,
            "chronic_conditions_count": chronic_conditions_count,
            "prior_hospitalizations": prior_hospitalizations,
            "risk_group": risk_group,
        }
    )

    # Inject realistic missingness (~3-5% across selected clinical columns)
    missing_cols = [
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "fasting_glucose",
        "hba1c",
        "total_cholesterol",
        "hdl_cholesterol",
        "smoking_status",
        "physical_activity",
    ]
    for col in missing_cols:
        mask = rng.random(n_samples) < 0.035
        df.loc[mask, col] = np.nan

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    return df


if __name__ == "__main__":
    data_file = Path(__file__).resolve().parents[1] / "data" / "raw" / "patient_records.csv"
    df_generated = generate_patient_dataset(n_samples=1500, output_path=data_file)
    print(f"Generated {len(df_generated)} patient records -> {data_file}")
    print(df_generated["risk_group"].value_counts())

