"""Inference module to score new patient records using the trained pipeline."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import ALL_RAW_FEATURES, INT_TO_LABEL


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "patient_risk_pipeline.joblib"


def predict_patient_risk(
    records: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Predict risk groups ('Low', 'Medium', 'High') and probabilities for patients."""
    pipeline = joblib.load(model_path)
    X = records[ALL_RAW_FEATURES]
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    result = records.copy()
    result["predicted_risk_group"] = [INT_TO_LABEL[int(p)] for p in preds]
    result["prob_low"] = np.round(probs[:, 0], 4)
    result["prob_medium"] = np.round(probs[:, 1], 4)
    result["prob_high"] = np.round(probs[:, 2], 4)
    return result

