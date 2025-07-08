"""Domain-driven clinical feature engineering transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ENGINEERED_NUMERIC_FEATURES = [
    "pulse_pressure",
    "mean_arterial_pressure",
    "cholesterol_hdl_ratio",
    "metabolic_syndrome_index",
    "age_chronic_burden",
    "hypertension_flag",
    "glycemic_risk_flag",
    "lifestyle_risk_score",
]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-specific cardiovascular and metabolic risk features.

    Designed to operate on a DataFrame with imputed clinical columns and
    remain 100% compatible with scikit-learn Pipelines (zero data leakage).
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401
        self.input_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        sbp = df["systolic_bp"].astype(float)
        dbp = df["diastolic_bp"].astype(float)
        bmi = df["bmi"].astype(float)
        glucose = df["fasting_glucose"].astype(float)
        hba1c = df["hba1c"].astype(float)
        total_chol = df["total_cholesterol"].astype(float)
        hdl_chol = np.clip(df["hdl_cholesterol"].astype(float), 1.0, None)
        age = df["age"].astype(float)
        chronic = df["chronic_conditions_count"].astype(float)

        df["pulse_pressure"] = (sbp - dbp).round(2)
        df["mean_arterial_pressure"] = (dbp + (sbp - dbp) / 3.0).round(2)
        df["cholesterol_hdl_ratio"] = (total_chol / hdl_chol).round(3)
        df["metabolic_syndrome_index"] = (
            (bmi / 25.0) * (glucose / 100.0) * (hba1c / 5.7)
        ).round(3)
        df["age_chronic_burden"] = (age * (1.0 + chronic)).round(1)
        df["hypertension_flag"] = ((sbp >= 140) | (dbp >= 90)).astype(int)
        df["glycemic_risk_flag"] = ((glucose >= 126) | (hba1c >= 6.5)).astype(int)

        smoking_map = {"Never": 0.0, "Former": 1.0, "Current": 2.0}
        activity_map = {"High": 0.0, "Moderate": 1.0, "Low": 2.0}
        smoking_score = df["smoking_status"].map(smoking_map).fillna(1.0)
        activity_score = df["physical_activity"].map(activity_map).fillna(1.0)
        df["lifestyle_risk_score"] = smoking_score + activity_score

        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else self.input_columns_
        return np.array(base + ENGINEERED_NUMERIC_FEATURES)

