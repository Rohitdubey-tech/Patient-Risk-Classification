"""Preprocessing utilities and sklearn-compatible pipeline builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.feature_engineering import (
    ENGINEERED_NUMERIC_FEATURES,
    ClinicalFeatureEngineer,
)


RISK_ORDER = ["Low", "Medium", "High"]
LABEL_TO_INT = {label: idx for idx, label in enumerate(RISK_ORDER)}
INT_TO_LABEL = {idx: label for idx, label in enumerate(RISK_ORDER)}

NUMERIC_RAW_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "hba1c",
    "total_cholesterol",
    "hdl_cholesterol",
    "resting_heart_rate",
    "family_history",
    "chronic_conditions_count",
    "prior_hospitalizations",
]

CATEGORICAL_RAW_FEATURES = [
    "sex",
    "smoking_status",
    "physical_activity",
]

ALL_RAW_FEATURES = NUMERIC_RAW_FEATURES + CATEGORICAL_RAW_FEATURES
ALL_NUMERIC_FEATURES = NUMERIC_RAW_FEATURES + ENGINEERED_NUMERIC_FEATURES


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Impute missing values while preserving DataFrame structure and column names."""

    def __init__(
        self,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ):
        self.numeric_cols = numeric_cols or NUMERIC_RAW_FEATURES
        self.categorical_cols = categorical_cols or CATEGORICAL_RAW_FEATURES

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_medians_ = {
            col: float(X[col].median()) for col in self.numeric_cols
        }
        self.categorical_modes_ = {
            col: X[col].mode(dropna=True).iloc[0] for col in self.categorical_cols
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[self.numeric_cols + self.categorical_cols].copy()
        for col, val in self.numeric_medians_.items():
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val)
        for col, val in self.categorical_modes_.items():
            df[col] = df[col].fillna(val).astype(str)
        return df


def build_feature_pipeline() -> Pipeline:
    """Build the end-to-end preprocessing + feature engineering pipeline."""
    column_encoder = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), ALL_NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_RAW_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("imputer", DataFrameImputer()),
            ("feature_engineer", ClinicalFeatureEngineer()),
            ("encoder", column_encoder),
        ]
    )


def encode_target(y: pd.Series) -> np.ndarray:
    """Encode string risk labels ('Low', 'Medium', 'High') to integers (0, 1, 2)."""
    return y.map(LABEL_TO_INT).to_numpy(dtype=int)


def decode_target(y_int: np.ndarray) -> list[str]:
    """Decode integer risk predictions (0, 1, 2) back to string labels."""
    return [INT_TO_LABEL[int(val)] for val in y_int]

