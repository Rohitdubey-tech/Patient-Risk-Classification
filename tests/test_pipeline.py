"""Unit and integration tests for the Patient Risk Stratification ML Pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generator import RISK_LABELS, generate_patient_dataset
from src.feature_engineering import ENGINEERED_NUMERIC_FEATURES, ClinicalFeatureEngineer
from src.predict import predict_patient_risk
from src.preprocessing import (
    ALL_RAW_FEATURES,
    DataFrameImputer,
    build_feature_pipeline,
    decode_target,
    encode_target,
)
from src.train import train_and_validate


def test_dataset_generation_has_over_1000_patients():
    df = generate_patient_dataset(n_samples=1500, random_state=42)
    assert len(df) >= 1000
    assert set(df["risk_group"].unique()) == set(RISK_LABELS)
    assert df.isna().sum().sum() > 0  # Realistic missingness present


def test_preprocessing_and_feature_engineering_no_nans():
    df = generate_patient_dataset(n_samples=200, random_state=7)
    X = df[ALL_RAW_FEATURES]

    imputer = DataFrameImputer().fit(X)
    imputed = imputer.transform(X)
    assert imputed.isna().sum().sum() == 0

    engineer = ClinicalFeatureEngineer().fit(imputed)
    engineered = engineer.transform(imputed)
    for col in ENGINEERED_NUMERIC_FEATURES:
        assert col in engineered.columns
        assert engineered[col].isna().sum() == 0

    pipe = build_feature_pipeline()
    transformed = pipe.fit_transform(X)
    assert not np.isnan(transformed).any()


def test_target_encoding_roundtrip():
    s = pd.Series(["Low", "Medium", "High", "Low"])
    encoded = encode_target(s)
    assert list(encoded) == [0, 1, 2, 0]
    assert decode_target(encoded) == ["Low", "Medium", "High", "Low"]


def test_end_to_end_training_and_prediction(tmp_path):
    df = generate_patient_dataset(n_samples=300, random_state=99)
    model_path = tmp_path / "model.joblib"
    reports_dir = tmp_path / "reports"

    results = train_and_validate(
        df=df,
        model_output_path=model_path,
        reports_dir=reports_dir,
        cv_folds=3,
        random_state=99,
    )

    assert model_path.exists()
    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "model_comparison.csv").exists()
    assert results["metrics"]["test_metrics"]["accuracy"] > 0.70

    scored = predict_patient_risk(df.head(10), model_path=model_path)
    assert "predicted_risk_group" in scored.columns
    assert set(scored["predicted_risk_group"]).issubset(set(RISK_LABELS))
    prob_sums = scored[["prob_low", "prob_medium", "prob_high"]].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-2)

