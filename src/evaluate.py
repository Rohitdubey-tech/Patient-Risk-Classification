"""Evaluation helpers, feature importance extraction, and high-risk case analysis."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.preprocessing import ALL_RAW_FEATURES, INT_TO_LABEL


def extract_feature_importances(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    random_state: int = 42,
) -> pd.DataFrame:
    """Extract sorted feature importances from the fitted pipeline."""
    feature_pipe = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = list(feature_pipe.named_steps["encoder"].get_feature_names_out())

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        importances = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        perm = permutation_importance(
            pipeline,
            X_test[ALL_RAW_FEATURES],
            y_test,
            n_repeats=5,
            random_state=random_state,
            scoring="f1_macro",
        )
        importances = perm.importances_mean
        feature_names = ALL_RAW_FEATURES

    df_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp


def identify_high_risk_cases(
    pipeline: Pipeline,
    df_records: pd.DataFrame,
    output_dir: str | Path,
    high_risk_threshold: float = 0.50,
) -> pd.DataFrame:
    """Score patients and export flagged High-Risk cases for predictive disease analysis."""
    X = df_records[ALL_RAW_FEATURES]
    probs = pipeline.predict_proba(X)
    preds = pipeline.predict(X)

    scored = df_records.copy()
    scored["predicted_risk_group"] = [INT_TO_LABEL[int(p)] for p in preds]
    scored["prob_low"] = np.round(probs[:, 0], 4)
    scored["prob_medium"] = np.round(probs[:, 1], 4)
    scored["prob_high"] = np.round(probs[:, 2], 4)

    high_risk_df = (
        scored[
            (scored["predicted_risk_group"] == "High")
            | (scored["prob_high"] >= high_risk_threshold)
        ]
        .sort_values("prob_high", ascending=False)
        .reset_index(drop=True)
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    high_risk_df.to_csv(out_path / "high_risk_cases.csv", index=False)

    # Cohort summary comparing Low, Medium, High predicted risk groups
    summary_cols = [
        "age",
        "bmi",
        "systolic_bp",
        "fasting_glucose",
        "hba1c",
        "total_cholesterol",
        "hdl_cholesterol",
        "chronic_conditions_count",
        "prior_hospitalizations",
    ]
    cohort_summary = (
        scored.groupby("predicted_risk_group")[summary_cols]
        .mean(numeric_only=True)
        .round(2)
        .reindex(["Low", "Medium", "High"])
    )
    cohort_summary.to_csv(out_path / "high_risk_cohort_summary.csv")

    return high_risk_df

