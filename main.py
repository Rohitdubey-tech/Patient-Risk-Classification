"""End-to-end runner for the Patient Risk Stratification ML Pipeline."""

from __future__ import annotations

from pathlib import Path

from src.data_generator import generate_patient_dataset
from src.evaluate import extract_feature_importances, identify_high_risk_cases
from src.train import train_and_validate
from src.visualize import generate_all_visualizations


def run_pipeline(n_samples: int = 1500, random_state: int = 42) -> None:
    """Execute the complete ML workflow from data generation to reporting."""
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "raw" / "patient_records.csv"
    model_path = root / "models" / "patient_risk_pipeline.joblib"
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"

    print(f"[1/4] Generating patient dataset ({n_samples:,} individuals)...")
    df = generate_patient_dataset(
        n_samples=n_samples,
        random_state=random_state,
        output_path=data_path,
    )

    print("[2/4] Training & validating candidate models (5-fold Stratified CV)...")
    results = train_and_validate(
        df=df,
        model_output_path=model_path,
        reports_dir=reports_dir,
        random_state=random_state,
    )

    print("\nCross-Validation Comparison:")
    print(results["comparison_df"].to_string(index=False))
    print(f"\nSelected Best Model: {results['best_model_name']}")
    print("Hold-Out Test Metrics:", results["metrics"]["test_metrics"])

    print("\n[3/4] Extracting feature importances & flagging High-Risk cases...")
    imp_df = extract_feature_importances(
        pipeline=results["best_pipeline"],
        X_test=results["X_test"],
        y_test=results["y_test"],
        random_state=random_state,
    )
    imp_df.to_csv(reports_dir / "feature_importances.csv", index=False)

    high_risk_df = identify_high_risk_cases(
        pipeline=results["best_pipeline"],
        df_records=results["test_df"],
        output_dir=reports_dir,
    )
    print(f"Flagged {len(high_risk_df)} high-risk test cases -> {reports_dir / 'high_risk_cases.csv'}")

    print("[4/4] Generating visualizations...")
    saved_figs = generate_all_visualizations(
        df=df,
        comparison_df=results["comparison_df"],
        y_test=results["y_test"],
        y_pred=results["y_pred"],
        y_proba=results["y_proba"],
        feature_importances_df=imp_df,
        figures_dir=figures_dir,
    )
    for fig_path in saved_figs:
        print(f"  Saved figure: {fig_path.name}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()

