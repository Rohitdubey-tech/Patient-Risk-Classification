"""Data visualization module for EDA, model comparison, and risk stratification."""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from src.feature_engineering import ClinicalFeatureEngineer
from src.preprocessing import ALL_RAW_FEATURES, DataFrameImputer, RISK_ORDER


RISK_PALETTE = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def generate_all_visualizations(
    df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_importances_df: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Generate and save all 5 analytical and evaluation figures."""
    _setup_style()
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # 1. Risk distribution & key clinical EDA
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Patient Cohort EDA & Risk Group Stratification (N = {:,})".format(len(df)),
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    counts = df["risk_group"].value_counts().reindex(RISK_ORDER)
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=RISK_PALETTE,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Patient Count by Risk Group")
    axes[0, 0].set_xlabel("Risk Group")
    axes[0, 0].set_ylabel("Number of Patients")
    for idx, val in enumerate(counts.values):
        axes[0, 0].text(idx, val + 12, f"{val} ({val / len(df):.1%})", ha="center", fontsize=9)

    for ax, col, title in [
        (axes[0, 1], "systolic_bp", "Systolic Blood Pressure by Risk Group"),
        (axes[1, 0], "hba1c", "HbA1c (%) by Risk Group"),
        (axes[1, 1], "bmi", "BMI Distribution by Risk Group"),
    ]:
        sns.boxplot(
            data=df,
            x="risk_group",
            y=col,
            hue="risk_group",
            order=RISK_ORDER,
            palette=RISK_PALETTE,
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Risk Group")

    plt.tight_layout()
    p1 = out_dir / "01_risk_distribution_and_eda.png"
    fig.savefig(p1)
    plt.close(fig)
    saved_paths.append(p1)

    # 2. Correlation heatmap & engineered feature separation
    imputed = DataFrameImputer().fit_transform(df[ALL_RAW_FEATURES])
    engineered = ClinicalFeatureEngineer().fit_transform(imputed)
    engineered["risk_group"] = df["risk_group"].values

    corr_cols = [
        "age",
        "bmi",
        "systolic_bp",
        "fasting_glucose",
        "hba1c",
        "mean_arterial_pressure",
        "metabolic_syndrome_index",
        "age_chronic_burden",
        "lifestyle_risk_score",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    corr = engineered[corr_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=axes[0],
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Clinical & Engineered Feature Correlations", fontweight="bold")

    sns.scatterplot(
        data=engineered,
        x="metabolic_syndrome_index",
        y="mean_arterial_pressure",
        hue="risk_group",
        hue_order=RISK_ORDER,
        palette=RISK_PALETTE,
        alpha=0.72,
        s=42,
        ax=axes[1],
    )
    axes[1].set_title(
        "Engineered Features: Metabolic Index vs Mean Arterial Pressure",
        fontweight="bold",
    )
    axes[1].set_xlabel("Metabolic Syndrome Index")
    axes[1].set_ylabel("Mean Arterial Pressure (mmHg)")

    plt.tight_layout()
    p2 = out_dir / "02_correlation_and_engineered_features.png"
    fig.savefig(p2)
    plt.close(fig)
    saved_paths.append(p2)

    # 3. Model comparison chart
    melted = comparison_df.melt(
        id_vars="model",
        value_vars=["cv_accuracy", "cv_macro_f1", "cv_roc_auc_ovr", "cv_high_risk_recall"],
        var_name="metric",
        value_name="score",
    )
    metric_labels = {
        "cv_accuracy": "Accuracy",
        "cv_macro_f1": "Macro F1",
        "cv_roc_auc_ovr": "ROC-AUC (OvR)",
        "cv_high_risk_recall": "High-Risk Recall",
    }
    melted["metric"] = melted["metric"].map(metric_labels)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax, palette="viridis")
    ax.set_ylim(0.65, 1.02)
    ax.set_title("5-Fold Stratified Cross-Validation Model Comparison", fontweight="bold")
    ax.set_xlabel("Candidate Model")
    ax.set_ylabel("Score")
    ax.legend(title="Evaluation Metric", loc="lower right")
    plt.tight_layout()
    p3 = out_dir / "03_model_comparison.png"
    fig.savefig(p3)
    plt.close(fig)
    saved_paths.append(p3)

    # 4. Confusion matrix & multi-class ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=RISK_ORDER,
        yticklabels=RISK_ORDER,
        ax=axes[0],
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("Hold-Out Test Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted Risk Group")
    axes[0].set_ylabel("Actual Risk Group")

    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    for idx, label in enumerate(RISK_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(
            fpr,
            tpr,
            lw=2.2,
            color=RISK_PALETTE[label],
            label=f"{label} Risk (AUC = {roc_auc:.3f})",
        )
    axes[1].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    axes[1].set_title("One-vs-Rest ROC Curves by Risk Group", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    p4 = out_dir / "04_confusion_matrix_and_roc.png"
    fig.savefig(p4)
    plt.close(fig)
    saved_paths.append(p4)

    # 5. Top feature importances & High-Risk probability separation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    top_feats = feature_importances_df.head(12).iloc[::-1]
    axes[0].barh(top_feats["feature"], top_feats["importance"], color="#3498db")
    axes[0].set_title("Top 12 Predictive Features Driving Risk Stratification", fontweight="bold")
    axes[0].set_xlabel("Feature Importance")

    prob_df = pd.DataFrame(
        {
            "prob_high": y_proba[:, 2],
            "actual_risk": [RISK_ORDER[int(i)] for i in y_test],
        }
    )
    sns.kdeplot(
        data=prob_df,
        x="prob_high",
        hue="actual_risk",
        hue_order=RISK_ORDER,
        palette=RISK_PALETTE,
        fill=True,
        alpha=0.35,
        common_norm=False,
        ax=axes[1],
    )
    axes[1].axvline(0.50, color="#c0392b", linestyle="--", lw=1.5, label="High-Risk Threshold (0.50)")
    axes[1].set_title("Predicted High-Risk Probability by Actual Cohort", fontweight="bold")
    axes[1].set_xlabel("Predicted P(High Risk)")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    p5 = out_dir / "05_feature_importance_and_high_risk_drivers.png"
    fig.savefig(p5)
    plt.close(fig)
    saved_paths.append(p5)

    return saved_paths

