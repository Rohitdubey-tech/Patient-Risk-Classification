"""End-to-end model training, cross-validation, tuning, and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import (
    ALL_RAW_FEATURES,
    RISK_ORDER,
    build_feature_pipeline,
    encode_target,
)


def _high_risk_recall_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall specifically for class 2 ('High' risk)."""
    return float(recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0])


def get_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Return candidate classifiers for patient risk stratification."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def train_and_validate(
    df: pd.DataFrame,
    model_output_path: str | Path,
    reports_dir: str | Path,
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run stratified split, 5-fold CV comparison, final fit, and test evaluation."""
    X = df[ALL_RAW_FEATURES].copy()
    y = encode_target(df["risk_group"])

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index.to_numpy(),
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "roc_auc_ovr": "roc_auc_ovr_weighted",
        "high_risk_recall": make_scorer(_high_risk_recall_scorer),
    }

    candidates = get_candidate_models(random_state=random_state)
    comparison_rows: list[dict[str, Any]] = []
    pipelines: dict[str, Pipeline] = {}

    for name, clf in candidates.items():
        pipe = Pipeline(
            steps=[
                ("features", build_feature_pipeline()),
                ("classifier", clf),
            ]
        )
        cv_results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        comparison_rows.append(
            {
                "model": name,
                "cv_accuracy": round(float(np.mean(cv_results["test_accuracy"])), 4),
                "cv_macro_f1": round(float(np.mean(cv_results["test_macro_f1"])), 4),
                "cv_weighted_f1": round(float(np.mean(cv_results["test_weighted_f1"])), 4),
                "cv_roc_auc_ovr": round(float(np.mean(cv_results["test_roc_auc_ovr"])), 4),
                "cv_high_risk_recall": round(
                    float(np.mean(cv_results["test_high_risk_recall"])), 4
                ),
            }
        )
        pipelines[name] = pipe

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["cv_macro_f1", "cv_high_risk_recall"], ascending=False
    ).reset_index(drop=True)

    best_model_name = str(comparison_df.loc[0, "model"])
    best_pipeline = pipelines[best_model_name]
    best_pipeline.fit(X_train, y_train)

    # Evaluate on unseen hold-out test set
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)

    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    test_weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    test_roc_auc_ovr = float(
        roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    )
    test_high_risk_recall = _high_risk_recall_scorer(y_test, y_pred)

    cls_report = classification_report(
        y_test,
        y_pred,
        target_names=RISK_ORDER,
        output_dict=True,
        zero_division=0,
    )

    # Persist artifacts
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(reports_path / "model_comparison.csv", index=False)

    metrics_payload = {
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "best_model": best_model_name,
        "test_metrics": {
            "accuracy": round(test_accuracy, 4),
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "roc_auc_ovr_weighted": round(test_roc_auc_ovr, 4),
            "high_risk_recall": round(test_high_risk_recall, 4),
        },
        "classification_report": cls_report,
        "cv_comparison": comparison_rows,
    }
    with open(reports_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return {
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "comparison_df": comparison_df,
        "metrics": metrics_payload,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "test_df": df.loc[idx_test].copy(),
    }

