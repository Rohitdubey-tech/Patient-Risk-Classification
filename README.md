# From Data to Decisions: Patient Risk Stratification ML Pipeline

An end-to-end machine learning project that stratifies **1,500+ individuals** into **Low**, **Medium**, and **High** risk groups for predictive disease analysis.

## Project Highlights

- **Patient Risk Stratification (`1,500` individuals):** Generates and processes multi-dimensional clinical, biomarker, and lifestyle records (`age`, `bmi`, `systolic_bp`, `diastolic_bp`, `fasting_glucose`, `hba1c`, `total_cholesterol`, `hdl_cholesterol`, `resting_heart_rate`, `smoking_status`, `physical_activity`, `family_history`, `chronic_conditions_count`, `prior_hospitalizations`) with realistic missingness.
- **End-to-End `scikit-learn` Pipeline (Zero Leakage):**
  - **Preprocessing (`src/preprocessing.py`):** Fold-safe median/mode imputation (`DataFrameImputer`), `RobustScaler` for numerical features, and `OneHotEncoder` for categoricals.
  - **Feature Engineering (`src/feature_engineering.py`):** `ClinicalFeatureEngineer` computes domain-specific features including `pulse_pressure`, `mean_arterial_pressure`, `cholesterol_hdl_ratio`, `metabolic_syndrome_index`, `age_chronic_burden`, `hypertension_flag`, `glycemic_risk_flag`, and `lifestyle_risk_score`.
  - **Model Training & Validation (`src/train.py`):** 5-fold Stratified Cross-Validation comparing `LogisticRegression`, `XGBoost`, `GradientBoosting`, and `RandomForest` with hold-out test evaluation.
- **Predictive Disease Analysis & Visualization (`src/evaluate.py`, `src/visualize.py`):**
  - Exports ranked high-risk cases (`reports/high_risk_cases.csv`) and cohort summaries (`reports/high_risk_cohort_summary.csv`).
  - Produces 5 publication-ready diagnostic & EDA plots in `reports/figures/`.

## Results Summary

### 5-Fold Stratified Cross-Validation

| Model | CV Accuracy | CV Macro F1 | CV Weighted F1 | CV ROC-AUC (OvR) | CV High-Risk Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression** | **0.8275** | **0.8256** | **0.8271** | **0.9462** | **0.8638** |
| XGBoost | 0.7817 | 0.7758 | 0.7825 | 0.9212 | 0.7197 |
| GradientBoosting | 0.7758 | 0.7684 | 0.7760 | 0.9154 | 0.7047 |
| RandomForest | 0.7575 | 0.7548 | 0.7582 | 0.8959 | 0.7612 |

### Hold-Out Test Set Performance (`LogisticRegression`)

- **Accuracy:** `0.8567`
- **Macro F1:** `0.8557`
- **Weighted F1:** `0.8567`
- **ROC-AUC (OvR Weighted):** `0.9658`
- **High-Risk Recall:** `0.8939`

## Project Structure

```text
├── main.py                      # End-to-end pipeline runner
├── pyproject.toml               # Pytest configuration
├── requirements.txt             # Python dependencies
├── data/
│   └── raw/
│       └── patient_records.csv  # Generated 1,500-patient dataset
├── models/
│   └── patient_risk_pipeline.joblib
├── reports/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── feature_importances.csv
│   ├── high_risk_cases.csv
│   ├── high_risk_cohort_summary.csv
│   └── figures/
│       ├── 01_risk_distribution_and_eda.png
│       ├── 02_correlation_and_engineered_features.png
│       ├── 03_model_comparison.png
│       ├── 04_confusion_matrix_and_roc.png
│       └── 05_feature_importance_and_high_risk_drivers.png
├── src/
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── predict.py
└── tests/
    └── test_pipeline.py
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full end-to-end pipeline
python3 main.py

# Run tests
pytest -v
```

