# KKBox Churn Prediction

> WSDM 2018 churn prediction. Portfolio project with a leak-safe snapshot pipeline, persisted artifacts, and notebook-based sanity checks.

This repo predicts whether a KKBox subscriber will churn after membership expiry using transaction history, listening logs, and member demographics. The current codebase is organized as a reproducible pipeline from ingestion to training.

## Current Status

The current training run is stable and ready to push.

Current validation results:

| Model | val AUC-ROC | val AUC-PR | Lift@10% | Train/Val gap |
|---|---:|---:|---:|---:|
| **XGBoost** | **0.8807** | **0.4881** | **6.19x** | **+0.0494** |
| LightGBM | 0.8811 | 0.4866 | 6.23x | +0.0317 |
| Random Forest | 0.8784 | 0.4800 | 6.12x | +0.0782 |
| Logistic Regression | 0.8302 | 0.3767 | 5.51x | +0.0027 |

Notes:
- The Kaggle competition metric is Log Loss.
- This repo uses AUC-PR and AUC-ROC internally for model comparison and portfolio reporting.
- The pipeline uses a fixed cutoff date (`2017-01-31`) to avoid temporal leakage.

## Repository Layout

```text
project_churn_prediction/
├── config/
│   └── config.yaml              # Paths, split sizes, cutoff date, and model hyperparameters
├── data/
│   ├── raw/                     # Source CSVs
│   ├── interim/                 # Ingested modeling frame
│   └── processed/               # Feature frame and train/val/test splits
├── logs/                        # Runtime logs
├── models/                      # Saved model artifacts and preprocessor
├── notebooks/
│   ├── debug_pipeline.ipynb      # Full interactive pipeline notebook
│   ├── pipeline_healthcheck.ipynb# End-to-end sanity check notebook
│   └── training_debug.ipynb     # Training diagnostics notebook
├── reports/
│   └── model_comparison.csv      # Saved comparison table
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── merger.py
│   │   └── run_ingestion.py
│   ├── features/
│   │   ├── engineer.py
│   │   ├── pipeline.py
│   │   ├── preprocess.py
│   │   ├── run_engineer.py
│   │   └── run_preprocessing.py
│   ├── models/
│   │   ├── train.py
│   │   └── run_train.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── requirements.txt
└── README.md
```

## How to Run

### 1) Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Pipeline order

```bash
python -m src.data.run_ingestion
python -m src.features.run_engineer
python -m src.features.run_preprocessing
python -m src.models.run_train
```

### 3) Notebook workflow

- `notebooks/debug_pipeline.ipynb` for the full interactive pipeline and detailed diagnostics.
- `notebooks/pipeline_healthcheck.ipynb` for a compact end-to-end verification pass.
- `notebooks/training_debug.ipynb` for training-only analysis and coefficient inspection.

## Feature Engineering

The current feature pipeline builds leak-safe user-level features from the raw tables using a fixed snapshot cutoff.

Key feature groups:

| Category | Examples |
|---|---|
| Transaction behavior | `trans_count`, `cancel_rate`, `auto_renew_rate`, `retention_rate_from_transactions` |
| Recency | `days_since_last_transaction`, `days_since_last_log`, `recent_transaction_flag` |
| Subscription value | `mean_plan_price`, `total_spend`, `mean_plan_days` |
| Listening behavior | `total_secs`, `mean_secs`, `total_unq`, `mean_completion_rate` |
| Demographics | `age_clean`, `gender_clean`, `city_clean`, `member_age_days` |
| Missingness flags | `gender_missing`, `registration_init_time_missing`, `last_log_date_missing` |

## Output Locations

The scripts write artifacts to these locations:

- `data/interim/modeling_frame.parquet`
- `data/processed/feature_frame.parquet`
- `data/processed/X_train.parquet`, `X_val.parquet`, `X_test.parquet`
- `data/processed/y_train.parquet`, `y_val.parquet`, `y_test.parquet`
- `models/preprocessor.pkl`
- `models/champion_model.pkl`
- `models/champion_name.txt`
- `models/*.pkl` for each trained model
- `reports/model_comparison.csv`
- `logs/feature_engineering.log`
- `logs/training.log`

MLflow is configured for local tracking as well. If enabled, artifacts are logged under `mlruns/`.

## Configuration

All runtime controls live in `config/config.yaml`.

Important values:

- `feature_engineering.cutoff_date`: `2017-01-31`
- `split.validation_size` and `split.test_size`
- `modeling.candidate_models`
- `modeling.champion_metric`: `auc_pr`

## Notes

- The project is portfolio-oriented, but the implementation remains production-minded: fixed snapshot cutoffs, persisted artifacts, explicit logs, and notebook sanity checks.
- Logistic Regression uses a separate OneHot-based linear preprocessor, while tree models use the ordinal preprocessed splits.
- The current champion is `xgboost`; performance is balanced and the train/val gap is not showing severe overfit.

## Next Improvements

- Hyperparameter tuning for XGBoost and LightGBM
- Held-out test evaluation report
- Inference script for scoring new users
- Business-cost analysis for threshold selection
