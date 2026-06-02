# KKBox Churn Prediction

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-large--data-yellow.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blueviolet.svg)
![Pytest](https://img.shields.io/badge/tests-pytest-blue.svg)

Production-style machine learning pipeline for the KKBox Music Streaming Churn Prediction problem. The goal is to identify subscribers likely to churn so the business can prioritize retention campaigns before revenue is lost.

## Highlights

- End-to-end reproducible pipeline: ingestion, validation, feature engineering, preprocessing, training, evaluation, interpretation, and scoring.
- Large-data ingestion with DuckDB for the 30GB `user_logs.csv` file.
- Snapshot-safe feature generation using a fixed cutoff date to reduce temporal leakage.
- Multiple model families: Logistic Regression, Random Forest, XGBoost, LightGBM, and Optuna-tuned XGBoost.
- MLflow experiment tracking for model parameters, metrics, and artifacts.
- Champion model persisted with threshold and evaluation reports.

## Business Problem

KKBox wants to predict whether a subscriber will fail to renew. A useful model should rank high-risk users well enough for targeted intervention, not just optimize raw accuracy on an imbalanced dataset.

Primary modeling metric: **AUC-PR**, because churn is the minority class and precision-recall behavior matters more than ROC alone for campaign targeting.

## Dataset

The project uses the KKBox churn competition data stored locally in `data/raw/`.
The raw files are relational and have different grains:

| File | Grain | Local size | Notes |
|---|---|---:|---|
| `train.csv` | one labeled user row | 46.7 MB | 992,931 users with target `is_churn` |
| `members_v3.csv` | one member profile row | 427.9 MB | 6,769,473 users with demographics and registration metadata |
| `transactions.csv` | subscription transaction events | 1.73 GB | payment, renewal, cancellation, plan, and expiry history |
| `user_logs.csv` | daily listening behavior events | 30.5 GB | largest table; daily play-count and listening-duration logs |

Because `user_logs.csv` is too large for comfortable in-memory pandas processing,
ingestion uses DuckDB to scan CSVs with column pruning, apply the configured
temporal cutoff, and aggregate event tables to one row per user before writing
compact Parquet artifacts under `data/interim/`.

Current post-ingestion artifact shapes:

| Artifact | Shape | Purpose |
|---|---:|---|
| `transactions_summary.parquet` | 2,330,992 rows x 14 cols | user-level subscription/payment summary |
| `user_logs_summary.parquet` | 5,106,101 rows x 36 cols | user-level listening and behavior-window summary |
| `modeling_frame.parquet` | 992,931 rows x 55 cols | labeled modeling table after joining train, members, transactions, and logs |
| `feature_frame.parquet` | 992,931 rows x 136 cols | final engineered feature table before preprocessing |
| `X_train.parquet` | 695,051 rows x 125 cols | model-ready training features |
| `X_val.parquet` | 148,940 rows x 125 cols | validation features |
| `X_test.parquet` | 148,940 rows x 125 cols | held-out test features |

Temporal safety is controlled by `feature_engineering.cutoff_date` in
`config/config.yaml`. Transaction and listening events after that cutoff are
excluded before feature generation so the model does not learn from future
renewal or listening activity.


## Current Results

Active champion: `xgboost_optuna_topk`

Held-out test performance:

| Metric | Value |
|---|---:|
| AUC-ROC | 0.8909 |
| AUC-PR | 0.5251 |
| F1 | 0.4504 |
| Precision | 0.3242 |
| Recall | 0.7375 |
| Lift at top 10% | 6.42x |
| Decision threshold | 0.10 |

Validation comparison:

| Model | Val AUC-PR | Val Recall | Val Precision | Threshold |
|---|---:|---:|---:|---:|
| xgboost_optuna_topk | 0.5229 | 0.7246 | 0.3220 | 0.10 |
| xgboost_topk | 0.5048 | 0.6945 | 0.3432 | 0.65 |
| lightgbm | 0.5048 | 0.6903 | 0.3482 | 0.66 |
| xgboost | 0.5031 | 0.6937 | 0.3444 | 0.64 |
| random_forest | 0.4886 | 0.7041 | 0.3213 | 0.50 |
| logistic_regression | 0.4113 | 0.6899 | 0.2779 | 0.08 |

Reports are stored in `reports/model_comparison.csv`, `reports/test_evaluation.csv`, `reports/feature_importances.csv`, and SHAP outputs under `reports/`.

## Architecture

```text
data/raw/*.csv
        |
        v
src.data.run_ingestion
  - DuckDB aggregates transactions and user logs
  - pandas loads smaller member/label tables
        |
        v
data/interim/*.parquet
        |
        v
src.features.run_engineer
  - demographic, transaction, listening, recency, ratio, and trend features
        |
        v
src.features.run_preprocessing
  - stratified train/validation/test split
  - train-only preprocessing fit
        |
        v
src.models.run_train
  - candidate training
  - MLflow logging
  - champion selection by validation AUC-PR
        |
        v
src.models.evaluate / src.analysis.*
        |
        v
models/ + reports/
```

## Repository Layout

```text
project_churn_prediction/
|-- config/config.yaml
|-- data/
|   |-- raw/          # local Kaggle CSVs, never modified by code
|   |-- interim/      # compact aggregated parquet artifacts
|   `-- processed/    # feature frame and model-ready splits
|-- logs/
|-- mlruns/
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|   |-- analysis/
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- utils/
|-- tests/
|-- MODEL_CARD.md
|-- requirements.txt
`-- README.md
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place the KKBox files in `data/raw/`:

- `train.csv`
- `members_v3.csv`
- `transactions.csv`
- `user_logs.csv`

## Run The Full Pipeline

```powershell
python -m src.data.run_ingestion
python -m src.data.validate_artifacts
python -m src.features.run_engineer
python -m src.features.run_preprocessing
python -m src.models.run_train
python -m src.models.evaluate
python -m src.analysis.feature_importances
python -m src.analysis.shap_analysis
python -m pytest -q
```

## Scoring

Batch scoring from an engineered feature file:

```powershell
python -m src.models.score --input data/processed/feature_frame.parquet --output reports/predictions.csv
```

## Interpretability Notes

`auto_renew_rate` is currently the highest SHAP feature for the champion model. That is not automatically a bug:

- Auto-renew is a direct subscription-behavior signal and is expected to be predictive.
- The ingestion stage filters transactions to `transaction_date <= feature_engineering.cutoff_date`, so future renewal events after the snapshot are excluded.
- Related features such as `cancel_rate`, `retention_rate_from_transactions`, and `no_auto_renew_recent_usage` also rank highly, which is consistent with subscriber renewal mechanics.

Residual risk: if the competition label definition is tightly coupled to subscription renewal mechanics, transaction-derived renewal features can dominate. The right validation is an ablation experiment: retrain without `auto_renew_rate` and related auto-renew/cancel features, then compare AUC-PR, recall, lift, and SHAP stability.

Run the isolated ablation experiment:

```powershell
python -m src.analysis.auto_renew_ablation
```

The experiment writes `reports/auto_renew_ablation.csv` and does not overwrite champion model artifacts.

Latest ablation result:

| Experiment | Val AUC-PR | Val Recall | Val Precision | Lift at top 10% |
|---|---:|---:|---:|---:|
| XGBoost baseline | 0.5031 | 0.6937 | 0.3444 | 6.26x |
| XGBoost without auto-renew/cancel family | 0.4652 | 0.6986 | 0.2714 | 5.79x |

Interpretation: the renewal/cancel feature family is materially useful, but the model still retains meaningful ranking power without it. This suggests `auto_renew_rate` is a strong business signal rather than a single-feature leakage failure.


## Configuration

All key thresholds and paths live in `config/config.yaml`.

Important settings:

- `project.random_state`: reproducible split/model seed
- `feature_engineering.cutoff_date`: leak-safe feature snapshot date
- `feature_engineering.behavior_windows`: listening recency windows
- `feature_engineering.recency_decay_windows`: recency-weighted feature windows
- `ingestion.backend`: `duckdb` or `pandas`
- `ingestion.duckdb_memory_limit`: DuckDB memory cap
- `ingestion.duckdb_max_temp_directory_size`: DuckDB spill-space cap
- `modeling.champion_metric`: model selection metric
- `modeling.decision_threshold_metric`: threshold tuning objective

## Data Safety

The pipeline never writes to `data/raw/`. Generated outputs are written to:

- `data/interim/`
- `data/processed/`
- `models/`
- `reports/`
- `logs/`


