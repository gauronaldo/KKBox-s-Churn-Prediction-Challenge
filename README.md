# KKBox Churn Prediction

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-large--data-yellow.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blueviolet.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-teal.svg)
![Pytest](https://img.shields.io/badge/tests-pytest-blue.svg)

Production-style machine learning pipeline for the KKBox Music Streaming Churn Prediction problem. The goal is to identify subscribers likely to churn so the business can prioritize retention campaigns before revenue is lost.

## Highlights

- End-to-end reproducible pipeline: ingestion, validation, feature engineering, preprocessing, training, evaluation, interpretation, and scoring.
- Large-data ingestion with DuckDB for the 30GB `user_logs.csv` file.
- Snapshot-safe feature generation using a fixed cutoff date to reduce temporal leakage.
- Multiple model families: Logistic Regression, Random Forest, XGBoost, LightGBM, and Optuna-tuned XGBoost.
- MLflow experiment tracking for model parameters, metrics, and artifacts.
- Champion model persisted with threshold and evaluation reports.
- FastAPI scoring app for quick demo predictions from engineered feature CSV/JSON.

## Business Problem

KKBox wants to predict whether a subscriber will fail to renew. A useful model should rank high-risk users well enough for targeted intervention, not just optimize raw accuracy on an imbalanced dataset.

Primary modeling metric: **AUC-PR**, because churn is the minority class and precision-recall behavior matters more than ROC alone for campaign targeting.

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
|   `-- app.py        # FastAPI demo scoring app
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

Run the FastAPI demo app:

```powershell
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The app expects rows with the engineered feature schema, such as columns from `data/processed/feature_frame.parquet`. This is intentional: online scoring should use the same preprocessor and feature contract that the model was trained on.

API examples:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict-csv -ContentType "text/csv" -InFile sample_features.csv
```

```json
POST /predict
{
  "records": [
    {
      "msno": "example_user",
      "auto_renew_rate": 1.0,
      "trans_count": 3,
      "total_spend": 447.0
    }
  ]
}
```

For JSON scoring, omitted feature columns are handled by the fitted preprocessor/model alignment where possible, but production scoring should send the full engineered schema.

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


