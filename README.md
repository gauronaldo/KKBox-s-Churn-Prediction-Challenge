# KKBox Churn Prediction

> **WSDM 2018 · Binary classification · 992K users · Primary metric: AUC-PR**

Predict which KKBox subscribers will churn (not renew) within 30 days, using transaction history, app usage logs, and member demographics.

---

## Results (Baseline — no tuning)

| Model | val AUC-ROC | val AUC-PR | Lift@10% | Overfit gap |
|---|---|---|---|---|
| **XGBoost** ⭐ | **0.881** | **0.488** | **6.2×** | +0.049 |
| LightGBM | 0.881 | 0.487 | 6.2× | +0.032 |
| Random Forest | 0.871 | 0.478 | 6.0× | +0.217 ⚠️ |
| Logistic Regression | 0.429 | 0.054 | 0.7× | — |

*Baseline AUC-PR = churn rate ≈ 0.064. XGBoost is 7.6× better than random.*  
*Competition metric was Log Loss (not AUC-ROC). AUC-ROC 0.88+ is the typical range for well-engineered solutions on this dataset.*

---

## Project Structure

```
project_churn_prediction/
├── config/
│   └── config.yaml              # All hyperparameters and paths
├── data/
│   ├── raw/                     # Original CSVs (train, members, transactions, user_logs)
│   ├── interim/                 # Merged modeling frame (modeling_frame.parquet)
│   └── processed/               # Feature frame + train/val/test splits
├── models/                      # Saved model artifacts (.pkl)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_preview.ipynb # Feature engineering preview
│   └── debug_pipeline.ipynb    # End-to-end pipeline debug (Feature Eng → Preprocess → Train)
├── reports/                     # model_comparison.csv, figures
├── src/
│   ├── data/
│   │   ├── loader.py            # Raw CSV loaders
│   │   ├── merger.py            # Join members + transactions + logs
│   │   └── run_ingestion.py     # CLI: data ingestion entry point
│   ├── features/
│   │   ├── engineer.py          # Feature definitions (54 engineered features)
│   │   ├── pipeline.py          # run_feature_engineering() orchestrator
│   │   ├── preprocess.py        # ColumnGroups, build_preprocessor, split_dataset
│   │   ├── run_engineer.py      # CLI: feature engineering entry point
│   │   └── run_preprocessing.py # CLI: preprocessing entry point
│   ├── models/
│   │   ├── train.py             # All 4 model trainers + MLflow logging + champion selection
│   │   └── run_train.py         # CLI: training entry point
│   └── utils/
│       ├── config.py            # Config loader (get_value helper)
│       └── logger.py            # Structured logging setup
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Pipeline (run in order)

```bash
# Step 1 — Data ingestion (merge raw CSVs into interim frame)
python -m src.data.run_ingestion

# Step 2 — Feature engineering (54 features, cutoff 2017-01-31)
python -m src.features.run_engineer

# Step 3 — Preprocessing (OrdinalEncoder splits for tree models)
python -m src.features.run_preprocessing

# Step 4 — Training (LR, RF, XGBoost, LightGBM + MLflow)
python -m src.models.run_train
```

### 3. Debug / Inspect

Open `notebooks/debug_pipeline.ipynb` to run the full pipeline interactively with per-step diagnostics.

---

## Feature Engineering

**54 features** engineered from 3 data sources, with strict cutoff `2017-01-31` to prevent label leakage:

| Category | Features |
|---|---|
| Transaction behaviour | `trans_count`, `cancel_rate`, `auto_renew_rate`, `retention_rate_from_transactions` |
| Recency | `days_since_last_transaction`, `days_since_last_log`, `recent_transaction_flag` |
| Subscription value | `mean_plan_price`, `total_amount_paid`, `mean_payment_plan_days` |
| App engagement | `total_secs_log`, `num_unq_log`, `num_100_log`, `daily_listening_secs` |
| Demographics | `age_clean`, `gender_clean`, `city_clean`, `registration_tenure_days` |
| Missing flags | `last_log_date_missing`, `gender_missing`, `registration_init_time_missing` |

---

## Configuration

All hyperparameters live in [`config/config.yaml`](config/config.yaml):

```yaml
modeling:
  candidate_models: [logistic_regression, random_forest, xgboost, lightgbm]
  champion_metric: auc_pr       # Primary ranking metric

xgboost:
  n_estimators: 1000
  learning_rate: 0.03
  early_stopping_rounds: 100   # Stops at iter 643 on current data

lightgbm:
  metric: auc
  n_estimators: 1000
  learning_rate: 0.03
  early_stopping_rounds: 100   # Stops at iter 663 on current data
```

---

## Experiment Tracking

MLflow is configured locally. To view the UI:

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

---

## Next Steps

- [ ] Hyperparameter tuning (Optuna) for XGBoost and LightGBM
- [ ] Fix Random Forest overfitting (increase `min_samples_leaf`, reduce `max_depth`)
- [ ] Business cost evaluation (cost of false negative vs. false positive)
- [ ] Model evaluation on held-out test set
- [ ] Prediction inference pipeline