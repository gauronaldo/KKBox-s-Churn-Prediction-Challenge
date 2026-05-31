# Model Card: KKBox Churn Prediction

## Intended Use

This model predicts the probability that a KKBox subscriber will churn after membership expiry. The intended business use is retention prioritization: rank users so a marketing or CRM team can target the highest-risk subscribers before revenue is lost.

## Not Intended For

- Automated adverse decisions about users.
- Explaining individual customer intent without human review.
- Production deployment without fresh data validation, monitoring, and threshold approval.

## Training Data

The project uses the WSDM 2018 KKBox churn dataset:

- `train.csv`: user-level churn labels.
- `members_v3.csv`: demographics and registration metadata.
- `transactions.csv`: subscription payment events.
- `user_logs.csv`: daily listening behavior.

Raw files are read from `data/raw/`. Pipeline outputs are written only to `data/interim/`, `data/processed/`, `models/`, `reports/`, and `logs/`.

## Features

Feature groups include transaction behavior, recency, subscription value, listening behavior, ratios, interactions, decay proxies, demographics, and missingness indicators. The feature engineering code uses a fixed snapshot cutoff from `config/config.yaml`.

## Leakage Controls

- Transactions and listening logs after `feature_engineering.cutoff_date` are excluded during ingestion.
- Preprocessing is fit on the training split only and then applied to validation/test/scoring data.
- Raw datetime columns and identifiers are dropped before modeling.
- `analysis_reference_date` is retained for auditability but excluded from model features.
- Held-out test evaluation is separate from validation-based champion selection.

## Metrics

Primary model-selection metric:

- AUC-PR, because churn is imbalanced and precision-recall performance is more informative than AUC-ROC for minority-class targeting.

Reported metrics:

- AUC-ROC
- AUC-PR
- F1
- Precision
- Recall
- Log Loss on held-out test
- Lift at configured top targeting fraction

## Decision Threshold

The binary classification threshold is configured by `modeling.decision_threshold`. Business deployment should tune this threshold against campaign capacity and retention economics rather than defaulting to `0.5`.

## Limitations

- The dataset is historical and competition-based; production behavior may drift.
- Random train/validation/test splits are suitable for this static snapshot, but a live churn system should also validate on later time windows.
- SHAP and feature importance analyses explain model behavior, not causal churn drivers.
- The pipeline can run Optuna with automatic top-K feature search (`top_k` as a hyperparameter). This improves validation search efficiency but still risks overfitting to validation if retrained too frequently without robust monitoring.

## Operational Checks Before Production

- Validate raw schema and temporal cutoff on every data refresh.
- Re-run test evaluation and compare metrics against the current baseline.
- Monitor feature distributions and prediction score distributions.
- Log model version, config version, training data snapshot, and metrics.
