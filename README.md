# KKBox-s-Churn-Prediction-Challenge

## Feature Engineering

Run the feature engineering stage after ingestion:

```bash
c:/Users/ADMIN/Documents/project_churn_prediction/venv/Scripts/python.exe -m src.features.run_feature_engineering
```

This reads `data/interim/modeling_frame.parquet` and writes:
- `data/processed/feature_frame.parquet`
- `data/processed/feature_metadata.json`