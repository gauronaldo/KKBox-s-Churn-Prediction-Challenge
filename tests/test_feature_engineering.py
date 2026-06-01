import pandas as pd
import pytest
from src.features.engineer import engineer_features


def test_engineer_features_uses_config_cutoff_for_recency() -> None:
    config = {
        "feature_engineering": {
            "age_min": 7,
            "age_max": 70,
            "recent_days_window": 30,
            "cutoff_date": "2017-01-31",
            "log_scale_features": ["total_secs"],
        }
    }
    frame = pd.DataFrame(
        {
            "msno": ["u1"],
            "is_churn": [0],
            "bd": [25],
            "gender": ["female"],
            "city": [1],
            "registered_via": [7],
            "registration_init_time": [pd.Timestamp("2016-01-01")],
            "trans_count": [2],
            "total_spend": [200.0],
            "cancel_rate": [0.0],
            "days_since_last_transaction": [10],
            "active_days": [4],
            "total_secs": [1000.0],
            "last_transaction_date": [pd.Timestamp("2017-01-21")],
            "last_log_date": [pd.Timestamp("2017-01-30")],
            "days_since_last_log": [1],
        }
    )

    features = engineer_features(frame, config)

    assert features.loc[0, "analysis_reference_date"] == pd.Timestamp("2017-01-31")
    assert features.loc[0, "recent_usage_flag"] == 1
    assert "total_secs_log1p" in features.columns


def test_engineer_features_adds_behavior_trends_as_numeric() -> None:
    config = {
        "feature_engineering": {
            "age_min": 7,
            "age_max": 70,
            "recent_days_window": 30,
            "cutoff_date": "2017-01-31",
            "recency_decay_windows": [7, 14, 30, 90],
            "behavior_windows": [7, 30],
            "log_scale_features": [],
        }
    }
    frame = pd.DataFrame(
        {
            "msno": ["u1"],
            "is_churn": [0],
            "registration_init_time": [pd.Timestamp("2016-01-01")],
            "last_transaction_date": [pd.Timestamp("2017-01-21")],
            "last_log_date": [pd.Timestamp("2017-01-30")],
            "days_since_last_transaction": [10],
            "days_since_last_log": [1],
            "active_days": [10],
            "total_secs": [1000.0],
            "total_25": [5.0],
            "total_50": [5.0],
            "total_75": [5.0],
            "total_985": [5.0],
            "total_100": [80.0],
            "total_unq": [40.0],
            "total_spend": [149.0],
            "mean_plan_price": [149.0],
            "auto_renew_rate": [0.0],
            "active_days_7d": [2.0],
            "active_days_30d": [10.0],
            "total_secs_7d": [300.0],
            "total_secs_30d": [1000.0],
            "total_unq_7d": [12.0],
            "total_unq_30d": [40.0],
            "completion_rate_7d": [0.8],
            "completion_rate_30d": [0.6],
            "skip_rate_7d": [0.1],
            "skip_rate_30d": [0.3],
            "secs_per_active_day_30d": [100.0],
        }
    )

    features = engineer_features(frame, config)

    assert features.loc[0, "secs_7d_vs_30d_ratio"] == pytest.approx(0.3)
    assert features.loc[0, "secs_7d_vs_prior_23d_ratio"] == pytest.approx(300.0 / 700.0)
    assert features.loc[0, "unq_7d_vs_prior_23d_ratio"] == pytest.approx(12.0 / 28.0)
    assert features.loc[0, "completion_rate_delta_7d_30d"] == pytest.approx(0.2)
    assert features.loc[0, "no_auto_renew_recent_usage"] == pytest.approx(30.0)
    assert "recency_weighted_total_secs" in features.columns
    assert "recency_weighted_usage_to_spend" in features.columns
    assert pd.api.types.is_numeric_dtype(features["secs_7d_vs_30d_ratio"])
