import pandas as pd

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
