import pandas as pd
import pytest

from src.data.validation import (
    validate_binary_target,
    validate_required_columns,
    validate_temporal_cutoff,
)


def test_validate_required_columns_raises_for_missing_column() -> None:
    frame = pd.DataFrame({"msno": ["u1"]})

    with pytest.raises(ValueError, match="Missing columns"):
        validate_required_columns(frame, ["msno", "is_churn"], "train")


def test_validate_binary_target_rejects_non_binary_values() -> None:
    frame = pd.DataFrame({"is_churn": [0, 1, 3]})

    with pytest.raises(ValueError, match="0/1"):
        validate_binary_target(frame, "is_churn")


def test_validate_temporal_cutoff_rejects_future_observations() -> None:
    frame = pd.DataFrame({"last_log_date": [pd.Timestamp("2017-02-01")]})

    with pytest.raises(ValueError, match="Temporal cutoff violation"):
        validate_temporal_cutoff(frame, ["last_log_date"], pd.Timestamp("2017-01-31"))
