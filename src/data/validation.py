"""Data validation checks for KKBox churn pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from src.utils.config import get_value


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: list[str] | tuple[str, ...],
    table_name: str,
) -> None:
    """Validate that a DataFrame contains all required columns."""

    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {table_name}: {missing}")


def validate_binary_target(frame: pd.DataFrame, target_col: str) -> None:
    """Validate that the target exists, is non-null, and contains only 0/1."""

    validate_required_columns(frame, [target_col], "target frame")
    if frame[target_col].isna().any():
        raise ValueError(f"{target_col} contains null values.")
    values = set(frame[target_col].dropna().unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"{target_col} must contain only 0/1 values.")


def validate_temporal_cutoff(
    frame: pd.DataFrame,
    date_columns: list[str],
    cutoff_date: pd.Timestamp,
) -> None:
    """Ensure observed event dates do not extend beyond the leakage cutoff."""

    violations: dict[str, int] = {}
    for column in date_columns:
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce")
        count = int((values > cutoff_date).sum())
        if count:
            violations[column] = count

    if violations:
        raise ValueError(
            "Temporal cutoff violation. Rows after cutoff "
            f"{cutoff_date.date()}: {violations}"
        )


def validate_raw_schema(
    frames: Mapping[str, pd.DataFrame],
    config: Mapping,
) -> None:
    """Validate required raw-table schemas from config."""

    required = {
        "train": get_value(config, "data_validation", "required_train_columns"),
        "members": get_value(config, "data_validation", "required_members_columns"),
        "transactions": get_value(config, "data_validation", "required_transactions_columns"),
        "user_logs": get_value(config, "data_validation", "required_user_logs_columns"),
    }
    for table_name, required_columns in required.items():
        if table_name in frames:
            validate_required_columns(frames[table_name], required_columns, table_name)


def validate_interim_artifacts(interim_dir: Path, config: Mapping) -> None:
    """Validate persisted interim artifacts after ingestion."""

    target_col = str(get_value(config, "project", "target_col"))
    cutoff_date = pd.Timestamp(get_value(config, "feature_engineering", "cutoff_date"))

    modeling_path = interim_dir / "modeling_frame.parquet"
    transactions_path = interim_dir / "transactions_summary.parquet"
    logs_path = interim_dir / "user_logs_summary.parquet"

    if not modeling_path.exists():
        raise FileNotFoundError(f"Missing interim artifact: {modeling_path}")

    modeling_frame = pd.read_parquet(modeling_path)
    validate_binary_target(modeling_frame, target_col)
    validate_temporal_cutoff(
        modeling_frame,
        ["last_transaction_date", "last_log_date"],
        cutoff_date,
    )

    if transactions_path.exists():
        transactions = pd.read_parquet(transactions_path)
        validate_temporal_cutoff(transactions, ["last_transaction_date"], cutoff_date)
    if logs_path.exists():
        logs = pd.read_parquet(logs_path)
        validate_temporal_cutoff(logs, ["last_log_date"], cutoff_date)


def validate_processed_artifacts(processed_dir: Path, config: Mapping) -> None:
    """Validate persisted processed split artifacts before training/evaluation."""

    target_col = str(get_value(config, "project", "target_col"))
    required_files = [
        "X_train.parquet",
        "X_val.parquet",
        "X_test.parquet",
        "y_train.parquet",
        "y_val.parquet",
        "y_test.parquet",
    ]
    missing = [file_name for file_name in required_files if not (processed_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed artifacts: {missing}")

    for split in ["train", "val", "test"]:
        X = pd.read_parquet(processed_dir / f"X_{split}.parquet")
        y = pd.read_parquet(processed_dir / f"y_{split}.parquet").iloc[:, 0].rename(target_col)
        if X.empty:
            raise ValueError(f"X_{split} is empty.")
        validate_binary_target(y.to_frame(), target_col)
