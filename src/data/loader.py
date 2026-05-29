"""Raw data loading and schema validation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_COLUMNS = ("msno", "is_churn")
MEMBERS_COLUMNS = (
    "msno",
    "city",
    "bd",
    "gender",
    "registered_via",
    "registration_init_time",
)
TRANSACTIONS_COLUMNS = (
    "msno",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
)
USER_LOG_COLUMNS = (
    "msno",
    "date",
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
)


def _resolve_csv_path(raw_dir: Path, file_name: str) -> Path:
    """Build and validate the path for a raw CSV file."""

    path = raw_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return path


def _validate_columns(df: pd.DataFrame, required_columns: tuple[str, ...], table_name: str) -> None:
    """Validate required columns and log the schema check result."""

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {table_name}: {missing}")


def _validate_nulls(df: pd.DataFrame, non_null_columns: tuple[str, ...], table_name: str) -> None:
    """Ensure critical columns do not contain null values."""

    for column in non_null_columns:
        if df[column].isna().any():
            raise ValueError(f"Null values found in {table_name}.{column}")


def _log_schema(df: pd.DataFrame, table_name: str) -> None:
    """Log a compact schema summary for traceability."""

    logger.info("Loaded %s: %s rows x %s columns", table_name, df.shape[0], df.shape[1])
    logger.info("%s dtypes: %s", table_name, df.dtypes.astype(str).to_dict())


def _read_csv(path: Path, *, dtype: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read a CSV using a consistent project-wide pattern."""

    return pd.read_csv(path, dtype=dtype, low_memory=False)


def load_train(raw_dir: Path) -> pd.DataFrame:
    """Load and validate the churn training table.

    Args:
        raw_dir: Directory containing the raw CSV files.

    Returns:
        Validated training dataframe.
    """

    path = _resolve_csv_path(raw_dir, "train.csv")
    df = _read_csv(path, dtype={"msno": "string", "is_churn": "Int64"})
    _validate_columns(df, TRAIN_COLUMNS, "train")
    _validate_nulls(df, TRAIN_COLUMNS, "train")
    if not set(df["is_churn"].dropna().unique()).issubset({0, 1}):
        raise ValueError("train.is_churn must contain only 0/1 values.")
    _log_schema(df, "train")
    return df


def load_members(raw_dir: Path) -> pd.DataFrame:
    """Load and validate the members demographic table."""

    path = _resolve_csv_path(raw_dir, "members_v3.csv")
    df = _read_csv(
        path,
        dtype={
            "msno": "string",
            "city": "Int64",
            "bd": "Int64",
            "gender": "string",
            "registered_via": "Int64",
            "registration_init_time": "string",
        },
    )
    _validate_columns(df, MEMBERS_COLUMNS, "members")
    _validate_nulls(df, ("msno", "registration_init_time"), "members")
    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"],
        errors="coerce",
        format="%Y%m%d",
    )
    if df["registration_init_time"].isna().any():
        raise ValueError("members.registration_init_time contains invalid dates.")
    _log_schema(df, "members")
    return df


def load_transactions(raw_dir: Path) -> pd.DataFrame:
    """Load and validate the subscription transaction history."""

    path = _resolve_csv_path(raw_dir, "transactions.csv")
    df = _read_csv(
        path,
        dtype={
            "msno": "string",
            "is_cancel": "Int64",
            "payment_plan_days": "Int64",
            "plan_list_price": "Float64",
            "actual_amount_paid": "Float64",
            "is_auto_renew": "Int64",
            "transaction_date": "string",
            "membership_expire_date": "string",
        },
    )
    _validate_columns(df, TRANSACTIONS_COLUMNS, "transactions")
    _validate_nulls(df, ("msno", "transaction_date", "membership_expire_date"), "transactions")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce", format="%Y%m%d")
    df["membership_expire_date"] = pd.to_datetime(
        df["membership_expire_date"],
        errors="coerce",
        format="%Y%m%d",
    )
    if df[["transaction_date", "membership_expire_date"]].isna().any().any():
        raise ValueError("transactions contains invalid date values.")
    if not set(df["is_cancel"].dropna().unique()).issubset({0, 1}):
        raise ValueError("transactions.is_cancel must contain only 0/1 values.")
    if not set(df["is_auto_renew"].dropna().unique()).issubset({0, 1}):
        raise ValueError("transactions.is_auto_renew must contain only 0/1 values.")
    _log_schema(df, "transactions")
    return df


def load_user_logs(raw_dir: Path) -> pd.DataFrame:
    """Load and validate the listening behavior log table."""

    path = _resolve_csv_path(raw_dir, "user_logs.csv")
    df = _read_csv(
        path,
        dtype={
            "msno": "string",
            "date": "string",
            "num_25": "Float64",
            "num_50": "Float64",
            "num_75": "Float64",
            "num_985": "Float64",
            "num_100": "Float64",
            "num_unq": "Float64",
            "total_secs": "Float64",
        },
    )
    _validate_columns(df, USER_LOG_COLUMNS, "user_logs")
    _validate_nulls(df, ("msno", "date"), "user_logs")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%Y%m%d")
    if df["date"].isna().any():
        raise ValueError("user_logs.date contains invalid dates.")
    _log_schema(df, "user_logs")
    return df


def load_sample_submission(raw_dir: Path) -> pd.DataFrame:
    """Load the sample submission file used for inference."""

    path = _resolve_csv_path(raw_dir, "sample_submission.csv")
    df = _read_csv(path, dtype={"msno": "string"})
    _validate_columns(df, ("msno",), "sample_submission")
    _validate_nulls(df, ("msno",), "sample_submission")
    _log_schema(df, "sample_submission")
    return df


def load_all_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the four core raw tables in one call."""

    return {
        "train": load_train(raw_dir),
        "members": load_members(raw_dir),
        "transactions": load_transactions(raw_dir),
        "user_logs": load_user_logs(raw_dir),
    }
