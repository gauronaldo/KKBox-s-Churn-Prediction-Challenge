"""Feature engineering helpers for the KKBox churn project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.config import get_value


@dataclass(slots=True)
class FeatureFrameSummary:
    """Lightweight summary of an engineered feature frame."""

    row_count: int
    column_count: int
    numeric_features: list[str]
    categorical_features: list[str]
    datetime_features: list[str]
    derived_features: list[str]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while protecting against zero denominators."""

    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def _ensure_datetime(frame: pd.DataFrame, column: str) -> None:
    """Convert a column to datetime when present."""

    if column in frame.columns and not pd.api.types.is_datetime64_any_dtype(frame[column]):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")


def _reference_date(frame: pd.DataFrame, columns: list[str]) -> pd.Timestamp | pd.NaT:
    """Compute a stable reference date from available datetime columns."""

    dates: list[pd.Timestamp] = []
    for column in columns:
        if column in frame.columns and pd.api.types.is_datetime64_any_dtype(frame[column]):
            series_max = frame[column].max()
            if pd.notna(series_max):
                dates.append(series_max)
    if not dates:
        return pd.NaT
    return max(dates)


def _categorize_age(age_series: pd.Series, age_min: int, age_max: int) -> pd.Series:
    """Create a coarse age band from a cleaned age series."""

    bins = [0, 17, 24, 34, 44, 54, 64, age_max]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_band = pd.cut(age_series, bins=bins, labels=labels, include_lowest=True, right=True)
    return age_band.astype("string")


def _normalize_gender(series: pd.Series) -> pd.Series:
    """Normalize gender labels into a small set of values."""

    normalized = series.astype("string").str.strip().str.lower()
    replacements = {
        "m": "male",
        "f": "female",
        "male": "male",
        "female": "female",
        "unknown": "unknown",
        "u": "unknown",
        "": "unknown",
    }
    normalized = normalized.replace(replacements)
    return normalized.fillna("unknown")


def _add_missing_indicators(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Add boolean missing indicators for the specified columns."""

    created_columns: list[str] = []
    for column in columns:
        if column in frame.columns:
            indicator_name = f"{column}_missing"
            frame[indicator_name] = frame[column].isna().astype("int8")
            created_columns.append(indicator_name)
    return created_columns


def _log1p_safe(series: pd.Series) -> pd.Series:
    """Apply log1p after clipping negative values to zero."""

    return np.log1p(series.clip(lower=0))


def engineer_features(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Create model-ready features from the merged KKBox analysis frame.

    Args:
        frame: Merged modeling frame produced by the ingestion stage.
        config: Parsed project configuration.

    Returns:
        A copy of the frame with additional engineered features.
    """

    if "msno" not in frame.columns:
        raise KeyError("The modeling frame must contain an 'msno' column.")

    df = frame.copy()

    age_min = int(get_value(config, "feature_engineering", "age_min", default=7))
    age_max = int(get_value(config, "feature_engineering", "age_max", default=70))
    recent_days_window = int(get_value(config, "feature_engineering", "recent_days_window", default=30))
    log_scale_features = list(get_value(config, "feature_engineering", "log_scale_features", default=[]))

    datetime_columns = ["registration_init_time", "last_transaction_date", "last_expire_date", "last_log_date"]
    for column in datetime_columns:
        _ensure_datetime(df, column)

    reference_date = _reference_date(df, datetime_columns)
    if pd.notna(reference_date):
        df["analysis_reference_date"] = reference_date

    derived_columns: list[str] = []

    if "registration_init_time" in df.columns:
        if pd.notna(reference_date):
            df["member_age_days"] = (reference_date - df["registration_init_time"]).dt.days
            derived_columns.append("member_age_days")
        df["registration_year"] = df["registration_init_time"].dt.year
        df["registration_month"] = df["registration_init_time"].dt.month
        df["registration_dayofweek"] = df["registration_init_time"].dt.dayofweek
        derived_columns.extend(["registration_year", "registration_month", "registration_dayofweek"])

    if "bd" in df.columns:
        age_raw = pd.to_numeric(df["bd"], errors="coerce")
        age_clean = age_raw.where(age_raw.between(age_min, age_max))
        df["bd_clean"] = age_clean
        df["bd_age_valid"] = age_clean.notna().astype("int8")
        df["age_band"] = _categorize_age(age_clean, age_min, age_max)
        derived_columns.extend(["bd_clean", "bd_age_valid", "age_band"])

    if "gender" in df.columns:
        df["gender_clean"] = _normalize_gender(df["gender"])
        derived_columns.append("gender_clean")

    if {"city", "registered_via"}.intersection(df.columns):
        profile_columns = [column for column in ["bd", "gender", "city", "registered_via"] if column in df.columns]
        if profile_columns:
            observed = pd.DataFrame(index=df.index)
            for column in profile_columns:
                observed[column] = df[column].notna().astype("int8")
            df["profile_completeness"] = observed.mean(axis=1)
            derived_columns.append("profile_completeness")

    if {"trans_count", "total_spend"}.issubset(df.columns):
        df["spend_per_transaction"] = _safe_divide(df["total_spend"], df["trans_count"])
        df["transactions_per_member_day"] = _safe_divide(df["trans_count"], df.get("member_age_days", pd.Series(index=df.index, dtype="float64")))
        df["mean_discount_amount"] = df.get("mean_plan_price", 0) - df.get("mean_spend", 0)
        derived_columns.extend(["spend_per_transaction", "transactions_per_member_day", "mean_discount_amount"])

    if "cancel_rate" in df.columns:
        df["retention_rate_from_transactions"] = 1 - df["cancel_rate"].fillna(0)
        derived_columns.append("retention_rate_from_transactions")

    if {"total_25", "total_50", "total_75", "total_985", "total_100"}.issubset(df.columns):
        listen_events = df[["total_25", "total_50", "total_75", "total_985", "total_100"]].fillna(0).sum(axis=1)
        df["listen_events_total"] = listen_events
        df["listen_completion_share"] = _safe_divide(df["total_100"], listen_events)
        df["listen_unique_share"] = _safe_divide(df.get("total_unq", pd.Series(index=df.index, dtype="float64")), listen_events)
        derived_columns.extend(["listen_events_total", "listen_completion_share", "listen_unique_share"])

    if {"total_secs", "active_days"}.issubset(df.columns):
        df["secs_per_active_day"] = _safe_divide(df["total_secs"], df["active_days"])
        derived_columns.append("secs_per_active_day")

    if {"listen_events_total", "active_days"}.issubset(df.columns):
        df["events_per_active_day"] = _safe_divide(df["listen_events_total"], df["active_days"])
        derived_columns.append("events_per_active_day")

    if {"total_secs", "total_spend"}.issubset(df.columns):
        df["usage_to_spend_ratio"] = _safe_divide(df["total_secs"], df["total_spend"])
        derived_columns.append("usage_to_spend_ratio")

    if {"days_since_last_transaction", "days_since_last_log"}.issubset(df.columns):
        df["recent_activity_gap"] = df["days_since_last_log"] - df["days_since_last_transaction"]
        derived_columns.append("recent_activity_gap")

    if "days_since_last_transaction" in df.columns:
        df["recent_transaction_flag"] = (df["days_since_last_transaction"] <= recent_days_window).astype("int8")
        derived_columns.append("recent_transaction_flag")

    if "days_since_last_log" in df.columns:
        df["recent_usage_flag"] = (df["days_since_last_log"] <= recent_days_window).astype("int8")
        derived_columns.append("recent_usage_flag")

    missing_indicators = _add_missing_indicators(
        df,
        ["bd", "gender", "registered_via", "registration_init_time", "last_transaction_date", "last_expire_date", "last_log_date"],
    )
    derived_columns.extend(missing_indicators)

    for column in log_scale_features:
        if column in df.columns:
            df[f"{column}_log1p"] = _log1p_safe(pd.to_numeric(df[column], errors="coerce").fillna(0))
            derived_columns.append(f"{column}_log1p")

    numeric_candidates = [
        "bd",
        "bd_clean",
        "profile_completeness",
        "trans_count",
        "total_spend",
        "mean_spend",
        "max_spend",
        "cancel_count",
        "cancel_rate",
        "auto_renew_rate",
        "mean_plan_days",
        "mean_plan_price",
        "mean_discount_rate",
        "days_since_last_transaction",
        "days_since_last_log",
        "active_days",
        "total_secs",
        "mean_secs",
        "total_unq",
        "mean_unq",
        "mean_completion_rate",
        "member_age_days",
        "spend_per_transaction",
        "transactions_per_member_day",
        "mean_discount_amount",
        "retention_rate_from_transactions",
        "listen_events_total",
        "listen_completion_share",
        "listen_unique_share",
        "secs_per_active_day",
        "events_per_active_day",
        "usage_to_spend_ratio",
        "recent_activity_gap",
    ]
    for column in numeric_candidates:
        if column in df.columns and pd.api.types.is_bool_dtype(df[column]):
            df[column] = df[column].astype("int8")

    logger_columns = [column for column in derived_columns if column in df.columns]
    return df


def summarize_feature_frame(frame: pd.DataFrame) -> FeatureFrameSummary:
    """Summarize the engineered feature frame for metadata output."""

    numeric_features = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column]) and column not in {"is_churn"}]
    categorical_features = [
        column
        for column in frame.columns
        if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column]) or pd.api.types.is_categorical_dtype(frame[column])
    ]
    datetime_features = [column for column in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[column])]
    derived_features = [column for column in frame.columns if column not in {"msno", "is_churn"}]
    return FeatureFrameSummary(
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        datetime_features=datetime_features,
        derived_features=derived_features,
    )


def build_feature_frame(frame: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, FeatureFrameSummary]:
    """Build the final feature frame and its metadata summary."""

    engineered = engineer_features(frame, config)
    summary = summarize_feature_frame(engineered)
    return engineered, summary


def save_feature_summary(summary: FeatureFrameSummary, path: Path) -> None:
    """Persist a feature summary as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "row_count": summary.row_count,
        "column_count": summary.column_count,
        "numeric_features": summary.numeric_features,
        "categorical_features": summary.categorical_features,
        "datetime_features": summary.datetime_features,
        "derived_features": summary.derived_features,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
