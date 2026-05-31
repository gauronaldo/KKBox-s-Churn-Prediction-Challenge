"""Feature engineering helpers for the KKBox churn project.

This module transforms the merged modeling frame (produced by the ingestion
stage) into a richer feature frame ready for model training.  All thresholds
and configuration knobs are read from ``config.yaml`` — no magic numbers live
here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.config import get_value

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureFrameSummary",
    "engineer_features",
    "summarize_feature_frame",
    "build_feature_frame",
    "save_feature_summary",
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeatureFrameSummary:
    """Lightweight summary of an engineered feature frame.

    Attributes:
        row_count: Total number of rows in the feature frame.
        column_count: Total number of columns (original + derived).
        numeric_features: List of numeric column names (excludes target).
        categorical_features: List of categorical/string column names.
        datetime_features: List of datetime column names.
        derived_features: List of column names added by feature engineering.
    """

    row_count: int
    column_count: int
    numeric_features: list[str]
    categorical_features: list[str]
    datetime_features: list[str]
    derived_features: list[str]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while protecting against zero denominators.

    Args:
        numerator: The dividend series.
        denominator: The divisor series.

    Returns:
        Element-wise quotient; zero denominators yield ``pd.NA``.
    """
    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def _ensure_datetime(frame: pd.DataFrame, column: str) -> None:
    """Coerce a column to datetime in-place when it is present but not typed.

    Args:
        frame: The DataFrame to modify in-place.
        column: Column name to convert.
    """
    if column in frame.columns and not pd.api.types.is_datetime64_any_dtype(
        frame[column]
    ):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        logger.debug("Coerced column '%s' to datetime.", column)


def _reference_date(
    frame: pd.DataFrame, columns: list[str]
) -> pd.Timestamp | None:
    """Compute a stable reference date from the latest OBSERVED (not future) datetime.

    ``last_expire_date`` is intentionally excluded: membership expiry can be a
    future date for active subscribers, which would push the reference far beyond
    the actual data collection cutoff and inflate all tenure/recency features.
    Only columns reflecting *actual observed events* (log activity, transactions,
    registration) should anchor the reference date.

    Args:
        frame: DataFrame containing the candidate datetime columns.
        columns: Ordered list of column names to inspect.

    Returns:
        The maximum valid timestamp found, or ``None`` if none exist.
    """
    # Exclude forward-looking dates: expiry dates reflect future obligations,
    # not past observations. Using them shifts the reference date into the
    # future, breaking days_since and tenure computations.
    EXCLUDED_FROM_REFERENCE = {"last_expire_date"}

    dates: list[pd.Timestamp] = []
    for column in columns:
        if column in EXCLUDED_FROM_REFERENCE:
            logger.debug(
                "Skipping '%s' from reference date calculation "
                "(forward-looking date).",
                column,
            )
            continue
        if column in frame.columns and pd.api.types.is_datetime64_any_dtype(
            frame[column]
        ):
            series_max = frame[column].max()
            if pd.notna(series_max):
                dates.append(series_max)
    if not dates:
        logger.warning(
            "No valid datetime columns found in %s; reference date is None.",
            columns,
        )
        return None
    ref = max(dates)
    logger.info("Reference date resolved to %s (excluded: %s).", ref, EXCLUDED_FROM_REFERENCE)
    return ref


def _categorize_age(
    age_series: pd.Series, age_min: int, age_max: int
) -> pd.Series:
    """Bin a cleaned age series into coarse demographic bands.

    Coarse bands reduce sensitivity to individual age noise while preserving
    the non-linear relationship between age and churn propensity.

    Args:
        age_series: Numeric age values (already cleaned / out-of-range → NaN).
        age_min: Minimum valid age (values below are already NaN).
        age_max: Maximum valid age used as the final bin boundary.

    Returns:
        String-typed Series with labels such as ``"18-24"`` or ``"65+"``.
    """
    bins = [0, 17, 24, 34, 44, 54, 64, age_max]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_band = pd.cut(
        age_series,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return age_band.astype("string")


def _normalize_gender(series: pd.Series) -> pd.Series:
    """Normalize raw gender values into a canonical three-way vocabulary.

    Handles abbreviated, full, and unknown representations so downstream
    one-hot encoding sees a stable, finite category set.

    Args:
        series: Raw gender column from the members table.

    Returns:
        String series containing only ``"male"``, ``"female"``, or ``"unknown"``.
    """
    normalized = series.astype("string").str.strip().str.lower()
    replacements: dict[str, str] = {
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


def _add_missing_indicators(
    frame: pd.DataFrame, columns: list[str]
) -> list[str]:
    """Add binary missing-value indicators for the specified columns in-place.

    Missing indicators allow tree-based models to learn imputation patterns
    and give linear models an explicit signal for data absence.

    Args:
        frame: DataFrame to augment in-place.
        columns: Columns for which to create ``<col>_missing`` indicators.

    Returns:
        Names of the newly created indicator columns.
    """
    created_columns: list[str] = []
    for column in columns:
        if column in frame.columns:
            indicator_name = f"{column}_missing"
            frame[indicator_name] = frame[column].isna().astype("int8")
            created_columns.append(indicator_name)
    if created_columns:
        logger.debug(
            "Created %d missing-value indicators: %s",
            len(created_columns),
            created_columns,
        )
    return created_columns


def _log1p_safe(series: pd.Series) -> pd.Series:
    """Apply ``log1p`` after clipping negative values to zero.

    Negative values can appear from aggregation rounding; clipping avoids
    ``NaN`` propagation without masking genuine data errors.

    Args:
        series: Numeric series to transform.

    Returns:
        Log1p-transformed series.
    """
    return np.log1p(series.clip(lower=0))


def _get_column_safe(
    frame: pd.DataFrame, column: str, dtype: str = "float64"
) -> pd.Series:
    """Return a column if present, otherwise an all-NA series of the given dtype.

    Using an explicit fallback Series (rather than ``DataFrame.get()``) ensures
    that safe-divide operations always receive a properly indexed Series.

    Args:
        frame: Source DataFrame.
        column: Column name to retrieve.
        dtype: NumPy dtype string for the fallback Series.

    Returns:
        The column Series or a same-index all-NA Series.
    """
    if column in frame.columns:
        return frame[column]
    return pd.Series(index=frame.index, dtype=dtype, name=column)


# ---------------------------------------------------------------------------
# Core feature engineering
# ---------------------------------------------------------------------------


def engineer_features(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Create model-ready features from the merged KKBox analysis frame.

    Feature groups applied (each guarded by column availability):

    1. **Datetime coercion** — ensures all date columns are proper timestamps.
    2. **Membership tenure** — days since registration; registration calendar
       components (year, month, day-of-week).
    3. **Demographics** — cleaned age, age band, gender normalization, profile
       completeness score.
    4. **Transaction aggregates** — spend per transaction, transaction rate,
       mean discount amount, retention rate from cancellation history.
    5. **Listening behavior** — listen event totals, completion share, unique
       track share, seconds per active day, events per active day.
    6. **Cross-source ratios** — usage-to-spend ratio, activity gap between
       last log and last transaction.
    7. **Recency flags** — binary flags for activity within the rolling window
       configured by ``feature_engineering.recent_days_window``.
    8. **Missing indicators** — binary columns flagging NaN presence.
    9. **Log-scale transforms** — applied to high-skew columns listed in
       ``feature_engineering.log_scale_features``.

    Args:
        frame: Merged modeling frame produced by the ingestion stage.
        config: Parsed project configuration (from ``config.yaml``).

    Returns:
        A copy of the input frame enriched with all engineered features.

    Raises:
        KeyError: If the ``msno`` identifier column is absent from ``frame``.
    """
    if "msno" not in frame.columns:
        raise KeyError("The modeling frame must contain an 'msno' column.")

    df = frame.copy()
    original_columns: set[str] = set(df.columns)

    # --- Read config knobs ---------------------------------------------------
    age_min: int = int(
        get_value(config, "feature_engineering", "age_min", default=7)
    )
    age_max: int = int(
        get_value(config, "feature_engineering", "age_max", default=70)
    )
    recent_days_window: int = int(
        get_value(
            config, "feature_engineering", "recent_days_window", default=30
        )
    )
    log_scale_features: list[str] = list(
        get_value(
            config, "feature_engineering", "log_scale_features", default=[]
        )
    )
    # Read cutoff from config.  This MUST match the cutoff applied during
    # ingestion (run_ingestion.py) so that all days_since_* features and
    # member_age_days use an identical, leak-free anchor point.
    cutoff_str: str | None = get_value(
        config, "feature_engineering", "cutoff_date", default=None
    )

    logger.info(
        "Starting feature engineering | age_min=%d, age_max=%d, "
        "recent_window=%d days, log_features=%s, cutoff=%s",
        age_min,
        age_max,
        recent_days_window,
        log_scale_features,
        cutoff_str,
    )

    # -------------------------------------------------------------------------
    # 1. Datetime coercion
    # -------------------------------------------------------------------------
    datetime_columns = [
        "registration_init_time",
        "last_transaction_date",
        "last_expire_date",
        "last_log_date",
    ]
    for column in datetime_columns:
        _ensure_datetime(df, column)

    # Resolve reference_date from config (preferred) or fall back to data-driven.
    # IMPORTANT: last_expire_date is intentionally excluded from the data-driven
    # fallback because it is a FORWARD-LOOKING date (the membership expiry can
    # extend into the churn observation window), which would push the anchor
    # into the future and cause temporal leakage.
    if cutoff_str:
        reference_date: pd.Timestamp | None = pd.Timestamp(cutoff_str)
        logger.info(
            "Reference date from config cutoff_date: %s", reference_date.date()
        )
    else:
        reference_date = _reference_date(
            df,
            [
                "last_transaction_date",
                "last_log_date",
                "registration_init_time",
                # last_expire_date deliberately excluded: forward-looking.
            ],
        )
        logger.warning(
            "feature_engineering.cutoff_date not set in config. "
            "Falling back to data-driven reference date: %s. "
            "Verify this does not overlap with the churn observation window.",
            reference_date,
        )
    if reference_date is not None:
        # Store as a scalar column so downstream code can audit the snapshot.
        df["analysis_reference_date"] = reference_date

    # -------------------------------------------------------------------------
    # 2. Membership tenure features
    # -------------------------------------------------------------------------
    if "registration_init_time" in df.columns:
        if reference_date is not None:
            df["member_age_days"] = (
                reference_date - df["registration_init_time"]
            ).dt.days
            logger.debug("Computed 'member_age_days'.")

        df["registration_year"] = df["registration_init_time"].dt.year
        df["registration_month"] = df["registration_init_time"].dt.month
        # Day-of-week (0=Mon, 6=Sun) can capture user acquisition cohort bias.
        df["registration_dayofweek"] = (
            df["registration_init_time"].dt.dayofweek
        )
        logger.debug(
            "Computed registration calendar features "
            "(year, month, day-of-week)."
        )

    # -------------------------------------------------------------------------
    # 3. Demographics
    # -------------------------------------------------------------------------
    if "bd" in df.columns:
        age_raw = pd.to_numeric(df["bd"], errors="coerce")
        # Clip implausible ages to NaN; these are common in KKBox (0, 999, etc.)
        age_clean = age_raw.where(age_raw.between(age_min, age_max))
        df["bd_clean"] = age_clean
        df["bd_age_valid"] = age_clean.notna().astype("int8")
        df["age_band"] = _categorize_age(age_clean, age_min, age_max)
        logger.debug(
            "Age cleaning: %d valid ages out of %d total.",
            int(age_clean.notna().sum()),
            len(age_clean),
        )

    if "gender" in df.columns:
        df["gender_clean"] = _normalize_gender(df["gender"])
        logger.debug("Normalized 'gender' -> 'gender_clean'.")

    profile_columns = [
        c for c in ["bd", "gender", "city", "registered_via"] if c in df.columns
    ]
    if profile_columns:
        # Profile completeness: fraction of demographic fields that are non-null.
        # Low completeness is a proxy for low engagement with registration flow.
        observed = pd.DataFrame(
            {c: df[c].notna().astype("int8") for c in profile_columns}
        )
        df["profile_completeness"] = observed.mean(axis=1)
        logger.debug(
            "Computed 'profile_completeness' from %d fields.", len(profile_columns)
        )

    # -------------------------------------------------------------------------
    # 4. Transaction aggregate features
    # -------------------------------------------------------------------------
    if {"trans_count", "total_spend"}.issubset(df.columns):
        df["spend_per_transaction"] = _safe_divide(
            df["total_spend"], df["trans_count"]
        )
        df["transactions_per_member_day"] = _safe_divide(
            df["trans_count"],
            _get_column_safe(df, "member_age_days"),
        )
        # Mean discount amount = list price minus actual paid; negative means
        # the user paid more than list (rare but possible with KKBox promotions).
        df["mean_discount_amount"] = _get_column_safe(
            df, "mean_plan_price"
        ).fillna(0) - _get_column_safe(df, "mean_spend").fillna(0)
        logger.debug(
            "Computed transaction ratio features "
            "(spend_per_transaction, transactions_per_member_day, "
            "mean_discount_amount)."
        )

    if "cancel_rate" in df.columns:
        # Retention rate is the complement of cancel rate; more intuitive for
        # business stakeholders and useful as a direct churn predictor.
        df["retention_rate_from_transactions"] = (
            1 - df["cancel_rate"].fillna(0)
        )
        logger.debug("Computed 'retention_rate_from_transactions'.")

    # -------------------------------------------------------------------------
    # 5. Listening behavior features
    # -------------------------------------------------------------------------
    listen_cols = {"total_25", "total_50", "total_75", "total_985", "total_100"}
    if listen_cols.issubset(df.columns):
        listen_events = df[list(listen_cols)].fillna(0).sum(axis=1)
        df["listen_events_total"] = listen_events

        df["listen_completion_share"] = _safe_divide(
            df["total_100"], listen_events
        )
        df["listen_unique_share"] = _safe_divide(
            _get_column_safe(df, "total_unq"), listen_events
        )
        logger.debug(
            "Computed listening event features "
            "(listen_events_total, listen_completion_share, "
            "listen_unique_share)."
        )

    if {"total_secs", "active_days"}.issubset(df.columns):
        # Seconds per active day measures session depth, not just frequency.
        df["secs_per_active_day"] = _safe_divide(
            df["total_secs"], df["active_days"]
        )
        logger.debug("Computed 'secs_per_active_day'.")

    if {"listen_events_total", "active_days"}.issubset(df.columns):
        df["events_per_active_day"] = _safe_divide(
            df["listen_events_total"], df["active_days"]
        )
        logger.debug("Computed 'events_per_active_day'.")

    # -------------------------------------------------------------------------
    # 6. Cross-source ratio features
    # -------------------------------------------------------------------------
    if {"total_secs", "total_spend"}.issubset(df.columns):
        # Usage-to-spend ratio captures value perception: high listening + low
        # spend may signal a free-rider; low listening + high spend may churn.
        df["usage_to_spend_ratio"] = _safe_divide(
            df["total_secs"], df["total_spend"]
        )
        logger.debug("Computed 'usage_to_spend_ratio'.")

    if {"days_since_last_transaction", "days_since_last_log"}.issubset(
        df.columns
    ):
        # Positive gap = user listened more recently than they transacted
        # (healthy). Negative = last transaction is more recent than last listen
        # (potential churn signal).
        df["recent_activity_gap"] = (
            df["days_since_last_log"] - df["days_since_last_transaction"]
        )
        logger.debug("Computed 'recent_activity_gap'.")

    # -------------------------------------------------------------------------
    # 7. Recency flags
    # -------------------------------------------------------------------------
    if "days_since_last_transaction" in df.columns:
        df["recent_transaction_flag"] = (
            df["days_since_last_transaction"] <= recent_days_window
        ).astype("int8")
        logger.debug(
            "Created 'recent_transaction_flag' (window=%d days).",
            recent_days_window,
        )

    if "days_since_last_log" in df.columns:
        df["recent_usage_flag"] = (
            df["days_since_last_log"] <= recent_days_window
        ).astype("int8")
        logger.debug(
            "Created 'recent_usage_flag' (window=%d days).",
            recent_days_window,
        )

    # ---------------------------------------------------------------------
    # Additional recency windows and simple trend/ratio features
    # ---------------------------------------------------------------------
    # Add binary flags for multiple short-term windows (7,14,30,90 days).
    multi_windows = [7, 14, 30, 90]
    for w in multi_windows:
        col_name_tx = f"recent_transaction_{w}d"
        col_name_log = f"recent_usage_{w}d"
        if "days_since_last_transaction" in df.columns:
            df[col_name_tx] = (df["days_since_last_transaction"] <= w).astype("int8")
        if "days_since_last_log" in df.columns:
            df[col_name_log] = (df["days_since_last_log"] <= w).astype("int8")

    # Decay-based proxy counts: when raw per-window counts are unavailable
    # we approximate short-term activity via exponential decay of aggregate
    # totals using the recency days as a timescale. These are not exact
    # counts but provide graded recency-weighted activity signals.
    for w in multi_windows:
        if "trans_count" in df.columns and "days_since_last_transaction" in df.columns:
            df[f"trans_count_decay_{w}d"] = (
                df["trans_count"].fillna(0) * np.exp(-df["days_since_last_transaction"].fillna(9999) / float(w))
            )
        if "total_secs" in df.columns and "days_since_last_log" in df.columns:
            df[f"secs_decay_{w}d"] = (
                df["total_secs"].fillna(0) * np.exp(-df["days_since_last_log"].fillna(9999) / float(w))
            )
        if "listen_events_total" in df.columns and "days_since_last_log" in df.columns:
            df[f"listen_decay_{w}d"] = (
                df["listen_events_total"].fillna(0) * np.exp(-df["days_since_last_log"].fillna(9999) / float(w))
            )

    # Simple ratio features that are robust to zero/NA via _safe_divide.
    if {"total_spend", "active_days"}.issubset(df.columns):
        df["spend_per_active_day"] = _safe_divide(df["total_spend"], df["active_days"])

    if {"trans_count", "active_days"}.issubset(df.columns):
        df["trans_per_active_day"] = _safe_divide(df["trans_count"], df["active_days"])

    if {"mean_spend", "total_secs"}.issubset(df.columns):
        df["spend_per_sec"] = _safe_divide(df["mean_spend"], df["total_secs"])

    # Interaction features: combine recent activity signals with value metrics.
    if "usage_to_spend_ratio" in df.columns and "recent_usage_flag" in df.columns:
        df["recent_usage_value_interaction"] = (
            df["usage_to_spend_ratio"].fillna(0) * df["recent_usage_flag"].astype(int)
        )

    if "retention_rate_from_transactions" in df.columns and "profile_completeness" in df.columns:
        df["retention_profile_interaction"] = (
            df["retention_rate_from_transactions"].fillna(0) * df["profile_completeness"].fillna(0)
        )

    # Robust z-score for secs_per_active_day as a trend/proxy signal. Use
    # median and MAD so that outliers don't dominate the scaling.
    if "secs_per_active_day" in df.columns:
        median = df["secs_per_active_day"].median(skipna=True)
        # Compute median absolute deviation (MAD) for robustness. Use the
        # median of absolute deviations from the median as a stable scale.
        mad = float(df["secs_per_active_day"].sub(median).abs().median(skipna=True) or 0)
        # Avoid divide-by-zero: if mad is zero, fall back to 1.
        mad = mad if mad > 0 else 1.0
        df["secs_per_active_day_z"] = (df["secs_per_active_day"].fillna(median) - median) / mad
        logger.debug("Computed 'secs_per_active_day_z' (median=%.2f, mad=%.2f).", median, mad)

    # -------------------------------------------------------------------------
    # 8. Missing-value indicators
    # -------------------------------------------------------------------------
    _add_missing_indicators(
        df,
        [
            "bd",
            "gender",
            "registered_via",
            "registration_init_time",
            "last_transaction_date",
            "last_expire_date",
            "last_log_date",
        ],
    )

    # -------------------------------------------------------------------------
    # 9. Log-scale transforms
    # -------------------------------------------------------------------------
    for column in log_scale_features:
        if column in df.columns:
            log_col = f"{column}_log1p"
            df[log_col] = _log1p_safe(
                pd.to_numeric(df[column], errors="coerce").fillna(0)
            )
            logger.debug(
                "Applied log1p transform: '%s' -> '%s'.", column, log_col
            )

    # -------------------------------------------------------------------------
    # Summary log
    # -------------------------------------------------------------------------
    derived_columns = sorted(set(df.columns) - original_columns)
    logger.info(
        "Feature engineering complete | %d new columns added: %s",
        len(derived_columns),
        derived_columns,
    )

    return df


# ---------------------------------------------------------------------------
# Summary and persistence helpers
# ---------------------------------------------------------------------------


def summarize_feature_frame(frame: pd.DataFrame) -> FeatureFrameSummary:
    """Summarize the engineered feature frame for metadata output.

    Args:
        frame: The fully engineered feature DataFrame.

    Returns:
        A ``FeatureFrameSummary`` capturing column counts and lists by dtype.
    """
    # Exclude the target and ID columns from feature lists.
    non_feature: set[str] = {"msno", "is_churn", "analysis_reference_date"}

    numeric_features = [
        c
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and c not in non_feature
    ]
    categorical_features = [
        c
        for c in frame.columns
        if (
            pd.api.types.is_object_dtype(frame[c])
            or pd.api.types.is_string_dtype(frame[c])
            # Use isinstance check to avoid deprecated is_categorical_dtype.
            or isinstance(frame[c].dtype, pd.CategoricalDtype)
        )
        and c not in non_feature
    ]
    datetime_features = [
        c
        for c in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[c])
    ]
    # Derived features = everything that is not the raw ID or target.
    derived_features = [
        c for c in frame.columns if c not in non_feature
    ]

    logger.info(
        "Frame summary | rows=%d, cols=%d, numeric=%d, "
        "categorical=%d, datetime=%d, derived=%d",
        frame.shape[0],
        frame.shape[1],
        len(numeric_features),
        len(categorical_features),
        len(datetime_features),
        len(derived_features),
    )

    return FeatureFrameSummary(
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        datetime_features=datetime_features,
        derived_features=derived_features,
    )


def build_feature_frame(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, FeatureFrameSummary]:
    """Build the final feature frame and its metadata summary.

    This is the single public entrypoint intended for use by the pipeline
    orchestrator (``pipeline.py``).

    Args:
        frame: Merged modeling frame from the ingestion stage.
        config: Parsed project configuration.

    Returns:
        A tuple of ``(engineered_frame, summary)``.
    """
    engineered = engineer_features(frame, config)
    summary = summarize_feature_frame(engineered)
    return engineered, summary


def save_feature_summary(summary: FeatureFrameSummary, path: Path) -> None:
    """Persist a ``FeatureFrameSummary`` as a human-readable JSON file.

    Args:
        summary: The summary dataclass to serialise.
        path: Destination file path (parent directories are created if absent).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "row_count": summary.row_count,
        "column_count": summary.column_count,
        "numeric_features": summary.numeric_features,
        "categorical_features": summary.categorical_features,
        "datetime_features": summary.datetime_features,
        "derived_features": summary.derived_features,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    logger.info("Feature summary written to %s.", path)