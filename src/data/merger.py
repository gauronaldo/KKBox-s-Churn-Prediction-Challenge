"""Merge raw tables into a single modeling frame."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute a safe rate without dividing by zero."""

    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def _aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate subscription history to one row per user."""

    frame = transactions.copy()
    frame["discount_amount"] = frame["plan_list_price"] - frame["actual_amount_paid"]
    frame["discount_rate"] = _safe_rate(frame["discount_amount"], frame["plan_list_price"])

    grouped = (
        frame.groupby("msno", dropna=False)
        .agg(
            trans_count=("msno", "size"),
            total_spend=("actual_amount_paid", "sum"),
            mean_spend=("actual_amount_paid", "mean"),
            max_spend=("actual_amount_paid", "max"),
            cancel_count=("is_cancel", "sum"),
            cancel_rate=("is_cancel", "mean"),
            auto_renew_rate=("is_auto_renew", "mean"),
            mean_plan_days=("payment_plan_days", "mean"),
            mean_plan_price=("plan_list_price", "mean"),
            mean_discount_rate=("discount_rate", "mean"),
            last_transaction_date=("transaction_date", "max"),
            last_expire_date=("membership_expire_date", "max"),
        )
        .reset_index()
    )

    reference_date = frame["transaction_date"].max()
    if pd.notna(reference_date):
        grouped["days_since_last_transaction"] = (
            reference_date - grouped["last_transaction_date"]
        ).dt.days

    logger.info("Aggregated transactions into %s user rows", grouped.shape[0])
    return grouped


def _ensure_transaction_aggregate(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transaction data in user-level aggregate form.

    The ingestion script can persist an already aggregated table to avoid
    re-reading the raw transaction history in notebooks.
    """

    required_raw_columns = {
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
    }
    if required_raw_columns.issubset(transactions.columns):
        return _aggregate_transactions(transactions)
    return transactions.copy()


def _aggregate_user_logs(user_logs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate listening behavior logs to one row per user."""

    frame = user_logs.copy()
    frame["total_listen_events"] = frame[["num_25", "num_50", "num_75", "num_985", "num_100"]].sum(axis=1)
    frame["completion_rate"] = _safe_rate(frame["num_100"], frame["total_listen_events"])

    grouped = (
        frame.groupby("msno", dropna=False)
        .agg(
            active_days=("date", "nunique"),
            last_log_date=("date", "max"),
            total_secs=("total_secs", "sum"),
            mean_secs=("total_secs", "mean"),
            total_25=("num_25", "sum"),
            total_50=("num_50", "sum"),
            total_75=("num_75", "sum"),
            total_985=("num_985", "sum"),
            total_100=("num_100", "sum"),
            total_unq=("num_unq", "sum"),
            mean_unq=("num_unq", "mean"),
            mean_completion_rate=("completion_rate", "mean"),
        )
        .reset_index()
    )

    reference_date = frame["date"].max()
    if pd.notna(reference_date):
        grouped["days_since_last_log"] = (reference_date - grouped["last_log_date"]).dt.days

    logger.info("Aggregated user logs into %s user rows", grouped.shape[0])
    return grouped


def _ensure_user_log_aggregate(user_logs: pd.DataFrame) -> pd.DataFrame:
    """Return user log data in user-level aggregate form."""

    required_raw_columns = {
        "date",
        "num_25",
        "num_50",
        "num_75",
        "num_985",
        "num_100",
        "num_unq",
        "total_secs",
    }
    if required_raw_columns.issubset(user_logs.columns):
        return _aggregate_user_logs(user_logs)
    return user_logs.copy()


def build_modeling_frame(
    train: pd.DataFrame,
    members: pd.DataFrame,
    transactions: pd.DataFrame,
    user_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Join the core KKBox tables into a single modeling frame.

    Args:
        train: Labeled target table.
        members: Member demographics table.
        transactions: Subscription history table.
        user_logs: Listening behavior table.

    Returns:
        A merged modeling dataframe with user-level aggregates.
    """

    train_columns = [column for column in train.columns if column != "msno"]
    modeling_frame = train.merge(members, on="msno", how="left", validate="one_to_one")
    transaction_agg = _ensure_transaction_aggregate(transactions)
    log_agg = _ensure_user_log_aggregate(user_logs)

    modeling_frame = modeling_frame.merge(transaction_agg, on="msno", how="left", validate="one_to_one")
    modeling_frame = modeling_frame.merge(log_agg, on="msno", how="left", validate="one_to_one")

    aggregate_columns = [
        column
        for column in modeling_frame.columns
        if column not in train_columns and column not in {"msno", "gender", "registration_init_time", "last_transaction_date", "last_expire_date", "last_log_date"}
    ]
    for column in aggregate_columns:
        if pd.api.types.is_numeric_dtype(modeling_frame[column]):
            modeling_frame[column] = modeling_frame[column].fillna(0)

    logger.info("Built modeling frame with shape %s", modeling_frame.shape)
    return modeling_frame
