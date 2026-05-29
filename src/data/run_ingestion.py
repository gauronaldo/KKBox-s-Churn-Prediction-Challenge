"""Ingest raw KKBox data into compact interim parquet files."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_members, load_train
from src.data.merger import build_modeling_frame
from src.utils.config import get_path, load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while protecting against zero denominators."""

    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def _read_chunked_csv(path: Path, *, chunksize: int, usecols: list[str], dtype: dict[str, str]) -> pd.io.parsers.TextFileReader:
    """Create a chunked CSV reader for very large raw tables."""

    return pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize, low_memory=False)


def _combine_grouped_frames(base: pd.DataFrame | None, chunk: pd.DataFrame, sum_columns: list[str], max_columns: list[str]) -> pd.DataFrame:
    """Combine two grouped user-level frames."""

    if base is None:
        return chunk.copy()

    combined = base.set_index("msno")
    chunk_indexed = chunk.set_index("msno")

    combined = combined.reindex(combined.index.union(chunk_indexed.index))

    combined[sum_columns] = combined[sum_columns].add(chunk_indexed[sum_columns], fill_value=0)
    for column in max_columns:
        combined[column] = pd.concat([combined[column], chunk_indexed[column]], axis=1).max(axis=1)

    combined = combined.reset_index()
    return combined


def aggregate_transactions(raw_dir: Path, chunksize: int) -> pd.DataFrame:
    """Aggregate transaction history to one row per user."""

    path = raw_dir / "transactions.csv"
    usecols = [
        "msno",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
    ]
    dtype = {
        "msno": "string",
        "transaction_date": "string",
        "membership_expire_date": "string",
        "is_cancel": "float64",
        "payment_plan_days": "float64",
        "plan_list_price": "float64",
        "actual_amount_paid": "float64",
        "is_auto_renew": "float64",
    }

    partial_frames: list[pd.DataFrame] = []
    chunk_count = 0
    for chunk in _read_chunked_csv(path, chunksize=chunksize, usecols=usecols, dtype=dtype):
        chunk_count += 1
        logger.info("transactions chunk %s: %s rows", chunk_count, len(chunk))
        chunk["transaction_date"] = pd.to_datetime(chunk["transaction_date"], format="%Y%m%d", errors="coerce")
        chunk["membership_expire_date"] = pd.to_datetime(chunk["membership_expire_date"], format="%Y%m%d", errors="coerce")
        chunk["discount_amount"] = chunk["plan_list_price"] - chunk["actual_amount_paid"]
        chunk["discount_rate"] = _safe_divide(chunk["discount_amount"], chunk["plan_list_price"])

        grouped = (
            chunk.groupby("msno", dropna=False)
            .agg(
                trans_count=("msno", "size"),
                total_spend_sum=("actual_amount_paid", "sum"),
                max_spend=("actual_amount_paid", "max"),
                cancel_count=("is_cancel", "sum"),
                auto_renew_count=("is_auto_renew", "sum"),
                payment_plan_days_sum=("payment_plan_days", "sum"),
                plan_list_price_sum=("plan_list_price", "sum"),
                discount_rate_sum=("discount_rate", "sum"),
                last_transaction_date=("transaction_date", "max"),
                last_expire_date=("membership_expire_date", "max"),
            )
            .reset_index()
        )
        partial_frames.append(grouped)
    logger.info("transactions aggregation finished after %s chunks", chunk_count)

    if not partial_frames:
        return pd.DataFrame(columns=["msno"])

    aggregated = (
        pd.concat(partial_frames, ignore_index=True)
        .groupby("msno", as_index=False)
        .agg(
            trans_count=("trans_count", "sum"),
            total_spend_sum=("total_spend_sum", "sum"),
            max_spend=("max_spend", "max"),
            cancel_count=("cancel_count", "sum"),
            auto_renew_count=("auto_renew_count", "sum"),
            payment_plan_days_sum=("payment_plan_days_sum", "sum"),
            plan_list_price_sum=("plan_list_price_sum", "sum"),
            discount_rate_sum=("discount_rate_sum", "sum"),
            last_transaction_date=("last_transaction_date", "max"),
            last_expire_date=("last_expire_date", "max"),
        )
    )

    aggregated["total_spend"] = aggregated["total_spend_sum"]
    aggregated["mean_spend"] = _safe_divide(aggregated["total_spend_sum"], aggregated["trans_count"])
    aggregated["cancel_rate"] = _safe_divide(aggregated["cancel_count"], aggregated["trans_count"])
    aggregated["auto_renew_rate"] = _safe_divide(aggregated["auto_renew_count"], aggregated["trans_count"])
    aggregated["mean_plan_days"] = _safe_divide(aggregated["payment_plan_days_sum"], aggregated["trans_count"])
    aggregated["mean_plan_price"] = _safe_divide(aggregated["plan_list_price_sum"], aggregated["trans_count"])
    aggregated["mean_discount_rate"] = _safe_divide(aggregated["discount_rate_sum"], aggregated["trans_count"])

    reference_date = aggregated["last_transaction_date"].max()
    if pd.notna(reference_date):
        aggregated["days_since_last_transaction"] = (reference_date - aggregated["last_transaction_date"]).dt.days

    output_columns = [
        "msno",
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
        "last_transaction_date",
        "last_expire_date",
        "days_since_last_transaction",
    ]
    return aggregated[output_columns]


def aggregate_user_logs(raw_dir: Path, chunksize: int) -> pd.DataFrame:
    """Aggregate listening logs to one row per user."""

    path = raw_dir / "user_logs.csv"
    usecols = [
        "msno",
        "date",
        "num_25",
        "num_50",
        "num_75",
        "num_985",
        "num_100",
        "num_unq",
        "total_secs",
    ]
    dtype = {
        "msno": "string",
        "date": "string",
        "num_25": "float64",
        "num_50": "float64",
        "num_75": "float64",
        "num_985": "float64",
        "num_100": "float64",
        "num_unq": "float64",
        "total_secs": "float64",
    }

    partial_frames: list[pd.DataFrame] = []
    chunk_count = 0
    for chunk in _read_chunked_csv(path, chunksize=chunksize, usecols=usecols, dtype=dtype):
        chunk_count += 1
        logger.info("user_logs chunk %s: %s rows", chunk_count, len(chunk))
        chunk["date"] = pd.to_datetime(chunk["date"], format="%Y%m%d", errors="coerce")
        chunk["row_completion_rate"] = _safe_divide(
            chunk["num_100"],
            chunk[["num_25", "num_50", "num_75", "num_985", "num_100"]].sum(axis=1),
        )
        grouped = (
            chunk.groupby("msno", dropna=False)
            .agg(
                active_days=("date", "size"),
                total_secs_sum=("total_secs", "sum"),
                num_25_sum=("num_25", "sum"),
                num_50_sum=("num_50", "sum"),
                num_75_sum=("num_75", "sum"),
                num_985_sum=("num_985", "sum"),
                num_100_sum=("num_100", "sum"),
                num_unq_sum=("num_unq", "sum"),
                completion_rate_sum=("row_completion_rate", "sum"),
                last_log_date=("date", "max"),
            )
            .reset_index()
        )
        partial_frames.append(grouped)
    logger.info("user_logs aggregation finished after %s chunks", chunk_count)

    if not partial_frames:
        return pd.DataFrame(columns=["msno"])

    aggregated = (
        pd.concat(partial_frames, ignore_index=True)
        .groupby("msno", as_index=False)
        .agg(
            active_days=("active_days", "sum"),
            total_secs_sum=("total_secs_sum", "sum"),
            num_25_sum=("num_25_sum", "sum"),
            num_50_sum=("num_50_sum", "sum"),
            num_75_sum=("num_75_sum", "sum"),
            num_985_sum=("num_985_sum", "sum"),
            num_100_sum=("num_100_sum", "sum"),
            num_unq_sum=("num_unq_sum", "sum"),
            completion_rate_sum=("completion_rate_sum", "sum"),
            last_log_date=("last_log_date", "max"),
        )
    )

    aggregated["total_secs"] = aggregated["total_secs_sum"]
    aggregated["mean_secs"] = _safe_divide(aggregated["total_secs_sum"], aggregated["active_days"])
    aggregated["total_25"] = aggregated["num_25_sum"]
    aggregated["total_50"] = aggregated["num_50_sum"]
    aggregated["total_75"] = aggregated["num_75_sum"]
    aggregated["total_985"] = aggregated["num_985_sum"]
    aggregated["total_100"] = aggregated["num_100_sum"]
    aggregated["total_unq"] = aggregated["num_unq_sum"]
    aggregated["mean_unq"] = _safe_divide(aggregated["num_unq_sum"], aggregated["active_days"])
    aggregated["completion_rate"] = _safe_divide(
        aggregated["num_100_sum"],
        aggregated[["num_25_sum", "num_50_sum", "num_75_sum", "num_985_sum", "num_100_sum"]].sum(axis=1),
    )
    aggregated["mean_completion_rate"] = _safe_divide(aggregated["completion_rate_sum"], aggregated["active_days"])

    reference_date = aggregated["last_log_date"].max()
    if pd.notna(reference_date):
        aggregated["days_since_last_log"] = (reference_date - aggregated["last_log_date"]).dt.days

    output_columns = [
        "msno",
        "active_days",
        "total_secs",
        "mean_secs",
        "total_25",
        "total_50",
        "total_75",
        "total_985",
        "total_100",
        "total_unq",
        "mean_unq",
        "completion_rate",
        "mean_completion_rate",
        "last_log_date",
        "days_since_last_log",
    ]
    return aggregated[output_columns]


def main() -> None:
    """Run the ingestion pipeline and write compact parquet outputs."""

    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config" / "config.yaml")
    logger = setup_logger("run_ingestion")

    raw_dir = get_path(config, "raw_dir")
    interim_dir = get_path(config, "interim_dir")
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir
    if not interim_dir.is_absolute():
        interim_dir = project_root / interim_dir
    interim_dir.mkdir(parents=True, exist_ok=True)

    chunksize = int(config["ingestion"]["chunksize"])

    logger.info("Loading compact tables from raw data")
    train = load_train(raw_dir)
    logger.info("train loaded: %s rows x %s columns", train.shape[0], train.shape[1])
    members = load_members(raw_dir)
    logger.info("members loaded: %s rows x %s columns", members.shape[0], members.shape[1])
    transactions = aggregate_transactions(raw_dir, chunksize)
    user_logs = aggregate_user_logs(raw_dir, chunksize)

    logger.info("Building modeling frame")
    modeling_frame = build_modeling_frame(train, members, transactions, user_logs)

    outputs = {
        "train.parquet": train,
        "members.parquet": members,
        "transactions_summary.parquet": transactions,
        "user_logs_summary.parquet": user_logs,
        "modeling_frame.parquet": modeling_frame,
    }
    for file_name, frame in outputs.items():
        path = interim_dir / file_name
        frame.to_parquet(path, index=False)
        logger.info("Saved %s", path)


if __name__ == "__main__":
    main()