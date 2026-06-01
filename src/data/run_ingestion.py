"""Ingest raw KKBox data into compact interim parquet files."""

from __future__ import annotations

import logging
import sys
from tqdm import tqdm
from math import ceil
from pathlib import Path

import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    duckdb = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_members, load_train
from src.data.merger import build_modeling_frame
from src.utils.config import get_path, get_value, load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while protecting against zero denominators."""

    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, float("nan"))
    return numerator / denominator


def _read_chunked_csv(path: Path, *, chunksize: int, usecols: list[str], dtype: dict[str, str]) -> pd.io.parsers.TextFileReader:
    """Create a chunked CSV reader for very large raw tables."""

    return pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize, low_memory=False)


def _sql_string(value: str | Path) -> str:
    """Return a SQL string literal with embedded quotes escaped."""

    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def _connect_duckdb(config: dict) -> "duckdb.DuckDBPyConnection":
    """Create a configured DuckDB connection for large CSV aggregation."""

    if duckdb is None:
        raise ImportError(
            "DuckDB ingestion requires the 'duckdb' package. "
            "Install project dependencies with: pip install -r requirements.txt"
        )

    connection = duckdb.connect(database=":memory:")
    threads = get_value(config, "ingestion", "duckdb_threads")
    memory_limit = get_value(config, "ingestion", "duckdb_memory_limit")
    max_temp_directory_size = get_value(config, "ingestion", "duckdb_max_temp_directory_size")
    preserve_insertion_order = bool(
        get_value(config, "ingestion", "duckdb_preserve_insertion_order", default=False)
    )
    temp_dir = get_value(config, "ingestion", "duckdb_temp_dir")

    # Aggregations do not need source row order; disabling it reduces memory/spill pressure.
    connection.execute(
        f"SET preserve_insertion_order = {str(preserve_insertion_order).lower()}"
    )
    if threads is not None:
        connection.execute(f"SET threads = {int(threads)}")
    if memory_limit:
        connection.execute(f"SET memory_limit = {_sql_string(str(memory_limit))}")
    if max_temp_directory_size:
        connection.execute(
            f"SET max_temp_directory_size = {_sql_string(str(max_temp_directory_size))}"
        )
    if temp_dir:
        temp_path = Path(str(temp_dir))
        if not temp_path.is_absolute():
            temp_path = PROJECT_ROOT / temp_path
        temp_path.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = {_sql_string(temp_path)}")

    return connection


def _duckdb_read_csv_sql(path: Path, columns: dict[str, str]) -> str:
    """Build a typed DuckDB CSV scan expression for stable large-file reads."""

    column_sql = ", ".join(f"{name}: '{dtype}'" for name, dtype in columns.items())
    return (
        "read_csv("
        f"{_sql_string(path)}, "
        "header=true, "
        "auto_detect=false, "
        "delim=',', "
        "quote='\"', "
        "escape='\"', "
        f"columns={{{column_sql}}}, "
        "null_padding=false"
        ")"
    )


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


def _estimate_total_chunks(path: Path, chunksize: int, sample_lines: int = 1000) -> int:
    """Estimate total chunks instantly using file size and average bytes per row."""
    file_size = path.stat().st_size
    if file_size == 0:
        return 1

    with path.open("rb") as f:
        f.readline() # Bỏ qua dòng header
        
        bytes_read = 0
        lines_read = 0
        for _ in range(sample_lines):
            line = f.readline()
            if not line:
                break
            bytes_read += len(line)
            lines_read += 1

    if lines_read == 0:
        return 1

    avg_bytes_per_row = bytes_read / lines_read
    total_rows_est = int(file_size / avg_bytes_per_row)
    
    return max(1, ceil(total_rows_est / chunksize))


def _format_seconds(seconds: float) -> str:
    """Format seconds as H:MM:SS for progress messages."""

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:d}:{remaining_seconds:02d}"


def _render_progress_bar(completed: int, total: int, *, width: int = 20) -> str:
    """Render a compact ASCII progress bar for terminal logs."""

    if total <= 0:
        filled = width
        fraction = 1.0
    else:
        fraction = min(1.0, max(0.0, completed / total))
        filled = int(round(fraction * width))

    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {fraction * 100:5.1f}%"


def _aggregate_transactions_pandas(raw_dir: Path, chunksize: int, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate transaction history to one row per user."""

    path = raw_dir / "transactions.csv"
    
    # 1. Sử dụng hàm ước lượng siêu tốc thay vì đếm từng dòng
    total_chunks_est = _estimate_total_chunks(path, chunksize)
    
    usecols = [
        "msno", "transaction_date", "membership_expire_date", "is_cancel",
        "payment_plan_days", "plan_list_price", "actual_amount_paid", "is_auto_renew",
    ]
    dtype = {
        "msno": "string", "transaction_date": "string", "membership_expire_date": "string",
        "is_cancel": "float64", "payment_plan_days": "float64", "plan_list_price": "float64",
        "actual_amount_paid": "float64", "is_auto_renew": "float64",
    }

    aggregated: pd.DataFrame | None = None
    rows_seen_total = 0
    rows_kept_total = 0

    sum_columns = [
        "trans_count", "total_spend_sum", "cancel_count", "auto_renew_count",
        "payment_plan_days_sum", "plan_list_price_sum", "discount_rate_sum",
    ]
    max_columns = ["max_spend", "last_transaction_date", "last_expire_date"]

    # 2. Khởi tạo thanh tiến trình tqdm
    chunk_iterator = _read_chunked_csv(path, chunksize=chunksize, usecols=usecols, dtype=dtype)
    progress_bar = tqdm(chunk_iterator, total=total_chunks_est, desc="Aggregating Transactions", unit="chunk")

    for chunk in progress_bar:
        rows_seen = len(chunk)
        rows_seen_total += rows_seen

        chunk["transaction_date"] = pd.to_datetime(chunk["transaction_date"], format="%Y%m%d", errors="coerce")
        chunk["membership_expire_date"] = pd.to_datetime(chunk["membership_expire_date"], format="%Y%m%d", errors="coerce")
        
        # Exclude future renewal events so transaction features are snapshot-safe.
        chunk = chunk[chunk["transaction_date"].notna() & (chunk["transaction_date"] <= cutoff_date)]
        rows_kept_total += len(chunk)

        if chunk.empty:
            # Vẫn cập nhật thông tin lên thanh tiến trình dù chunk rỗng
            progress_bar.set_postfix(seen=rows_seen_total, kept=rows_kept_total)
            continue

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
        aggregated = _combine_grouped_frames(aggregated, grouped, sum_columns, max_columns)

        # 3. Cập nhật các chỉ số thực tế (metrics) lên góc phải của thanh tiến trình
        progress_bar.set_postfix(
            seen=rows_seen_total, 
            kept=rows_kept_total, 
            users_agg=len(aggregated)
        )

    # Đóng thanh tiến trình khi hoàn thành
    progress_bar.close()

    if aggregated is None:
        return pd.DataFrame(columns=["msno"])

    aggregated["total_spend"] = aggregated["total_spend_sum"]
    aggregated["mean_spend"] = _safe_divide(aggregated["total_spend_sum"], aggregated["trans_count"])
    aggregated["cancel_rate"] = _safe_divide(aggregated["cancel_count"], aggregated["trans_count"])
    aggregated["auto_renew_rate"] = _safe_divide(aggregated["auto_renew_count"], aggregated["trans_count"])
    aggregated["mean_plan_days"] = _safe_divide(aggregated["payment_plan_days_sum"], aggregated["trans_count"])
    aggregated["mean_plan_price"] = _safe_divide(aggregated["plan_list_price_sum"], aggregated["trans_count"])
    aggregated["mean_discount_rate"] = _safe_divide(aggregated["discount_rate_sum"], aggregated["trans_count"])

    aggregated["days_since_last_transaction"] = (
        cutoff_date - aggregated["last_transaction_date"]
    ).dt.days

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


def _aggregate_transactions_duckdb(
    raw_dir: Path,
    cutoff_date: pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    """Aggregate transaction history with DuckDB's vectorized CSV engine."""

    path = raw_dir / "transactions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    columns = {
        "msno": "VARCHAR",
        "payment_method_id": "VARCHAR",
        "payment_plan_days": "DOUBLE",
        "plan_list_price": "DOUBLE",
        "actual_amount_paid": "DOUBLE",
        "is_auto_renew": "DOUBLE",
        "transaction_date": "VARCHAR",
        "membership_expire_date": "VARCHAR",
        "is_cancel": "DOUBLE",
    }
    cutoff_sql = _sql_string(cutoff_date.strftime("%Y-%m-%d"))
    scan_sql = _duckdb_read_csv_sql(path, columns)

    query = f"""
        WITH typed AS (
            SELECT
                msno,
                TRY_STRPTIME(transaction_date, '%Y%m%d')::DATE AS transaction_date,
                TRY_STRPTIME(membership_expire_date, '%Y%m%d')::DATE AS membership_expire_date,
                is_cancel,
                payment_plan_days,
                plan_list_price,
                actual_amount_paid,
                is_auto_renew
            FROM {scan_sql}
        ),
        filtered AS (
            SELECT *
            FROM typed
            WHERE transaction_date IS NOT NULL
              AND transaction_date <= CAST({cutoff_sql} AS DATE)
        ),
        grouped AS (
            SELECT
                msno,
                COUNT(*) AS trans_count,
                SUM(actual_amount_paid) AS total_spend,
                AVG(actual_amount_paid) AS mean_spend,
                MAX(actual_amount_paid) AS max_spend,
                SUM(is_cancel) AS cancel_count,
                AVG(is_cancel) AS cancel_rate,
                AVG(is_auto_renew) AS auto_renew_rate,
                AVG(payment_plan_days) AS mean_plan_days,
                AVG(plan_list_price) AS mean_plan_price,
                AVG(
                    CASE
                        WHEN plan_list_price = 0 OR plan_list_price IS NULL THEN NULL
                        ELSE (plan_list_price - actual_amount_paid) / plan_list_price
                    END
                ) AS mean_discount_rate,
                MAX(transaction_date) AS last_transaction_date,
                MAX(membership_expire_date) AS last_expire_date
            FROM filtered
            GROUP BY msno
        )
        SELECT
            msno,
            trans_count,
            total_spend,
            mean_spend,
            max_spend,
            cancel_count,
            cancel_rate,
            auto_renew_rate,
            mean_plan_days,
            mean_plan_price,
            mean_discount_rate,
            last_transaction_date,
            last_expire_date,
            DATE_DIFF('day', last_transaction_date, CAST({cutoff_sql} AS DATE))
                AS days_since_last_transaction
        FROM grouped
    """

    logger.info("Aggregating transactions with DuckDB from %s", path)
    with _connect_duckdb(config) as connection:
        result = connection.execute(query).fetchdf()
    logger.info("DuckDB transactions aggregate complete: %s rows", result.shape[0])
    return result


def aggregate_transactions(
    raw_dir: Path,
    chunksize: int,
    cutoff_date: pd.Timestamp,
    config: dict | None = None,
) -> pd.DataFrame:
    """Aggregate transaction history to one row per user."""

    backend = str(get_value(config or {}, "ingestion", "backend", default="pandas")).lower()
    if backend == "duckdb":
        return _aggregate_transactions_duckdb(raw_dir, cutoff_date, config or {})
    if backend != "pandas":
        raise ValueError(f"Unsupported ingestion backend: {backend}")
    return _aggregate_transactions_pandas(raw_dir, chunksize, cutoff_date)


def _add_behavior_window_columns(
    chunk: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    behavior_windows: list[int],
) -> pd.DataFrame:
    """Add per-window listening columns before user-level aggregation."""

    days_before_cutoff = (cutoff_date - chunk["date"]).dt.days
    event_columns = ["num_25", "num_50", "num_75", "num_985", "num_100"]

    for window in behavior_windows:
        suffix = f"{window}d"
        in_window = days_before_cutoff.between(0, window)
        chunk[f"active_days_{suffix}"] = in_window.astype("int16")
        chunk[f"total_secs_{suffix}"] = chunk["total_secs"].where(in_window, 0.0)
        chunk[f"total_unq_{suffix}"] = chunk["num_unq"].where(in_window, 0.0)
        for column in event_columns:
            output_column = column.replace("num_", "total_")
            chunk[f"{output_column}_{suffix}"] = chunk[column].where(in_window, 0.0)

    return chunk


def _aggregate_user_logs_pandas(
    raw_dir: Path,
    chunksize: int,
    cutoff_date: pd.Timestamp,
    behavior_windows: list[int],
) -> pd.DataFrame:
    """Aggregate listening logs to one row per user."""

    path = raw_dir / "user_logs.csv"
    
    # 1. Dùng hàm ước lượng mới thay cho _count_data_rows
    total_chunks_est = _estimate_total_chunks(path, chunksize)
    
    usecols = [
        "msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs",
    ]
    dtype = {
        "msno": "string", "date": "string", "num_25": "float64", "num_50": "float64",
        "num_75": "float64", "num_985": "float64", "num_100": "float64", "num_unq": "float64",
        "total_secs": "float64",
    }

    aggregated: pd.DataFrame | None = None
    rows_seen_total = 0
    rows_kept_total = 0

    sum_columns = [
        "active_days", "total_secs_sum", "num_25_sum", "num_50_sum", "num_75_sum",
        "num_985_sum", "num_100_sum", "num_unq_sum", "completion_rate_sum",
    ]
    
    for window in behavior_windows:
        suffix = f"{window}d"
        sum_columns.extend([
            f"active_days_{suffix}", f"total_secs_{suffix}", f"total_unq_{suffix}",
            f"total_25_{suffix}", f"total_50_{suffix}", f"total_75_{suffix}",
            f"total_985_{suffix}", f"total_100_{suffix}",
        ])
    max_columns = ["last_log_date"]

    # 2. Khởi tạo thanh tiến trình tqdm
    chunk_iterator = _read_chunked_csv(path, chunksize=chunksize, usecols=usecols, dtype=dtype)
    progress_bar = tqdm(chunk_iterator, total=total_chunks_est, desc="Aggregating User Logs", unit="chunk")

    for chunk in progress_bar:
        rows_seen_total += len(chunk)

        chunk["date"] = pd.to_datetime(chunk["date"], format="%Y%m%d", errors="coerce")
        
        # SỬA LỖI PANDAS WARNING Ở ĐÂY: Thêm .copy() ở cuối
        chunk = chunk[chunk["date"].notna() & (chunk["date"] <= cutoff_date)].copy()
        
        rows_kept_total += len(chunk)

        if chunk.empty:
            progress_bar.set_postfix(seen=rows_seen_total, kept=rows_kept_total)
            continue

        chunk["row_completion_rate"] = _safe_divide(
            chunk["num_100"],
            chunk[["num_25", "num_50", "num_75", "num_985", "num_100"]].sum(axis=1),
        )
        chunk = _add_behavior_window_columns(chunk, cutoff_date, behavior_windows)
        
        window_aggregations = {}
        for window in behavior_windows:
            suffix = f"{window}d"
            window_aggregations.update({
                f"active_days_{suffix}": (f"active_days_{suffix}", "sum"),
                f"total_secs_{suffix}": (f"total_secs_{suffix}", "sum"),
                f"total_unq_{suffix}": (f"total_unq_{suffix}", "sum"),
                f"total_25_{suffix}": (f"total_25_{suffix}", "sum"),
                f"total_50_{suffix}": (f"total_50_{suffix}", "sum"),
                f"total_75_{suffix}": (f"total_75_{suffix}", "sum"),
                f"total_985_{suffix}": (f"total_985_{suffix}", "sum"),
                f"total_100_{suffix}": (f"total_100_{suffix}", "sum"),
            })
            
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
                **window_aggregations,
            )
            .reset_index()
        )
        aggregated = _combine_grouped_frames(aggregated, grouped, sum_columns, max_columns)

        # 3. Cập nhật thanh tiến trình
        progress_bar.set_postfix(
            seen=rows_seen_total, 
            kept=rows_kept_total, 
            users_agg=len(aggregated)
        )

    progress_bar.close()

    if aggregated is None:
        return pd.DataFrame(columns=["msno"])

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

    aggregated["days_since_last_log"] = (cutoff_date - aggregated["last_log_date"]).dt.days

    window_output_columns: list[str] = []
    for window in behavior_windows:
        suffix = f"{window}d"
        event_total = aggregated[
            [
                f"total_25_{suffix}",
                f"total_50_{suffix}",
                f"total_75_{suffix}",
                f"total_985_{suffix}",
                f"total_100_{suffix}",
            ]
        ].sum(axis=1)
        aggregated[f"completion_rate_{suffix}"] = _safe_divide(aggregated[f"total_100_{suffix}"], event_total)
        aggregated[f"skip_rate_{suffix}"] = _safe_divide(
            aggregated[f"total_25_{suffix}"] + aggregated[f"total_50_{suffix}"],
            event_total,
        )
        aggregated[f"secs_per_active_day_{suffix}"] = _safe_divide(
            aggregated[f"total_secs_{suffix}"],
            aggregated[f"active_days_{suffix}"],
        )
        aggregated[f"unq_per_active_day_{suffix}"] = _safe_divide(
            aggregated[f"total_unq_{suffix}"],
            aggregated[f"active_days_{suffix}"],
        )
        window_output_columns.extend(
            [
                f"active_days_{suffix}",
                f"total_secs_{suffix}",
                f"total_unq_{suffix}",
                f"completion_rate_{suffix}",
                f"skip_rate_{suffix}",
                f"secs_per_active_day_{suffix}",
                f"unq_per_active_day_{suffix}",
            ]
        )

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
        *window_output_columns,
    ]
    return aggregated[output_columns]


def _build_user_log_window_sql(behavior_windows: list[int], cutoff_sql: str) -> tuple[str, str]:
    """Build DuckDB aggregate and output SQL fragments for behavior windows."""

    aggregate_parts: list[str] = []
    output_parts: list[str] = []
    for window in behavior_windows:
        suffix = f"{window}d"
        # Window features preserve the pandas implementation's inclusive cutoff window.
        in_window = (
            f"log_date BETWEEN CAST({cutoff_sql} AS DATE) - INTERVAL {int(window)} DAY "
            f"AND CAST({cutoff_sql} AS DATE)"
        )
        aggregate_parts.extend(
            [
                f"SUM(CASE WHEN {in_window} THEN 1 ELSE 0 END) AS active_days_{suffix}",
                f"SUM(CASE WHEN {in_window} THEN total_secs ELSE 0 END) AS total_secs_{suffix}",
                f"SUM(CASE WHEN {in_window} THEN num_unq ELSE 0 END) AS total_unq_{suffix}",
                f"SUM(CASE WHEN {in_window} THEN num_25 ELSE 0 END) AS total_25_{suffix}",
                f"SUM(CASE WHEN {in_window} THEN num_50 ELSE 0 END) AS total_50_{suffix}",
                f"SUM(CASE WHEN {in_window} THEN num_75 ELSE 0 END) AS total_75_{suffix}",
                f"SUM(CASE WHEN {in_window} THEN num_985 ELSE 0 END) AS total_985_{suffix}",
                f"SUM(CASE WHEN {in_window} THEN num_100 ELSE 0 END) AS total_100_{suffix}",
            ]
        )
        event_total = (
            f"total_25_{suffix} + total_50_{suffix} + total_75_{suffix} "
            f"+ total_985_{suffix} + total_100_{suffix}"
        )
        output_parts.extend(
            [
                f"active_days_{suffix}",
                f"total_secs_{suffix}",
                f"total_unq_{suffix}",
                (
                    f"CASE WHEN ({event_total}) = 0 THEN NULL "
                    f"ELSE total_100_{suffix} / ({event_total}) END AS completion_rate_{suffix}"
                ),
                (
                    f"CASE WHEN ({event_total}) = 0 THEN NULL "
                    f"ELSE (total_25_{suffix} + total_50_{suffix}) / ({event_total}) END "
                    f"AS skip_rate_{suffix}"
                ),
                (
                    f"CASE WHEN active_days_{suffix} = 0 THEN NULL "
                    f"ELSE total_secs_{suffix} / active_days_{suffix} END "
                    f"AS secs_per_active_day_{suffix}"
                ),
                (
                    f"CASE WHEN active_days_{suffix} = 0 THEN NULL "
                    f"ELSE total_unq_{suffix} / active_days_{suffix} END "
                    f"AS unq_per_active_day_{suffix}"
                ),
            ]
        )

    aggregate_sql = ""
    if aggregate_parts:
        aggregate_sql = ",\n                " + ",\n                ".join(aggregate_parts)
    output_sql = ""
    if output_parts:
        output_sql = ",\n            " + ",\n            ".join(output_parts)
    return aggregate_sql, output_sql


def _aggregate_user_logs_duckdb(
    raw_dir: Path,
    cutoff_date: pd.Timestamp,
    behavior_windows: list[int],
    config: dict,
) -> pd.DataFrame:
    """Aggregate listening logs with DuckDB to avoid slow pandas chunk merges."""

    path = raw_dir / "user_logs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    columns = {
        "msno": "VARCHAR",
        "date": "VARCHAR",
        "num_25": "DOUBLE",
        "num_50": "DOUBLE",
        "num_75": "DOUBLE",
        "num_985": "DOUBLE",
        "num_100": "DOUBLE",
        "num_unq": "DOUBLE",
        "total_secs": "DOUBLE",
    }
    cutoff_sql = _sql_string(cutoff_date.strftime("%Y-%m-%d"))
    scan_sql = _duckdb_read_csv_sql(path, columns)
    window_aggregate_sql, window_output_sql = _build_user_log_window_sql(
        behavior_windows,
        cutoff_sql,
    )

    query = f"""
        WITH typed AS (
            SELECT
                msno,
                TRY_STRPTIME(date, '%Y%m%d')::DATE AS log_date,
                COALESCE(num_25, 0) AS num_25,
                COALESCE(num_50, 0) AS num_50,
                COALESCE(num_75, 0) AS num_75,
                COALESCE(num_985, 0) AS num_985,
                COALESCE(num_100, 0) AS num_100,
                COALESCE(num_unq, 0) AS num_unq,
                COALESCE(total_secs, 0) AS total_secs
            FROM {scan_sql}
        ),
        filtered AS (
            SELECT *
            FROM typed
            WHERE log_date IS NOT NULL
              AND log_date <= CAST({cutoff_sql} AS DATE)
        ),
        row_features AS (
            SELECT
                *,
                num_25 + num_50 + num_75 + num_985 + num_100 AS event_total,
                CASE
                    WHEN num_25 + num_50 + num_75 + num_985 + num_100 = 0 THEN NULL
                    ELSE num_100 / (num_25 + num_50 + num_75 + num_985 + num_100)
                END AS row_completion_rate
            FROM filtered
        ),
        grouped AS (
            SELECT
                msno,
                COUNT(*) AS active_days,
                SUM(total_secs) AS total_secs,
                SUM(num_25) AS total_25,
                SUM(num_50) AS total_50,
                SUM(num_75) AS total_75,
                SUM(num_985) AS total_985,
                SUM(num_100) AS total_100,
                SUM(num_unq) AS total_unq,
                SUM(COALESCE(row_completion_rate, 0)) AS completion_rate_sum,
                MAX(log_date) AS last_log_date
                {window_aggregate_sql}
            FROM row_features
            GROUP BY msno
        )
        SELECT
            msno,
            active_days,
            total_secs,
            CASE WHEN active_days = 0 THEN NULL ELSE total_secs / active_days END AS mean_secs,
            total_25,
            total_50,
            total_75,
            total_985,
            total_100,
            total_unq,
            CASE WHEN active_days = 0 THEN NULL ELSE total_unq / active_days END AS mean_unq,
            CASE
                WHEN total_25 + total_50 + total_75 + total_985 + total_100 = 0 THEN NULL
                ELSE total_100 / (total_25 + total_50 + total_75 + total_985 + total_100)
            END AS completion_rate,
            CASE
                WHEN active_days = 0 THEN NULL
                ELSE completion_rate_sum / active_days
            END AS mean_completion_rate,
            last_log_date,
            DATE_DIFF('day', last_log_date, CAST({cutoff_sql} AS DATE)) AS days_since_last_log
            {window_output_sql}
        FROM grouped
    """

    logger.info("Aggregating user logs with DuckDB from %s", path)
    with _connect_duckdb(config) as connection:
        result = connection.execute(query).fetchdf()
    logger.info("DuckDB user log aggregate complete: %s rows", result.shape[0])
    return result


def aggregate_user_logs(
    raw_dir: Path,
    chunksize: int,
    cutoff_date: pd.Timestamp,
    behavior_windows: list[int],
    config: dict | None = None,
) -> pd.DataFrame:
    """Aggregate listening logs to one row per user."""

    backend = str(get_value(config or {}, "ingestion", "backend", default="pandas")).lower()
    if backend == "duckdb":
        return _aggregate_user_logs_duckdb(raw_dir, cutoff_date, behavior_windows, config or {})
    if backend != "pandas":
        raise ValueError(f"Unsupported ingestion backend: {backend}")
    return _aggregate_user_logs_pandas(raw_dir, chunksize, cutoff_date, behavior_windows)


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
    cutoff_date = pd.Timestamp(get_value(config, "feature_engineering", "cutoff_date"))
    behavior_windows = [
        int(window)
        for window in get_value(config, "feature_engineering", "behavior_windows", default=[])
    ]

    logger.info("Loading compact tables from raw data")
    train = load_train(raw_dir)
    logger.info("train loaded: %s rows x %s columns", train.shape[0], train.shape[1])
    members = load_members(raw_dir)
    logger.info("members loaded: %s rows x %s columns", members.shape[0], members.shape[1])
    transactions = aggregate_transactions(raw_dir, chunksize, cutoff_date, config)
    user_logs = aggregate_user_logs(raw_dir, chunksize, cutoff_date, behavior_windows, config)

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
