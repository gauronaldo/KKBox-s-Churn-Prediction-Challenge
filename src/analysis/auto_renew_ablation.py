"""Run an isolated ablation experiment for auto-renewal feature families."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_path, get_value, load_config


def _load_split(processed_dir: Path, name: str) -> pd.DataFrame | pd.Series:
    """Load a processed split artifact."""

    path = processed_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")
    frame = pd.read_parquet(path)
    if name.startswith("y_"):
        return frame.iloc[:, 0].astype("int8")
    return frame


def _drop_feature_family(
    frame: pd.DataFrame,
    patterns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose names contain any configured ablation pattern."""

    lowered_patterns = [pattern.lower() for pattern in patterns]
    drop_columns = [
        column
        for column in frame.columns
        if any(pattern in column.lower() for pattern in lowered_patterns)
    ]
    return frame.drop(columns=drop_columns), drop_columns


def _scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    """Compute the XGBoost class-imbalance weight from training labels."""

    values = np.asarray(y_train)
    positive_count = int((values == 1).sum())
    negative_count = int((values == 0).sum())
    if positive_count == 0:
        raise ValueError("Training labels contain no churn examples.")
    return negative_count / positive_count


def _lift_at_fraction(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    fraction: float,
) -> float:
    """Compute churn lift in the highest-scored fraction."""

    y_array = np.asarray(y_true)
    top_count = max(1, int(len(y_array) * fraction))
    top_indices = np.argsort(y_score)[::-1][:top_count]
    baseline_rate = float(y_array.mean())
    if baseline_rate == 0.0:
        return 0.0
    return float(y_array[top_indices].mean()) / baseline_rate


def _tune_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str,
    grid_step: float,
) -> float:
    """Tune a decision threshold on validation predictions."""

    thresholds = np.arange(grid_step, 1.0, grid_step)
    if thresholds.size == 0:
        return 0.5

    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "f2":
            # Churn intervention usually values recall more than precision.
            score = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    lift_fraction: float,
) -> dict[str, float]:
    """Compute the ablation comparison metrics."""

    y_pred = (y_score >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "lift_at_top_fraction": _lift_at_fraction(y_true, y_score, lift_fraction),
    }


def _xgboost_params(config: Mapping[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Build XGBoost parameters from project config."""

    return {
        "n_estimators": int(get_value(config, "xgboost", "n_estimators")),
        "learning_rate": float(get_value(config, "xgboost", "learning_rate")),
        "max_depth": int(get_value(config, "xgboost", "max_depth")),
        "subsample": float(get_value(config, "xgboost", "subsample")),
        "colsample_bytree": float(get_value(config, "xgboost", "colsample_bytree")),
        "tree_method": str(get_value(config, "xgboost", "tree_method")),
        "eval_metric": str(get_value(config, "xgboost", "eval_metric")),
        "early_stopping_rounds": int(get_value(config, "xgboost", "early_stopping_rounds")),
        "scale_pos_weight": _scale_pos_weight(y_train),
        "random_state": int(get_value(config, "project", "random_state")),
        "n_jobs": int(get_value(config, "xgboost", "n_jobs", default=-1)),
        "verbosity": 0,
    }


def run_auto_renew_ablation(
    config_path: Path = PROJECT_ROOT / "config" / "config.yaml",
) -> pd.DataFrame:
    """Train and evaluate XGBoost after dropping auto-renew feature families."""

    config = load_config(config_path)
    processed_dir = get_path(config, "processed_dir", base_dir=PROJECT_ROOT)
    reports_dir = get_path(config, "reports_dir", base_dir=PROJECT_ROOT)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train = _load_split(processed_dir, "X_train")
    X_val = _load_split(processed_dir, "X_val")
    y_train = _load_split(processed_dir, "y_train")
    y_val = _load_split(processed_dir, "y_val")

    patterns = list(get_value(config, "analysis", "auto_renew_ablation_feature_patterns"))
    X_train_ablation, dropped_columns = _drop_feature_family(X_train, patterns)
    X_val_ablation = X_val.drop(columns=dropped_columns, errors="ignore")
    if not dropped_columns:
        raise ValueError(f"No columns matched ablation patterns: {patterns}")

    print("Auto-renew ablation patterns:", patterns)
    print("Dropped columns:", dropped_columns)
    print("Training shape:", X_train.shape, "->", X_train_ablation.shape)

    params = _xgboost_params(config, y_train)
    log_eval_period = int(get_value(config, "xgboost", "log_eval_period"))
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train_ablation,
        y_train,
        eval_set=[(X_val_ablation, y_val)],
        verbose=log_eval_period,
    )

    train_score = model.predict_proba(X_train_ablation)[:, 1]
    val_score = model.predict_proba(X_val_ablation)[:, 1]
    threshold = _tune_threshold(
        y_val,
        val_score,
        metric=str(get_value(config, "modeling", "decision_threshold_metric")),
        grid_step=float(get_value(config, "modeling", "decision_threshold_grid_step")),
    )
    lift_fraction = float(get_value(config, "analysis", "lift_top_fraction"))

    rows: list[dict[str, Any]] = []
    for split_name, y_true, y_score in [
        ("train", y_train, train_score),
        ("val", y_val, val_score),
    ]:
        row: dict[str, Any] = {
            "experiment": "xgboost_auto_renew_ablation",
            "split": split_name,
            "rows": int(len(y_true)),
            "features_before": int(X_train.shape[1]),
            "features_after": int(X_train_ablation.shape[1]),
            "features_dropped": int(len(dropped_columns)),
            "dropped_columns": "|".join(dropped_columns),
            "threshold": threshold,
            "best_iteration": int(getattr(model, "best_iteration", params["n_estimators"])),
        }
        row.update(_metrics(y_true, y_score, threshold=threshold, lift_fraction=lift_fraction))
        rows.append(row)

    report = pd.DataFrame(rows)
    output_path = reports_dir / str(get_value(config, "analysis", "auto_renew_ablation_report_file"))
    report.to_csv(output_path, index=False)
    print(f"Wrote ablation report to: {output_path}")
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    run_auto_renew_ablation()
