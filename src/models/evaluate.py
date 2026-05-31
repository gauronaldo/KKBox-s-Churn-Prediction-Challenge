"""Evaluate the saved champion model on the held-out test split."""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_path, get_value, load_config


def _load_pickle(path: Path) -> Any:
    """Load a pickle artifact without importing the training stack."""

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _predict_churn_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted classifier."""

    if not hasattr(model, "predict_proba"):
        raise TypeError("Champion model must expose predict_proba().")
    return model.predict_proba(X)[:, 1]


def _lift_at_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    """Compute churn lift among the highest-scored fraction of users."""

    n_top = max(1, int(len(y_score) * fraction))
    top_indices = np.argsort(y_score)[::-1][:n_top]
    baseline = float(y_true.mean())
    if baseline == 0.0:
        return 0.0
    return float(y_true[top_indices].mean()) / baseline


def evaluate_champion(config_path: Path = PROJECT_ROOT / "config" / "config.yaml") -> pd.DataFrame:
    """Evaluate the persisted champion on the test split and save a CSV report."""

    config = load_config(config_path)
    processed_dir = get_path(config, "processed_dir", base_dir=PROJECT_ROOT)
    models_dir = get_path(config, "models_dir", base_dir=PROJECT_ROOT)
    reports_dir = get_path(config, "reports_dir", base_dir=PROJECT_ROOT)
    reports_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(get_value(config, "modeling", "decision_threshold"))
    lift_fraction = float(get_value(config, "analysis", "lift_top_fraction"))

    model_path = models_dir / str(get_value(config, "artifacts", "champion_model_file"))
    champion_name_path = models_dir / str(get_value(config, "artifacts", "champion_name_file"))
    X_test_path = processed_dir / "X_test.parquet"
    y_test_path = processed_dir / "y_test.parquet"

    for path in [model_path, champion_name_path, X_test_path, y_test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation artifact not found: {path}")

    champion_name = champion_name_path.read_text(encoding="utf-8").strip()
    champion_threshold_path = models_dir / str(get_value(config, "artifacts", "champion_threshold_file", default="champion_threshold.txt"))
    if champion_threshold_path.exists():
        threshold = float(champion_threshold_path.read_text(encoding="utf-8").strip())

    model = _load_pickle(model_path)
    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path).iloc[:, 0].astype(int).to_numpy()

    y_score = _predict_churn_proba(model, X_test)
    y_pred = (y_score >= threshold).astype(int)

    report = pd.DataFrame(
        [
            {
                "model": champion_name,
                "split": "test",
                "rows": len(y_test),
                "positive_rate": float(y_test.mean()),
                "threshold": threshold,
                "auc_roc": float(roc_auc_score(y_test, y_score)),
                "auc_pr": float(average_precision_score(y_test, y_score)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "log_loss": float(log_loss(y_test, y_score, labels=[0, 1])),
                "lift_at_top_fraction": _lift_at_fraction(y_test, y_score, lift_fraction),
                "top_fraction": lift_fraction,
            }
        ]
    )

    output_path = reports_dir / str(get_value(config, "artifacts", "test_evaluation_file"))
    report.to_csv(output_path, index=False)
    print(f"Saved test evaluation to: {output_path}")
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    evaluate_champion()
