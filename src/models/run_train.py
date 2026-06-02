"""Command-line entrypoint for the training stage.

Usage
-----
    python src/models/run_train.py

This script loads the project configuration, then calls ``run_training``
which:
  1. Loads OrdinalEncoded splits from ``data/processed/``.
  2. Builds a OneHot linear preprocessor for Logistic Regression.
  3. Trains baseline candidates (LR, RF, XGBoost, LightGBM).
  4. Runs Optuna + Top-K tuning for XGBoost and LightGBM.
  5. Selects the champion by validation AUC-PR.
  6. Saves models to ``models/`` and a comparison CSV to ``reports/``.

Prerequisites (run in order):
  1. python src/data/run_ingestion.py
  2. python src/features/run_engineer.py
  3. python src/features/run_preprocessing.py
  4. python src/models/run_train.py        ← this script
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import run_training
from src.utils.config import load_config
from src.utils.logger import setup_logger


if __name__ == "__main__":
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("src.models.train", log_file=logs_dir / "training.log")

    logger = logging.getLogger(__name__)
    logger.info(
        "Training stage started | project_root=%s | config=%s",
        PROJECT_ROOT,
        config_path,
    )

    champion, _all_results, _comparison_df = run_training(config, PROJECT_ROOT)

    logger.info(
        "Champion selected | model=%s | val_auc_pr=%.4f | val_auc_roc=%.4f",
        champion.name,
        champion.val_metrics.auc_pr,
        champion.val_metrics.auc_roc,
    )
    print("Training complete")
    print(
        "Champion:",
        champion.name,
        f"| val_auc_pr={champion.val_metrics.auc_pr:.4f}",
        f"| threshold={champion.decision_threshold:.4f}",
    )
    print("Reports:", PROJECT_ROOT / "reports" / "model_comparison.csv")
    print("Models :", PROJECT_ROOT / "models")
