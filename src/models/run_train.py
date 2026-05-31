"""Command-line entrypoint for the training stage.

Usage
-----
    python src/models/run_train.py

This script loads the project configuration, then calls ``run_training``
which:
  1. Loads OrdinalEncoded splits from ``data/processed/``.
  2. Builds a OneHot linear preprocessor for Logistic Regression.
  3. Trains all candidate models (LR, RF, XGBoost, LightGBM).
  4. Selects the champion by validation AUC-PR.
  5. Saves models to ``models/`` and a comparison CSV to ``reports/``.

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
    logger.info("=" * 60)
    logger.info("Training stage started")
    logger.info("Project root : %s", PROJECT_ROOT)
    logger.info("Config       : %s", config_path)
    logger.info("=" * 60)

    champion, all_results, comparison_df = run_training(config, PROJECT_ROOT)

    logger.info("Champion model : %s", champion.name)
    logger.info("Val AUC-PR     : %.4f", champion.val_metrics.auc_pr)
    logger.info("Val AUC-ROC    : %.4f", champion.val_metrics.auc_roc)
    print("Training outputs written to:")
    print(" -", PROJECT_ROOT / "models" / "champion_model.pkl")
    print(" -", PROJECT_ROOT / "models" / "champion_name.txt")
    print(" -", PROJECT_ROOT / "reports" / "model_comparison.csv")
    for result in all_results:
      print(" -", PROJECT_ROOT / "models" / f"{result.name}.pkl")
