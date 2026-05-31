"""Command-line entrypoint for the preprocessing stage."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocess import run_preprocessing
from src.utils.config import load_config


if __name__ == "__main__":
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    splits = run_preprocessing(config, PROJECT_ROOT)
    processed_dir = PROJECT_ROOT / "data" / "processed"
    print("Preprocessing outputs written to:", processed_dir)
    for name in [
        "X_train.parquet",
        "X_val.parquet",
        "X_test.parquet",
        "y_train.parquet",
        "y_val.parquet",
        "y_test.parquet",
    ]:
        print(" -", processed_dir / name)
    print(" -", PROJECT_ROOT / "models" / "preprocessor.pkl")
    print("Split shapes:")
    print(" - X_train:", splits.X_train.shape)
    print(" - X_val  :", splits.X_val.shape)
    print(" - X_test :", splits.X_test.shape)
