"""Command-line entrypoint for the feature engineering stage."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import run_feature_engineering
from src.utils.config import get_path, get_value, load_config


if __name__ == "__main__":
    feature_frame = run_feature_engineering()
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    processed_dir = get_path(config, "processed_dir")
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir
    output_file = get_value(config, "feature_engineering", "output_file", default="feature_frame.parquet")
    metadata_file = get_value(config, "feature_engineering", "metadata_file", default="feature_metadata.json")
    print("Feature engineering output shape:", feature_frame.shape)
    print("Feature engineering outputs written to:")
    print(" -", processed_dir / str(output_file))
    print(" -", processed_dir / str(metadata_file))
