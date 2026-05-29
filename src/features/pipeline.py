"""Feature engineering pipeline entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.features.engineer import build_feature_frame, save_feature_summary
from src.utils.config import get_path, get_value, load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def run_feature_engineering(config_path: Path | str = Path("config") / "config.yaml") -> pd.DataFrame:
    """Run the feature engineering stage and persist its outputs.

    Args:
        config_path: Path to the project configuration file.

    Returns:
        The engineered feature dataframe.
    """

    config = load_config(config_path)
    project_root = Path(config_path).resolve().parents[1]
    logs_dir = get_path(config, "logs_dir")
    if not logs_dir.is_absolute():
        logs_dir = project_root / logs_dir
    log_file = logs_dir / "feature_engineering.log"
    setup_logger(__name__, log_file=log_file)

    interim_dir = get_path(config, "interim_dir")
    if not interim_dir.is_absolute():
        interim_dir = project_root / interim_dir
    processed_dir = get_path(config, "processed_dir")
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    modeling_frame_path = interim_dir / "modeling_frame.parquet"
    if not modeling_frame_path.exists():
        raise FileNotFoundError(f"Modeling frame not found: {modeling_frame_path}")

    logger.info("Loading modeling frame from %s", modeling_frame_path)
    modeling_frame = pd.read_parquet(modeling_frame_path)

    feature_frame, summary = build_feature_frame(modeling_frame, config)

    output_file = get_value(config, "feature_engineering", "output_file", default="feature_frame.parquet")
    metadata_file = get_value(config, "feature_engineering", "metadata_file", default="feature_metadata.json")
    feature_frame_path = processed_dir / str(output_file)
    metadata_path = processed_dir / str(metadata_file)

    feature_frame.to_parquet(feature_frame_path, index=False)
    save_feature_summary(summary, metadata_path)

    logger.info("Saved feature frame to %s", feature_frame_path)
    logger.info("Saved feature metadata to %s", metadata_path)
    logger.info("Feature frame shape: %s", feature_frame.shape)

    return feature_frame
