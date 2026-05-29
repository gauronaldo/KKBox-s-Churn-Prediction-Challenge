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
    run_preprocessing(config, PROJECT_ROOT)
