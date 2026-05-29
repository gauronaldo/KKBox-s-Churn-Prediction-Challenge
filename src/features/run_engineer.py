"""Command-line entrypoint for the feature engineering stage."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import run_feature_engineering


if __name__ == "__main__":
    run_feature_engineering()
