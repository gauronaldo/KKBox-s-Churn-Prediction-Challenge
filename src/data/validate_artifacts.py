"""Command-line validation for persisted pipeline artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation import validate_interim_artifacts, validate_processed_artifacts
from src.utils.config import get_path, load_config


def main() -> None:
    """Validate interim and processed artifacts written by the pipeline."""

    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    interim_dir = get_path(config, "interim_dir", base_dir=PROJECT_ROOT)
    processed_dir = get_path(config, "processed_dir", base_dir=PROJECT_ROOT)

    validate_interim_artifacts(interim_dir, config)
    validate_processed_artifacts(processed_dir, config)
    print("Artifact validation passed.")


if __name__ == "__main__":
    main()
