"""Configuration loading helpers for the KKBox churn project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the project configuration from YAML."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    return config


def get_path(
    config: Mapping[str, Any],
    key: str,
    *,
    section: str = "paths",
    base_dir: Path | None = None,
) -> Path:
    """Resolve a configured path value into a ``Path`` object."""

    if section not in config:
        raise KeyError(f"Missing config section: {section}")

    section_data = config[section]
    if not isinstance(section_data, Mapping):
        raise KeyError(f"Config section '{section}' must be a mapping.")
    if key not in section_data:
        raise KeyError(f"Missing config path: {section}.{key}")

    path = Path(section_data[key])
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def get_value(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Retrieve a nested value from the config dictionary."""

    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current