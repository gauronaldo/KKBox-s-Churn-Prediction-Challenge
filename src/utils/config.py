"""Configuration loading helpers for the KKBox churn project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the project configuration from YAML.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is empty or does not contain a mapping.
    """

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
    """Resolve a configured path value into a ``Path`` object.

    Args:
        config: Parsed project configuration.
        key: Name of the path entry inside ``section``.
        section: Top-level config section containing the path entry.
        base_dir: Optional base directory used for relative paths.

    Returns:
        Resolved path.

    Raises:
        KeyError: If the section or key is missing.
    """

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
    """Retrieve a nested value from the config dictionary.

    Args:
        config: Parsed project configuration.
        keys: Nested dictionary keys to traverse.
        default: Value returned when a key is missing.

    Returns:
        The nested config value or ``default``.
    """

    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current
import yaml
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_config(config_path: Path | str = "config/config.yaml") -> dict[str, Any]:
    """Load project configuration from a YAML file.

    Args:
        config_path: Path to the config YAML file.
                     Defaults to 'config/config.yaml'.

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from: {config_path}")
    return config


def get_path(config: dict[str, Any], key: str) -> Path:
    """Retrieve a path from config and return as a Path object.

    Args:
        config: Configuration dictionary loaded by load_config().
        key: Key name under config['paths'].

    Returns:
        Path object for the requested directory or file.

    Raises:
        KeyError: If the key does not exist under config['paths'].
    """
    try:
        raw_path = config["paths"][key]
    except KeyError:
        raise KeyError(f"Path key '{key}' not found in config['paths']")

    return Path(raw_path)