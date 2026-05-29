"""Logging helpers for the KKBox churn project."""

from __future__ import annotations

import logging
from pathlib import Path


def _coerce_level(level: int | str) -> int:
    """Convert a string or integer logging level into an integer value."""

    if isinstance(level, int):
        return level

    numeric_level = logging.getLevelName(level.upper())
    if isinstance(numeric_level, int):
        return numeric_level
    raise ValueError(f"Invalid logging level: {level}")


def setup_logger(
    name: str,
    level: int | str = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Create a consistent project logger.

    Args:
        name: Logger name.
        level: Logging level, either numeric or textual.
        log_file: Optional path to a log file.

    Returns:
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(_coerce_level(level))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    fmt: str = "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Set up a logger with console and optional file handler.

    Args:
        name: Logger name, typically __name__ of the calling module.
        log_file: Optional path to write logs to a file.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        fmt: Log message format string.
        datefmt: Date format string for log timestamps.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when the notebook cell is re-run.
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a logger by name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    fmt: str = "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Set up a logger with console and optional file handler.

    Args:
        name: Logger name, typically __name__ of the calling module.
        log_file: Optional path to write logs to a file.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        fmt: Log message format string.
        datefmt: Date format string for log timestamps.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Tránh duplicate handlers nếu logger đã được setup trước đó
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler — luôn có
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — chỉ thêm nếu log_file được chỉ định
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve an existing logger by name.

    Intended for modules that call setup_logger() at entry point
    and get_logger() everywhere else.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Logger instance (may not be configured if setup_logger
        was not called first).
    """
    return logging.getLogger(name)