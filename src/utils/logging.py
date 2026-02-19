"""Logging helpers with console and file handlers."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create or return a configured logger.

    Parameters
    ----------
    name:
        Logger name.
    log_file:
        Optional path for file logging.
    level:
        Logger level.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
