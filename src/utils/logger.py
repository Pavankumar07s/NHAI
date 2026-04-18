"""
logger.py — Structured logging for HighwayRetroAI.

Uses *loguru* with console + file output.  Console is colorized; file output
is plain-text with 10 MB rotation and 5-backup retention.

Usage:
    from src.utils.logger import logger
    logger.info("Pipeline started")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

# Remove the default loguru handler so we can configure our own.
_loguru_logger.remove()

# ---------------------------------------------------------------------------
# Console handler (colorized)
# ---------------------------------------------------------------------------
_loguru_logger.add(
    sys.stderr,
    level="DEBUG",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)


def setup_logger(
    name: str = "HighwayRetroAI",
    log_file: Path | str | None = None,
    level: str = "INFO",
) -> "loguru.Logger":
    """Return a configured *loguru* logger.

    Parameters
    ----------
    name : str
        Logical logger name (shown in the ``{name}`` field).
    log_file : Path | str | None
        Optional path for a rotating file sink.
    level : str
        Minimum severity to emit (``DEBUG``, ``INFO``, ``WARNING``, …).

    Returns
    -------
    loguru.Logger
        The configured logger instance.
    """
    if log_file is not None:
        _loguru_logger.add(
            str(log_file),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            rotation="10 MB",
            retention=5,
            enqueue=True,
        )
    return _loguru_logger.bind(name=name)


# Default logger instance — importable everywhere.
logger = setup_logger()
