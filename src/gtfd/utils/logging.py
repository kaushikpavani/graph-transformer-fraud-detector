"""Logging utilities for GTFD."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure logging for GTFD.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file to write logs to.
        format_string: Custom format string.
    """
    # Remove default handler
    loguru_logger.remove()

    # Console handler
    fmt = format_string or (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    loguru_logger.add(sys.stderr, level=level, format=fmt, colorize=True)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        loguru_logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
        )


def get_logger(name: str) -> "loguru.Logger":
    """Get a logger instance for a module."""
    return loguru_logger.bind(name=name)


# Also provide standard logging compatibility
class LoguruHandler(logging.Handler):
    """Redirect standard logging to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_stdlib_logging(level: int = logging.INFO) -> None:
    """Set up standard library logging to redirect to loguru."""
    handler = LoguruHandler()
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)