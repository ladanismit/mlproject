"""Centralized logging module for DocuMind AI.

This module provides a preconfigured logger for the entire application,
formatting console log records with consistent timestamps, levels, and module names.
"""

import logging
import sys
from app.core.config import settings

# Standard log format: timestamp | level | logger name | message
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "documind") -> logging.Logger:
    """Create and return a configured logger instance.

    Args:
        name: Name of the logger, typically __name__ of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Resolve log level from settings (fallback to INFO if unrecognized)
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Attach handler only if not already attached to prevent duplicate output
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Avoid propagation to parent/root loggers to eliminate duplicate log lines
    logger.propagate = False

    return logger
