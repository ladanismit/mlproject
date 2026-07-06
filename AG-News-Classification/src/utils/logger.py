"""Logger utility for the AGNews Text Classification project.

This module provides a reusable logging setup that routes log messages
to both the console (stdout) and a timestamped log file. It prevents
duplicate handler registration.
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Final

# Import logging configuration from the project configs
from configs.config import LOGS_DIR, LOG_FORMAT, LOG_LEVEL

# Automatically ensure log directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Generate a timestamped log filename
_TIMESTAMP: Final[str] = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FILE_PATH: Final[Path] = LOGS_DIR / f"project_{_TIMESTAMP}.log"

# Global flag to track root logger initialization
_is_initialized: bool = False


def _initialize_root_logger() -> None:
    """Configures the root logger with console and file handlers.
    
    This function is executed once to setup logging globally. Subsequent calls
    do nothing to prevent duplicate log handlers.
    """
    global _is_initialized
    if _is_initialized:
        return

    root_logger = logging.getLogger()
    
    # Parse and set the numeric log level
    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already configured elsewhere
    if not root_logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)

        # 1. Console Handler (outputs to stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 2. File Handler (outputs to timestamped log file)
        file_handler = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _is_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Retrieves a logger with the specified name and ensures logging is configured.
    
    Args:
        name (str): The name of the logger (typically __name__ of the module).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    _initialize_root_logger()
    return logging.getLogger(name)
