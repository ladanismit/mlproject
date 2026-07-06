"""Common helper utilities for the AGNews Text Classification project.

This module provides generic, reusable file and directory operations, serialization/
deserialization helpers, and timestamp generators. It integrates custom exception
handling and standard logging.
"""

from datetime import datetime
import json
import pickle
import sys
from pathlib import Path
from typing import Any

from src.utils.exception import CustomException
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def create_directories(paths: list[Path] | Path, verbose: bool = True) -> None:
    """Creates a directory or a list of directories if they do not exist.
    
    Args:
        paths (list[Path] | Path): Path or list of Path objects to be created.
        verbose (bool): If True, logs the creation of directories.
        
    Raises:
        CustomException: Wrapped exception detailing directory creation failure.
    """
    try:
        if isinstance(paths, Path):
            paths = [paths]
            
        for path in paths:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                if verbose:
                    logger.info(f"Created directory at: {path}")
            else:
                if verbose:
                    logger.debug(f"Directory already exists at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Saves dictionary data to a JSON file.
    
    Args:
        path (Path): Destination path for the JSON file.
        data (dict[str, Any]): Dictionary containing data to serialize.
        
    Raises:
        CustomException: Wrapped exception detailing JSON save failure.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"JSON file saved successfully at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_json(path: Path) -> dict[str, Any]:
    """Loads and returns dictionary data from a JSON file.
    
    Args:
        path (Path): Source path of the JSON file.
        
    Returns:
        dict[str, Any]: Deserialized dictionary contents.
        
    Raises:
        CustomException: Wrapped exception detailing JSON load failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path}")
        return data
    except Exception as e:
        raise CustomException(e, sys) from e


def save_bin(path: Path, data: Any) -> None:
    """Serializes and saves a Python object as a binary pickle file.
    
    Args:
        path (Path): Destination path for the binary file.
        data (Any): Python object to serialize.
        
    Raises:
        CustomException: Wrapped exception detailing binary save failure.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Binary file saved successfully at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_bin(path: Path) -> Any:
    """Loads and returns a deserialized Python object from a binary pickle file.
    
    Args:
        path (Path): Source path of the binary file.
        
    Returns:
        Any: Deserialized Python object.
        
    Raises:
        CustomException: Wrapped exception detailing binary load failure.
    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Binary file loaded successfully from: {path}")
        return data
    except Exception as e:
        raise CustomException(e, sys) from e


def file_exists(path: Path) -> bool:
    """Checks if a file exists and is a valid file.
    
    Args:
        path (Path): Path to examine.
        
    Returns:
        bool: True if the file exists and is a file, False otherwise.
        
    Raises:
        CustomException: Wrapped exception detailing file system check failure.
    """
    try:
        exists = path.exists() and path.is_file()
        logger.debug(f"File existence check for [{path}]: {exists}")
        return exists
    except Exception as e:
        raise CustomException(e, sys) from e


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Generates the current local timestamp formatted as a string.
    
    Args:
        format_str (str): Datetime formatting template. Defaults to '%Y%m%d_%H%M%S'.
        
    Returns:
        str: Formatted local timestamp.
        
    Raises:
        CustomException: Wrapped exception detailing timestamp generation failure.
    """
    try:
        return datetime.now().strftime(format_str)
    except Exception as e:
        raise CustomException(e, sys) from e
