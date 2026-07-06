"""File I/O utility module for the AGNews Text Classification project.

This module provides generic, reusable, type-hinted helper functions for
manipulating the file system, handling serialization (JSON/Pickle), reading/writing
text, and performing metadata operations like size querying and safe deletions.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any

from src.utils.exception import CustomException
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def save_json(path: Path, data: Any) -> None:
    """Saves dictionary or list data to a JSON file.
    
    Automatically creates parent directories if they do not exist.
    
    Args:
        path (Path): Path where the JSON file will be saved.
        data (Any): Python object to serialize.
        
    Raises:
        CustomException: Wrapped exception if saving fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"JSON file saved successfully at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_json(path: Path) -> Any:
    """Loads and returns data from a JSON file.
    
    Args:
        path (Path): Path of the JSON file to read.
        
    Returns:
        Any: Deserialized JSON data.
        
    Raises:
        CustomException: Wrapped exception if loading fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path}")
        return data
    except Exception as e:
        raise CustomException(e, sys) from e


def save_pickle(path: Path, data: Any) -> None:
    """Serializes and saves a Python object as a binary pickle file.
    
    Automatically creates parent directories if they do not exist.
    
    Args:
        path (Path): Path where the pickle file will be saved.
        data (Any): Python object to serialize.
        
    Raises:
        CustomException: Wrapped exception if serialization fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Pickle file saved successfully at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_pickle(path: Path) -> Any:
    """Loads and returns a deserialized Python object from a binary pickle file.
    
    Args:
        path (Path): Path of the pickle file to read.
        
    Returns:
        Any: Deserialized Python object.
        
    Raises:
        CustomException: Wrapped exception if deserialization fails.
    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Pickle file loaded successfully from: {path}")
        return data
    except Exception as e:
        raise CustomException(e, sys) from e


def read_text(path: Path) -> str:
    """Reads and returns the content of a text file.
    
    Args:
        path (Path): Path of the text file to read.
        
    Returns:
        str: Content of the text file.
        
    Raises:
        CustomException: Wrapped exception if file reading fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"Text file read successfully from: {path}")
        return content
    except Exception as e:
        raise CustomException(e, sys) from e


def write_text(path: Path, content: str) -> None:
    """Writes text content to a file.
    
    Automatically creates parent directories if they do not exist.
    
    Args:
        path (Path): Destination file path.
        content (str): Text content to write.
        
    Raises:
        CustomException: Wrapped exception if writing fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Text file written successfully at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def file_exists(path: Path) -> bool:
    """Checks if a file exists at the given path.
    
    Args:
        path (Path): File path to verify.
        
    Returns:
        bool: True if the file exists and is a file, False otherwise.
        
    Raises:
        CustomException: Wrapped exception if validation fails.
    """
    try:
        exists = path.exists() and path.is_file()
        logger.debug(f"File existence check for [{path}]: {exists}")
        return exists
    except Exception as e:
        raise CustomException(e, sys) from e


def delete_file(path: Path) -> None:
    """Safely deletes a file if it exists in the system.
    
    Args:
        path (Path): Path of the file to delete.
        
    Raises:
        CustomException: Wrapped exception if deletion fails.
    """
    try:
        if path.exists():
            if path.is_file():
                path.unlink()
                logger.info(f"Successfully deleted file at: {path}")
            else:
                logger.warning(f"Path exists but is not a file: {path}")
        else:
            logger.debug(f"File to delete does not exist: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def get_file_size(path: Path) -> int:
    """Returns the size of the file in bytes.
    
    Args:
        path (Path): Path of the target file.
        
    Returns:
        int: File size in bytes, or -1 if the file does not exist.
        
    Raises:
        CustomException: Wrapped exception if file size query fails.
    """
    try:
        if path.exists() and path.is_file():
            size = path.stat().st_size
            logger.debug(f"File size of [{path}]: {size} bytes")
            return size
        logger.warning(f"File not found for size retrieval: {path}")
        return -1
    except Exception as e:
        raise CustomException(e, sys) from e


def list_files(directory_path: Path, pattern: str = "*") -> list[Path]:
    """Lists files matching the glob pattern inside a directory (non-recursive).
    
    Args:
        directory_path (Path): Directory to scan.
        pattern (str): Glob matching pattern. Defaults to '*'.
        
    Returns:
        list[Path]: List of matched file Paths.
        
    Raises:
        CustomException: Wrapped exception if scanning fails.
    """
    try:
        if directory_path.exists() and directory_path.is_dir():
            files = [p for p in directory_path.glob(pattern) if p.is_file()]
            logger.debug(f"Found {len(files)} files in [{directory_path}] matching pattern [{pattern}]")
            return files
        logger.warning(f"Directory not found for listing: {directory_path}")
        return []
    except Exception as e:
        raise CustomException(e, sys) from e
