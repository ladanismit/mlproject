"""Data loader module for the AGNews Text Classification project.

This module provides reusable, type-hinted, and robust functions to load,
validate, and summarize train and test datasets. It uses configs from
configs/config.py and utilities for logging and custom exception handling.
"""

import sys
from pathlib import Path

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from configs.config import (
    CLASS_LABELS,
    LABEL_COLUMN,
    NUM_CLASSES,
    TEST_FILE,
    TEXT_COLUMN,
    TRAIN_FILE,
)
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_file_exists(file_path: Path) -> None:
    """Validates that a file exists at the specified path and is not empty.

    Args:
        file_path (Path): Path to the file.

    Raises:
        CustomException: If the file does not exist, is a directory, or is empty.
    """
    try:
        if not file_path.exists():
            error_msg = f"Dataset file not found at: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        if not file_path.is_file():
            error_msg = f"Specified path is not a file: {file_path}"
            logger.error(error_msg)
            raise IsADirectoryError(error_msg)
        if file_path.stat().st_size == 0:
            error_msg = f"Dataset file is empty (0 bytes): {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"File validated successfully: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validates that all required columns are present in the DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame to validate.
        required_columns (list[str]): List of column names that must exist.

    Raises:
        CustomException: If any required column is missing.
    """
    try:
        missing_cols = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_cols:
            error_msg = (
                f"Missing required columns in DataFrame: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"All required columns validated: {required_columns}")
    except Exception as e:
        raise CustomException(e, sys) from e


def check_missing_values(
    df: pd.DataFrame, columns: list[str]
) -> dict[str, int]:
    """Checks and counts missing values in the specified columns.

    Args:
        df (pd.DataFrame): The pandas DataFrame to check.
        columns (list[str]): The columns to inspect.

    Returns:
        dict[str, int]: A dictionary mapping column names to missing counts.

    Raises:
        CustomException: If checking missing values fails.
    """
    try:
        missing_report = {}
        for col in columns:
            if col in df.columns:
                missing_count = int(df[col].isnull().sum())
                missing_report[col] = missing_count
                if missing_count > 0:
                    logger.warning(
                        f"Column '{col}' has {missing_count} missing value(s)."
                    )
                else:
                    logger.info(f"Column '{col}' has no missing values.")
            else:
                logger.warning(
                    f"Column '{col}' not found in DataFrame for missing check."
                )
        return missing_report
    except Exception as e:
        raise CustomException(e, sys) from e


def check_duplicate_rows(df: pd.DataFrame) -> int:
    """Counts and reports duplicate rows in the DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame to check.

    Returns:
        int: The number of duplicate rows found.

    Raises:
        CustomException: If checking duplicate rows fails.
    """
    try:
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            logger.warning(
                f"Found {duplicate_count} duplicate row(s) in the dataset."
            )
        else:
            logger.info("No duplicate rows found in the dataset.")
        return duplicate_count
    except Exception as e:
        raise CustomException(e, sys) from e


def validate_class_distribution(
    df: pd.DataFrame, label_column: str, expected_num_classes: int
) -> None:
    """Validates the classes/labels in the DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame to validate.
        label_column (str): The column containing class labels.
        expected_num_classes (int): The expected number of unique classes.

    Raises:
        CustomException: If validation of classes fails.
    """
    try:
        if label_column not in df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in DataFrame."
            )

        unique_classes = df[label_column].unique()
        num_unique_classes = len(unique_classes)

        logger.info(
            f"Unique classes found: {list(unique_classes)} "
            f"(Count: {num_unique_classes})"
        )

        if num_unique_classes != expected_num_classes:
            expected_keys = set(CLASS_LABELS.keys())
            actual_keys = set(unique_classes)
            missing_keys = expected_keys - actual_keys
            extra_keys = actual_keys - expected_keys

            error_msg = (
                f"Mismatch in expected number of classes. "
                f"Expected: {expected_num_classes}, Got: {num_unique_classes}. "
                f"Missing classes: {missing_keys}, Extra classes: {extra_keys}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Class distribution validation passed successfully.")
    except Exception as e:
        raise CustomException(e, sys) from e


def display_dataset_summary(
    df: pd.DataFrame, name: str, label_column: str
) -> None:
    """Displays and logs a comprehensive summary of the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        name (str): A friendly name of the dataset.
        label_column (str): The label/class column name.

    Raises:
        CustomException: If displaying summary fails.
    """
    try:
        summary_lines = [
            "=" * 60,
            f"DATASET SUMMARY: {name.upper()}",
            "=" * 60,
            f"Shape: {df.shape[0]} rows, {df.shape[1]} columns",
            "-" * 60,
            "Columns & Types:",
        ]
        for col in df.columns:
            summary_lines.append(f"  - {col}: {df[col].dtype}")

        summary_lines.append("-" * 60)

        # Missing values summary
        missing_counts = df.isnull().sum()
        summary_lines.append("Missing Values:")
        for col, count in missing_counts.items():
            summary_lines.append(f"  - {col}: {count}")

        summary_lines.append("-" * 60)
        summary_lines.append(f"Duplicate Rows: {df.duplicated().sum()}")
        summary_lines.append("-" * 60)

        # Class distribution summary
        if label_column in df.columns:
            class_counts = df[label_column].value_counts()
            class_percentages = (
                df[label_column].value_counts(normalize=True) * 100
            )

            summary_lines.append("Class Distribution:")
            for val, count in class_counts.items():
                label_name = CLASS_LABELS.get(val, "Unknown Label")
                pct = class_percentages[val]
                summary_lines.append(
                    f"  - Class {val} ({label_name}): "
                    f"{count} rows ({pct:.2f}%)"
                )
        else:
            summary_lines.append(
                f"Warning: Label column '{label_column}' not found. "
                "Cannot compute class distribution."
            )

        summary_lines.append("=" * 60)

        summary_text = "\n".join(summary_lines)
        logger.info(f"\n{summary_text}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_train_dataset(file_path: Path = TRAIN_FILE) -> pd.DataFrame:
    """Loads and validates the training dataset from the configured CSV file.

    Args:
        file_path (Path): Path to the training dataset file. Defaults to
            TRAIN_FILE.

    Returns:
        pd.DataFrame: Clean, validated pandas DataFrame.

    Raises:
        CustomException: If reading, validating, or parsing fails.
    """
    try:
        logger.info(f"Loading training dataset from: {file_path}")

        # 1. Validate file exists and is not empty
        validate_file_exists(file_path)

        # 2. Read dataset
        df = pd.read_csv(file_path)

        # 3. Handle column renaming to standard names in config
        rename_map = {
            "Class Index": LABEL_COLUMN,
            "Description": TEXT_COLUMN,
        }
        rename_dict = {k: v for k, v in rename_map.items() if k in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.info(f"Renamed columns for standardization: {rename_dict}")

        # 4. Validate required columns
        validate_columns(df, [TEXT_COLUMN, LABEL_COLUMN])

        # 5. Check missing values
        check_missing_values(df, [TEXT_COLUMN, LABEL_COLUMN])

        # 6. Check duplicate rows
        check_duplicate_rows(df)

        # 7. Validate classes
        validate_class_distribution(df, LABEL_COLUMN, NUM_CLASSES)

        # 8. Display summary
        display_dataset_summary(df, "Training Dataset", LABEL_COLUMN)

        return df
    except Exception as e:
        raise CustomException(e, sys) from e


def load_test_dataset(file_path: Path = TEST_FILE) -> pd.DataFrame:
    """Loads and validates the testing dataset from the configured CSV file.

    Args:
        file_path (Path): Path to the testing dataset file. Defaults to
            TEST_FILE.

    Returns:
        pd.DataFrame: Clean, validated pandas DataFrame.

    Raises:
        CustomException: If reading, validating, or parsing fails.
    """
    try:
        logger.info(f"Loading testing dataset from: {file_path}")

        # 1. Validate file exists and is not empty
        validate_file_exists(file_path)

        # 2. Read dataset
        df = pd.read_csv(file_path)

        # 3. Handle column renaming to standard names in config
        rename_map = {
            "Class Index": LABEL_COLUMN,
            "Description": TEXT_COLUMN,
        }
        rename_dict = {k: v for k, v in rename_map.items() if k in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.info(f"Renamed columns for standardization: {rename_dict}")

        # 4. Validate required columns
        validate_columns(df, [TEXT_COLUMN, LABEL_COLUMN])

        # 5. Check missing values
        check_missing_values(df, [TEXT_COLUMN, LABEL_COLUMN])

        # 6. Check duplicate rows
        check_duplicate_rows(df)

        # 7. Validate classes
        validate_class_distribution(df, LABEL_COLUMN, NUM_CLASSES)

        # 8. Display summary
        display_dataset_summary(df, "Testing Dataset", LABEL_COLUMN)

        return df
    except Exception as e:
        raise CustomException(e, sys) from e


def load_both_datasets(
    train_path: Path = TRAIN_FILE, test_path: Path = TEST_FILE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads both the training and testing datasets.

    Args:
        train_path (Path): Path to the training dataset file. Defaults to
            TRAIN_FILE.
        test_path (Path): Path to the testing dataset file. Defaults to
            TEST_FILE.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing (train_df,
            test_df).

    Raises:
        CustomException: If loading either dataset fails.
    """
    try:
        logger.info("Initializing loading of both train and test datasets.")
        train_df = load_train_dataset(train_path)
        test_df = load_test_dataset(test_path)
        logger.info("Both training and testing datasets loaded successfully.")
        return train_df, test_df
    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info("Starting standalone data loader verification...")
        train_data, test_data = load_both_datasets()
        print(f"Loaded train shape: {train_data.shape}")
        print(f"Loaded test shape: {test_data.shape}")
    except Exception as error:
        print(f"Verification failed: {error}")
