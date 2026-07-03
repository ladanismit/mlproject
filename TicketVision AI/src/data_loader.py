"""
data_loader.py — Dataset Loading & Validation for TicketVision-AI
=================================================================

Responsible for reading the raw CSV dataset, validating its schema,
auditing data quality (missing values, duplicates, class balance),
and returning a clean ``pandas.DataFrame`` ready for downstream
preprocessing.

Every diagnostic is emitted through Python's ``logging`` module —
no ``print()`` statements are used outside the self-test block.

Usage
-----
>>> from data_loader import load_dataset
>>> df = load_dataset()                 # uses RAW_DATASET_PATH from config
>>> df = load_dataset(path=custom_path) # override path for experiments

Author : TicketVision-AI Team
Created: 2026-07-03
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root (parent of src/) is on sys.path so that
# ``config`` is importable regardless of how this module is invoked:
#   - python src/data_loader.py        (direct execution)
#   - python -m src.data_loader        (module execution)
#   - from src.data_loader import ...  (library import)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Project imports — single source of truth for paths & column names.
# ---------------------------------------------------------------------------
from config import (  # noqa: E402
    COL_CATEGORY,
    COL_ISSUE_DESCRIPTION,
    COL_PRODUCT,
    RAW_DATASET_PATH,
    setup_logging,
)

# Module-level logger.
logger: logging.Logger = logging.getLogger("ticketvision.data_loader")

# The three columns every raw dataset file *must* contain.
REQUIRED_COLUMNS: List[str] = [COL_PRODUCT, COL_ISSUE_DESCRIPTION, COL_CATEGORY]


# ============================================================================
# 1. SCHEMA VALIDATION
# ============================================================================

def _validate_columns(df: pd.DataFrame) -> None:
    """Verify that all required columns are present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The freshly-loaded DataFrame.

    Raises
    ------
    KeyError
        If one or more required columns are missing.
    """
    missing: set[str] = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(
            f"Dataset is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
    logger.info("Column validation passed — all required columns present.")


# ============================================================================
# 2. DATA-QUALITY CHECKS
# ============================================================================

def _check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Log per-column missing-value counts and return the Series.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame.

    Returns
    -------
    pd.Series
        Count of missing values per column.
    """
    missing: pd.Series = df.isnull().sum()
    total_missing: int = int(missing.sum())

    if total_missing == 0:
        logger.info("Missing values   : None detected.")
    else:
        logger.warning("Missing values   : %d total across %d column(s).",
                       total_missing, int((missing > 0).sum()))
        for col, count in missing[missing > 0].items():
            pct: float = count / len(df) * 100
            logger.warning("  %-25s : %d (%.2f%%)", col, count, pct)

    return missing


def _check_duplicates(df: pd.DataFrame) -> int:
    """Log the number of fully-duplicated rows.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame.

    Returns
    -------
    int
        Number of duplicate rows.
    """
    n_dupes: int = int(df.duplicated().sum())
    if n_dupes == 0:
        logger.info("Duplicate rows   : None detected.")
    else:
        logger.warning("Duplicate rows   : %d (%.2f%% of dataset).",
                       n_dupes, n_dupes / len(df) * 100)
    return n_dupes


def _log_shape(df: pd.DataFrame) -> None:
    """Log the dataset dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame.
    """
    rows, cols = df.shape
    logger.info("Dataset shape    : %s rows x %s columns.", f"{rows:,}", cols)


def _log_dtypes(df: pd.DataFrame) -> None:
    """Log per-column data types.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame.
    """
    logger.info("Data types:")
    for col in df.columns:
        logger.info("  %-25s : %s", col, df[col].dtype)


def _log_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Log the target-label distribution and return the value-counts Series.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame containing ``COL_CATEGORY``.

    Returns
    -------
    pd.Series
        Value counts for each category, sorted descending.
    """
    counts: pd.Series = df[COL_CATEGORY].value_counts()
    n_classes: int = len(counts)
    logger.info("Class distribution (%d unique categories):", n_classes)
    for label, count in counts.items():
        pct: float = count / len(df) * 100
        logger.info("  %-30s : %6d  (%5.2f%%)", label, count, pct)

    return counts


# ============================================================================
# 3. PUBLIC API
# ============================================================================

def load_dataset(
    path: Optional[Path] = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load, validate, and audit the raw customer-support-ticket dataset.

    Parameters
    ----------
    path : Path, optional
        Filesystem path to the CSV file.  Defaults to ``RAW_DATASET_PATH``
        from ``config.py``.
    validate : bool, default True
        When ``True``, run schema validation and all data-quality checks.
        Set to ``False`` for a fast, unchecked load.

    Returns
    -------
    pd.DataFrame
        The validated DataFrame ready for preprocessing.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the resolved path.
    KeyError
        If any required column is missing from the CSV.
    pd.errors.EmptyDataError
        If the CSV file is empty.
    """
    dataset_path: Path = path if path is not None else RAW_DATASET_PATH

    # --- File existence check ---
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            "Ensure the CSV is placed in the expected location or "
            "pass an explicit 'path' argument."
        )

    logger.info("Loading dataset from: %s", dataset_path)

    try:
        df: pd.DataFrame = pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError:
        logger.error("The dataset file is empty: %s", dataset_path)
        raise
    except pd.errors.ParserError as exc:
        logger.error("CSV parsing failed for %s: %s", dataset_path, exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error loading dataset: %s", exc)
        raise

    if df.empty:
        logger.warning("Dataset loaded but contains 0 rows.")
        return df

    logger.info("Dataset loaded successfully.")

    # --- Validation & quality audit ---
    if validate:
        _validate_columns(df)
        _log_shape(df)
        _log_dtypes(df)
        _check_missing_values(df)
        _check_duplicates(df)
        _log_class_distribution(df)

    return df


# ============================================================================
# 4. SELF-TEST
# ============================================================================

def _separator(title: str, width: int = 60) -> None:
    """Print a formatted section header (used only in __main__ self-test)."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


if __name__ == "__main__":
    # Bootstrap the project logger so all messages reach the console & log file.
    setup_logging()

    _separator("TicketVision-AI  -  Data Loader Self-Test")

    try:
        data: pd.DataFrame = load_dataset()
    except (FileNotFoundError, KeyError) as err:
        logger.critical("Self-test FAILED: %s", err)
        sys.exit(1)

    # ---- Extended summary (printed to stdout for quick visual inspection) ----
    _separator("Dataset Head (first 5 rows)")
    print(data.head().to_string(max_colwidth=80))

    _separator("Dataset Tail (last 5 rows)")
    print(data.tail().to_string(max_colwidth=80))

    _separator("Descriptive Statistics")
    print(data.describe(include="all").to_string())

    _separator("Memory Usage")
    mem_mb: float = data.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"  Total memory usage: {mem_mb:.2f} MB")

    _separator("Sample Values per Column")
    for col in data.columns:
        n_unique: int = data[col].nunique()
        samples: list = data[col].dropna().unique()[:5].tolist()
        print(f"  {col} ({n_unique} unique): {samples}")

    _separator("Self-Test Complete")
    print(f"  Loaded {len(data):,} rows x {data.shape[1]} columns.")
    print(f"  All checks passed.\n")
