"""
preprocessing.py — Text Preprocessing & Label Encoding for TicketVision-AI
==========================================================================

Pipeline stages
---------------
1. Load raw data via ``data_loader.load_dataset()``.
2. Merge ``product`` + ``issue_description`` into a unified ``text`` column.
3. Clean the text (lowercase, strip HTML/URLs/punctuation/numbers/whitespace).
4. Optionally remove stopwords and lemmatise with NLTK.
5. Encode the ``category`` column into integer labels (``LabelEncoder``).
6. Persist the cleaned DataFrame and the fitted encoder to disk.

Every diagnostic is emitted through Python's ``logging`` module.

Usage
-----
>>> from src.preprocessing import preprocess_dataset
>>> df = preprocess_dataset()                          # full pipeline
>>> df = preprocess_dataset(remove_stopwords=False)    # skip stopwords

Author : TicketVision-AI Team
Created: 2026-07-03
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Ensure project root is importable (same pattern as data_loader.py).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (  # noqa: E402
    COL_CATEGORY,
    COL_ISSUE_DESCRIPTION,
    COL_PRODUCT,
    COL_TEXT,
    LABEL_ENCODER_PATH,
    PROCESSED_DATASET_PATH,
    setup_logging,
)
from src.data_loader import load_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# NLTK bootstrap — download required corpora once (idempotent).
# ---------------------------------------------------------------------------
for _resource in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402

# Module-level logger.
logger: logging.Logger = logging.getLogger("ticketvision.preprocessing")

# Pre-compiled regex patterns (compiled once, reused on every row).
_RE_HTML_TAGS: re.Pattern    = re.compile(r"<[^>]+>")
_RE_URLS: re.Pattern         = re.compile(r"https?://\S+|www\.\S+")
_RE_NUMBERS: re.Pattern      = re.compile(r"\d+")
_RE_SPECIAL_CHARS: re.Pattern = re.compile(r"[^a-z\s]")
_RE_EXTRA_SPACES: re.Pattern = re.compile(r"\s{2,}")

# NLTK resources (initialised once at module level).
_STOP_WORDS: set[str]          = set(stopwords.words("english"))
_LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()


# ============================================================================
# 1. TEXT CLEANING
# ============================================================================

def clean_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
) -> str:
    """Apply the full text-cleaning pipeline to a single string.

    Parameters
    ----------
    text : str
        Raw input text.
    remove_stopwords : bool, default True
        Remove common English stopwords.
    lemmatize : bool, default True
        Reduce each token to its lemma.

    Returns
    -------
    str
        Cleaned, normalised text.
    """
    # 1. Lowercase.
    text = text.lower()

    # 2. Strip HTML tags.
    text = _RE_HTML_TAGS.sub("", text)

    # 3. Strip URLs.
    text = _RE_URLS.sub("", text)

    # 4. Remove numbers.
    text = _RE_NUMBERS.sub("", text)

    # 5. Remove punctuation & special characters (keep letters + spaces).
    text = _RE_SPECIAL_CHARS.sub(" ", text)

    # 6. Collapse extra whitespace.
    text = _RE_EXTRA_SPACES.sub(" ", text).strip()

    # 7. Tokenise.
    tokens: list[str] = text.split()

    # 8. Optional stopword removal.
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    # 9. Optional lemmatisation.
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ============================================================================
# 2. COLUMN MERGING
# ============================================================================

def _merge_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create the ``text`` column by combining ``product`` + ``issue_description``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``COL_PRODUCT`` and ``COL_ISSUE_DESCRIPTION`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new ``COL_TEXT`` column appended.
    """
    df[COL_TEXT] = (
        df[COL_PRODUCT].astype(str) + " " + df[COL_ISSUE_DESCRIPTION].astype(str)
    )
    logger.info(
        "Merged '%s' + '%s' into '%s' column.",
        COL_PRODUCT, COL_ISSUE_DESCRIPTION, COL_TEXT,
    )
    return df


# ============================================================================
# 3. LABEL ENCODING
# ============================================================================

def _encode_labels(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Fit a ``LabelEncoder`` on ``COL_CATEGORY`` and save the mapping.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the ``COL_CATEGORY`` column.
    save_path : Path, optional
        Where to persist the label mapping as JSON.
        Defaults to ``LABEL_ENCODER_PATH`` from config.

    Returns
    -------
    tuple[pd.DataFrame, LabelEncoder]
        The DataFrame with an added ``category_encoded`` column and the
        fitted encoder.
    """
    encoder = LabelEncoder()
    df["category_encoded"] = encoder.fit_transform(df[COL_CATEGORY])

    label_map: dict[str, int] = {
        label: int(idx) for idx, label in enumerate(encoder.classes_)
    }

    logger.info("Label encoding complete — %d classes.", len(label_map))
    for label, idx in label_map.items():
        logger.info("  %2d : %s", idx, label)

    # Persist the mapping.
    out_path: Path = save_path if save_path is not None else LABEL_ENCODER_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    logger.info("Label encoder saved to: %s", out_path)

    return df, encoder


# ============================================================================
# 4. PERSISTENCE
# ============================================================================

def _save_processed(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """Write the processed DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Fully processed DataFrame.
    save_path : Path, optional
        Destination CSV.  Defaults to ``PROCESSED_DATASET_PATH`` from config.

    Returns
    -------
    Path
        The path the file was written to.
    """
    out_path: Path = save_path if save_path is not None else PROCESSED_DATASET_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Processed dataset saved to: %s", out_path)
    return out_path


# ============================================================================
# 5. PUBLIC API
# ============================================================================

def preprocess_dataset(
    df: Optional[pd.DataFrame] = None,
    *,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    save: bool = True,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Run the full preprocessing pipeline end-to-end.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded raw DataFrame.  If ``None``, loads from
        ``RAW_DATASET_PATH`` via ``data_loader.load_dataset()``.
    remove_stopwords : bool, default True
        Whether to strip English stopwords.
    lemmatize : bool, default True
        Whether to lemmatise tokens.
    save : bool, default True
        Persist the cleaned CSV and label encoder to disk.

    Returns
    -------
    tuple[pd.DataFrame, LabelEncoder]
        The cleaned DataFrame and the fitted label encoder.

    Raises
    ------
    FileNotFoundError
        If the raw dataset cannot be located.
    KeyError
        If required columns are missing.
    ValueError
        If the dataset is empty after cleaning.
    """
    # --- 1. Load ---
    if df is None:
        logger.info("No DataFrame supplied — loading from raw dataset.")
        df = load_dataset()

    initial_rows: int = len(df)
    logger.info("Starting preprocessing on %s rows.", f"{initial_rows:,}")

    # --- 2. Drop rows with missing text / category ---
    required_cols = [COL_PRODUCT, COL_ISSUE_DESCRIPTION, COL_CATEGORY]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    dropped: int = initial_rows - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing values in required columns.", dropped)

    # --- 3. Merge columns ---
    df = _merge_text_columns(df)

    # --- 4. Clean text ---
    logger.info(
        "Cleaning text (stopwords=%s, lemmatize=%s) — this may take a moment...",
        remove_stopwords, lemmatize,
    )
    df[COL_TEXT] = df[COL_TEXT].apply(
        clean_text,
        remove_stopwords=remove_stopwords,
        lemmatize=lemmatize,
    )

    # Drop rows that became empty after cleaning.
    empty_mask = df[COL_TEXT].str.strip().eq("")
    n_empty: int = int(empty_mask.sum())
    if n_empty:
        logger.warning("Dropped %d rows with empty text after cleaning.", n_empty)
        df = df[~empty_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset is empty after preprocessing — nothing to train on.")

    logger.info("Text cleaning complete — %s rows remain.", f"{len(df):,}")

    # --- 5. Encode labels ---
    df, encoder = _encode_labels(df)

    # --- 6. Save ---
    if save:
        _save_processed(df)

    return df, encoder


# ============================================================================
# 6. SELF-TEST
# ============================================================================

def _separator(title: str, width: int = 60) -> None:
    """Print a formatted section header (self-test only)."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


if __name__ == "__main__":
    setup_logging()

    _separator("TicketVision-AI  -  Preprocessing Self-Test")

    try:
        processed_df, label_enc = preprocess_dataset()
    except (FileNotFoundError, KeyError, ValueError) as err:
        logger.critical("Preprocessing FAILED: %s", err)
        sys.exit(1)

    # ---- Sample cleaned records ----
    _separator("Sample Cleaned Records (first 5)")
    display_cols = [COL_TEXT, COL_CATEGORY, "category_encoded"]
    print(processed_df[display_cols].head().to_string(max_colwidth=90))

    _separator("Sample Cleaned Records (last 5)")
    print(processed_df[display_cols].tail().to_string(max_colwidth=90))

    # ---- Label mapping ----
    _separator("Label Mapping")
    for idx, label in enumerate(label_enc.classes_):
        print(f"  {idx:>2d}  ->  {label}")

    # ---- Dataset statistics ----
    _separator("Processed Dataset Statistics")
    print(f"  Total rows        : {len(processed_df):,}")
    print(f"  Total columns     : {processed_df.shape[1]}")
    print(f"  Unique categories : {processed_df[COL_CATEGORY].nunique()}")
    print(f"  Unique texts      : {processed_df[COL_TEXT].nunique():,}")

    avg_len: float = processed_df[COL_TEXT].str.split().str.len().mean()
    max_len: int   = int(processed_df[COL_TEXT].str.split().str.len().max())
    min_len: int   = int(processed_df[COL_TEXT].str.split().str.len().min())
    print(f"  Avg token length  : {avg_len:.1f}")
    print(f"  Max token length  : {max_len}")
    print(f"  Min token length  : {min_len}")

    mem_mb: float = processed_df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"  Memory usage      : {mem_mb:.2f} MB")

    # ---- Confirm files ----
    _separator("Saved Artefacts")
    print(f"  Processed CSV     : {PROCESSED_DATASET_PATH}")
    print(f"    -> exists: {PROCESSED_DATASET_PATH.exists()}")
    print(f"  Label Encoder     : {LABEL_ENCODER_PATH}")
    print(f"    -> exists: {LABEL_ENCODER_PATH.exists()}")

    _separator("Self-Test Complete")
    print(f"  Preprocessing finished successfully.\n")
