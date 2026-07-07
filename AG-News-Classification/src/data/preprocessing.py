"""Text preprocessing module for the AGNews Text Classification project.

This module provides modular, reusable functions for cleaning and preprocessing
text strings and Pandas DataFrames. It supports converting to lowercase,
removing HTML tags, URLs, email addresses, extra whitespaces, and special
characters, as well as Unicode normalization and missing value handling.
"""

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from configs.config import TEXT_COLUMN
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Compile regex patterns for efficiency
HTML_TAG_PATTERN = re.compile(r"<[^>]*>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def handle_missing_empty(text: Any, default_value: str = "") -> str:
    """Handles missing, null, or non-string inputs.

    Converts valid inputs to string and returns default_value for invalid
    or empty inputs.

    Args:
        text (Any): Input object (can be None, float, NaN, str, etc.).
        default_value (str): Default fallback value. Defaults to "".

    Returns:
        str: Validated text string.
    """
    if pd.isna(text) or text is None:
        return default_value

    val_str = str(text).strip()
    return val_str if val_str else default_value


def normalize_unicode(text: str, form: str = "NFKD") -> str:
    """Normalizes Unicode text to standard forms.

    Args:
        text (str): Input text string.
        form (str): Unicode normalization form (e.g., 'NFC', 'NFKC', 'NFD',
            'NFKD'). Defaults to 'NFKD'.

    Returns:
        str: Normalized text string.
    """
    return unicodedata.normalize(form, text)


def remove_html_tags(text: str) -> str:
    """Removes HTML tags from a text string.

    Args:
        text (str): Input text string.

    Returns:
        str: Text with HTML tags removed.
    """
    return HTML_TAG_PATTERN.sub("", text)


def remove_urls(text: str) -> str:
    """Removes HTTP/HTTPS URLs and www. links from a text string.

    Args:
        text (str): Input text string.

    Returns:
        str: Text with URLs removed.
    """
    return URL_PATTERN.sub("", text)


def remove_email_addresses(text: str) -> str:
    """Removes email addresses from a text string.

    Args:
        text (str): Input text string.

    Returns:
        str: Text with emails removed.
    """
    return EMAIL_PATTERN.sub("", text)


def remove_special_characters(text: str) -> str:
    """Removes special characters from a text string, keeping only alphanumeric and whitespace.

    Args:
        text (str): Input text string.

    Returns:
        str: Text with special characters removed.
    """
    return SPECIAL_CHAR_PATTERN.sub("", text)


def remove_extra_whitespaces(text: str) -> str:
    """Collapses multiple spaces, tabs, or newlines into a single space and strips.

    Args:
        text (str): Input text string.

    Returns:
        str: Text with extra whitespaces cleaned.
    """
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def lowercase_text(text: str) -> str:
    """Converts a text string to lowercase.

    Args:
        text (str): Input text string.

    Returns:
        str: Lowercase text string.
    """
    return text.lower()


def clean_text(
    text: Any,
    lowercase: bool = True,
    remove_html: bool = True,
    remove_url: bool = True,
    remove_email: bool = True,
    remove_special_chars: bool = True,
    unicode_normalization: Optional[str] = "NFKD",
    default_value: str = "",
) -> str:
    """Applies modular cleaning steps to a single input value.

    Args:
        text (Any): The raw input to clean.
        lowercase (bool): Convert text to lowercase. Defaults to True.
        remove_html (bool): Remove HTML tags. Defaults to True.
        remove_url (bool): Remove URL addresses. Defaults to True.
        remove_email (bool): Remove email addresses. Defaults to True.
        remove_special_chars (bool): Remove special characters (keep only
            alphanumeric and spaces). Defaults to True.
        unicode_normalization (Optional[str]): Unicode normalization form.
            Defaults to 'NFKD'. If None, normalization is skipped.
        default_value (str): Value to return if raw text is empty/missing.
            Defaults to "".

    Returns:
        str: Cleaned text string.
    """
    # 1. Handle missing/empty/non-string values
    cleaned = handle_missing_empty(text, default_value=default_value)
    if not cleaned:
        return default_value

    # 2. Normalize Unicode
    if unicode_normalization:
        cleaned = normalize_unicode(cleaned, form=unicode_normalization)

    # 3. Convert to lowercase
    if lowercase:
        cleaned = lowercase_text(cleaned)

    # 4. Remove HTML tags
    if remove_html:
        cleaned = remove_html_tags(cleaned)

    # 5. Remove URLs
    if remove_url:
        cleaned = remove_urls(cleaned)

    # 6. Remove email addresses
    if remove_email:
        cleaned = remove_email_addresses(cleaned)

    # 7. Remove special characters
    if remove_special_chars:
        cleaned = remove_special_characters(cleaned)

    # 8. Collapse extra whitespaces (should always run at the end to clean up patterns)
    cleaned = remove_extra_whitespaces(cleaned)

    return cleaned


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = TEXT_COLUMN,
    target_column: Optional[str] = None,
    lowercase: bool = True,
    remove_html: bool = True,
    remove_url: bool = True,
    remove_email: bool = True,
    remove_special_chars: bool = True,
    unicode_normalization: Optional[str] = "NFKD",
) -> pd.DataFrame:
    """Applies the cleaning pipeline to a specified column in a Pandas DataFrame.

    Returns a clean copy of the DataFrame without modifying the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): The column containing text to clean. Defaults to
            TEXT_COLUMN.
        target_column (Optional[str]): The column name to save cleaned text to.
            If None, overwrites text_column in the returned DataFrame. Defaults
            to None.
        lowercase (bool): Convert text to lowercase. Defaults to True.
        remove_html (bool): Remove HTML tags. Defaults to True.
        remove_url (bool): Remove URLs. Defaults to True.
        remove_email (bool): Remove emails. Defaults to True.
        remove_special_chars (bool): Remove special characters. Defaults to
            True.
        unicode_normalization (Optional[str]): Unicode normalization form.
            Defaults to 'NFKD'.

    Returns:
        pd.DataFrame: A new DataFrame with the cleaned column.

    Raises:
        CustomException: Wrapped exception if DataFrame cleaning fails.
    """
    try:
        logger.info(f"Preprocessing text column '{text_column}' in DataFrame...")
        if text_column not in df.columns:
            error_msg = f"Column '{text_column}' not found in the DataFrame."
            logger.error(error_msg)
            raise KeyError(error_msg)

        # Create a deep copy to ensure original DataFrame is unmodified
        df_copy = df.copy()
        dest_col = target_column if target_column is not None else text_column

        # Apply cleanup row-wise
        df_copy[dest_col] = df_copy[text_column].apply(
            lambda x: clean_text(
                text=x,
                lowercase=lowercase,
                remove_html=remove_html,
                remove_url=remove_url,
                remove_email=remove_email,
                remove_special_chars=remove_special_chars,
                unicode_normalization=unicode_normalization,
            )
        )

        logger.info(f"Preprocessing completed. Saved results to '{dest_col}'.")
        return df_copy
    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info("Starting standalone preprocessing verification...")

        # Test 1: Single text string
        sample_text = (
            "Check out <b>AG News</b> at http://example.com/agnews. "
            "Contact us at info@example.com for details! Text with Unicode: "
            "Café and naïve, and some extra   spaces."
        )
        print("Raw text:")
        print(f"  {sample_text}")

        cleaned = clean_text(sample_text)
        print("Cleaned text:")
        print(f"  {cleaned}")

        # Test 2: DataFrame mapping
        data = {
            "title": ["Sports Update", "Sci/Tech News", "Missing Text"],
            "description": [
                "The game ended 9-0! Visit www.sports.com.",
                "New AI chip released by TechCorp <br> (info@techcorp.com)",
                None,
            ],
        }
        test_df = pd.DataFrame(data)
        print("\nRaw DataFrame:")
        print(test_df)

        preprocessed_df = preprocess_dataframe(
            test_df,
            text_column="description",
            target_column="cleaned_description",
        )
        print("\nPreprocessed DataFrame:")
        print(preprocessed_df)

        print("\nRaw DataFrame remains unmodified:")
        print(test_df)

    except Exception as error:
        print(f"Verification failed: {error}")
