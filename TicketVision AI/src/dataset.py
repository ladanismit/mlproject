"""
dataset.py — Dataset Splitting, Tokenization & tf.data Pipelines for TicketVision-AI
=====================================================================================

Pipeline stages
---------------
1. Load the processed DataFrame via ``preprocessing.preprocess_dataset()``.
2. Split into Train (70 %), Validation (15 %), Test (15 %) using stratified
   sampling so every class is proportionally represented.
3. Fit a ``tf.keras.preprocessing.text.Tokenizer`` **only on the training
   split** to prevent data leakage.
4. Convert text to integer sequences, then pad / truncate to a fixed length.
5. Persist the fitted tokenizer to disk (JSON) for deterministic inference.
6. Build optimised ``tf.data.Dataset`` pipelines with batching, shuffling,
   caching, and prefetching.

All diagnostics are emitted through Python's ``logging`` module.

Usage
-----
>>> from src.dataset import prepare_datasets
>>> train_ds, val_ds, test_ds, tokenizer, info = prepare_datasets()

Author : TicketVision-AI Team
Created: 2026-07-04
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

# ---------------------------------------------------------------------------
# Ensure project root is importable (same pattern as data_loader.py).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (  # noqa: E402
    BATCH_SIZE,
    COL_TEXT,
    MAX_SEQUENCE_LENGTH,
    OOV_TOKEN,
    PADDING_TYPE,
    SEED,
    TOKENIZER_PATH,
    TRAIN_DATASET_PATH,
    TRUNCATION_TYPE,
    VAL_DATASET_PATH,
    TEST_DATASET_PATH,
    VALIDATION_SPLIT,
    VOCAB_SIZE,
    setup_logging,
)
from src.preprocessing import preprocess_dataset  # noqa: E402

# Module-level logger.
logger: logging.Logger = logging.getLogger("ticketvision.dataset")

# Column produced by preprocessing.py that holds integer labels.
_COL_LABEL: str = "category_encoded"

# Train fraction (the remainder is split equally between val and test).
_TRAIN_FRACTION: float = 1.0 - 2 * VALIDATION_SPLIT  # 0.70

# Relative size of the validation set within the combined val+test block.
_VAL_RELATIVE: float = VALIDATION_SPLIT / (2 * VALIDATION_SPLIT)  # 0.50


# ============================================================================
# 1. STRATIFIED SPLITTING
# ============================================================================

def _stratified_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / validation / test with stratified sampling.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with ``COL_TEXT`` and ``_COL_LABEL`` columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``

    Raises
    ------
    ValueError
        If a class has fewer than 2 samples (stratification impossible).
    """
    labels: pd.Series = df[_COL_LABEL]

    # First split: train vs. (val + test).
    train_df, temp_df = train_test_split(
        df,
        test_size=2 * VALIDATION_SPLIT,         # 0.30
        random_state=SEED,
        stratify=labels,
    )

    # Second split: val vs. test (50 / 50 of the remaining 30 %).
    val_df, test_df = train_test_split(
        temp_df,
        test_size=_VAL_RELATIVE,                 # 0.50
        random_state=SEED,
        stratify=temp_df[_COL_LABEL],
    )

    # Reset indices for clean downstream iteration.
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Stratified split complete — Train: %s | Val: %s | Test: %s",
        f"{len(train_df):,}", f"{len(val_df):,}", f"{len(test_df):,}",
    )

    return train_df, val_df, test_df


# ============================================================================
# 2. TOKENIZATION & SEQUENCE ENCODING
# ============================================================================

def _fit_tokenizer(train_texts: pd.Series) -> Tokenizer:
    """Fit a Keras Tokenizer **on training texts only**.

    Parameters
    ----------
    train_texts : pd.Series
        The cleaned text column from the training split.

    Returns
    -------
    Tokenizer
        Fitted tokenizer instance.
    """
    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token=OOV_TOKEN,
    )
    tokenizer.fit_on_texts(train_texts.tolist())

    actual_vocab: int = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
    logger.info(
        "Tokenizer fitted — %s unique tokens, vocab capped at %s.",
        f"{len(tokenizer.word_index):,}", f"{actual_vocab:,}",
    )
    return tokenizer


def _texts_to_padded_sequences(
    tokenizer: Tokenizer,
    texts: pd.Series,
) -> np.ndarray:
    """Convert texts to padded / truncated integer sequences.

    Parameters
    ----------
    tokenizer : Tokenizer
        Previously fitted tokenizer.
    texts : pd.Series
        Text column to encode.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_samples, MAX_SEQUENCE_LENGTH)``.
    """
    sequences = tokenizer.texts_to_sequences(texts.tolist())
    padded: np.ndarray = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding=PADDING_TYPE,
        truncating=TRUNCATION_TYPE,
    )
    return padded


# ============================================================================
# 3. TOKENIZER PERSISTENCE
# ============================================================================

def _save_tokenizer(
    tokenizer: Tokenizer,
    save_path: Optional[Path] = None,
) -> Path:
    """Serialise the fitted tokenizer to a JSON file.

    Parameters
    ----------
    tokenizer : Tokenizer
        Fitted tokenizer to persist.
    save_path : Path, optional
        Destination path.  Defaults to ``TOKENIZER_PATH`` from config.

    Returns
    -------
    Path
        The path the tokenizer was written to.
    """
    out_path: Path = save_path if save_path is not None else TOKENIZER_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_json: str = tokenizer.to_json()
    out_path.write_text(tokenizer_json, encoding="utf-8")

    logger.info("Tokenizer saved to: %s", out_path)
    return out_path


def load_tokenizer(load_path: Optional[Path] = None) -> Tokenizer:
    """Load a previously saved tokenizer from disk.

    Parameters
    ----------
    load_path : Path, optional
        Path to the JSON file.  Defaults to ``TOKENIZER_PATH`` from config.

    Returns
    -------
    Tokenizer
        Restored tokenizer instance.

    Raises
    ------
    FileNotFoundError
        If the tokenizer file does not exist.
    """
    in_path: Path = load_path if load_path is not None else TOKENIZER_PATH
    if not in_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {in_path}")

    tokenizer_json: str = in_path.read_text(encoding="utf-8")
    tokenizer: Tokenizer = tokenizer_from_json(tokenizer_json)
    logger.info("Tokenizer loaded from: %s", in_path)
    return tokenizer


# ============================================================================
# 4. SPLIT PERSISTENCE
# ============================================================================

def _save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Persist each split as a CSV for reproducibility / auditing.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        The three stratified splits.
    """
    for df, path, name in [
        (train_df, TRAIN_DATASET_PATH, "Train"),
        (val_df, VAL_DATASET_PATH, "Validation"),
        (test_df, TEST_DATASET_PATH, "Test"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("%s split saved to: %s (%s rows)", name, path, f"{len(df):,}")


# ============================================================================
# 5. tf.data.Dataset PIPELINE CONSTRUCTION
# ============================================================================

def _build_tf_dataset(
    sequences: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    buffer_size: int = 10_000,
) -> tf.data.Dataset:
    """Build an optimised ``tf.data.Dataset`` from NumPy arrays.

    Parameters
    ----------
    sequences : np.ndarray
        Padded integer sequences, shape ``(n, MAX_SEQUENCE_LENGTH)``.
    labels : np.ndarray
        Integer-encoded labels, shape ``(n,)``.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle (should be ``True`` only for training).
    buffer_size : int
        Shuffle-buffer size (ignored when ``shuffle=False``).

    Returns
    -------
    tf.data.Dataset
        A dataset that yields ``(sequence_batch, label_batch)`` tuples.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (sequences.astype(np.int32), labels.astype(np.int32)),
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = (
        dataset
        .batch(batch_size, drop_remainder=False)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


# ============================================================================
# 6. PUBLIC API
# ============================================================================

def prepare_datasets(
    df: Optional[pd.DataFrame] = None,
    *,
    batch_size: int = BATCH_SIZE,
    save: bool = True,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    Tokenizer,
    Dict[str, Any],
]:
    """End-to-end pipeline: preprocess → split → tokenize → tf.data.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-processed DataFrame.  If ``None``, runs the full preprocessing
        pipeline via ``preprocess_dataset()``.
    batch_size : int
        Batch size for the ``tf.data.Dataset`` pipelines.
    save : bool
        Persist splits, tokenizer, and label encoder to disk.

    Returns
    -------
    tuple
        ``(train_ds, val_ds, test_ds, tokenizer, info)``

        * ``train_ds`` / ``val_ds`` / ``test_ds`` — ``tf.data.Dataset``
          yielding ``(sequences, labels)`` batches.
        * ``tokenizer`` — the fitted ``Tokenizer`` instance.
        * ``info`` — metadata dict with keys ``num_classes``,
          ``vocab_size``, ``train_size``, ``val_size``, ``test_size``.

    Raises
    ------
    FileNotFoundError
        If the raw dataset cannot be found upstream.
    KeyError
        If required columns are missing.
    ValueError
        If the dataset is empty or a class has too few samples for
        stratification.
    """
    # ------------------------------------------------------------------
    # 1. Obtain the processed DataFrame.
    # ------------------------------------------------------------------
    if df is None:
        logger.info("No DataFrame supplied — running preprocessing pipeline.")
        df, _ = preprocess_dataset()

    required_cols = [COL_TEXT, _COL_LABEL]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise KeyError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )

    logger.info(
        "Preparing datasets from %s rows with %d classes.",
        f"{len(df):,}", df[_COL_LABEL].nunique(),
    )

    # ------------------------------------------------------------------
    # 2. Stratified train / val / test split.
    # ------------------------------------------------------------------
    train_df, val_df, test_df = _stratified_split(df)

    if save:
        _save_splits(train_df, val_df, test_df)

    # ------------------------------------------------------------------
    # 3. Fit tokenizer on training data ONLY.
    # ------------------------------------------------------------------
    tokenizer: Tokenizer = _fit_tokenizer(train_df[COL_TEXT])

    if save:
        _save_tokenizer(tokenizer)

    # ------------------------------------------------------------------
    # 4. Convert all splits to padded sequences.
    # ------------------------------------------------------------------
    logger.info(
        "Encoding sequences — max_len=%d, padding='%s', truncating='%s'.",
        MAX_SEQUENCE_LENGTH, PADDING_TYPE, TRUNCATION_TYPE,
    )

    train_seq = _texts_to_padded_sequences(tokenizer, train_df[COL_TEXT])
    val_seq = _texts_to_padded_sequences(tokenizer, val_df[COL_TEXT])
    test_seq = _texts_to_padded_sequences(tokenizer, test_df[COL_TEXT])

    train_labels = train_df[_COL_LABEL].values
    val_labels = val_df[_COL_LABEL].values
    test_labels = test_df[_COL_LABEL].values

    logger.info("Sequence shapes — Train: %s | Val: %s | Test: %s",
                train_seq.shape, val_seq.shape, test_seq.shape)

    # ------------------------------------------------------------------
    # 5. Build tf.data.Dataset pipelines.
    # ------------------------------------------------------------------
    train_ds = _build_tf_dataset(
        train_seq, train_labels,
        batch_size=batch_size, shuffle=True,
    )
    val_ds = _build_tf_dataset(
        val_seq, val_labels,
        batch_size=batch_size, shuffle=False,
    )
    test_ds = _build_tf_dataset(
        test_seq, test_labels,
        batch_size=batch_size, shuffle=False,
    )

    logger.info("tf.data pipelines built (batch_size=%d).", batch_size)

    # ------------------------------------------------------------------
    # 6. Assemble metadata dict.
    # ------------------------------------------------------------------
    actual_vocab: int = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
    info: Dict[str, Any] = {
        "num_classes": int(df[_COL_LABEL].nunique()),
        "vocab_size": actual_vocab,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "batch_size": batch_size,
    }

    logger.info("Dataset preparation complete.")
    return train_ds, val_ds, test_ds, tokenizer, info


# ============================================================================
# 7. SELF-TEST
# ============================================================================

def _separator(title: str, width: int = 60) -> None:
    """Print a formatted section header (self-test only)."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


if __name__ == "__main__":
    setup_logging()

    _separator("TicketVision-AI  —  Dataset Pipeline Self-Test")

    try:
        train_ds, val_ds, test_ds, tokenizer, info = prepare_datasets()
    except (FileNotFoundError, KeyError, ValueError) as err:
        logger.critical("Dataset pipeline FAILED: %s", err)
        sys.exit(1)

    # ---- Split sizes ----
    _separator("Dataset Split Sizes")
    print(f"  Train samples      : {info['train_size']:,}")
    print(f"  Validation samples : {info['val_size']:,}")
    print(f"  Test samples       : {info['test_size']:,}")
    total: int = info["train_size"] + info["val_size"] + info["test_size"]
    print(f"  Total              : {total:,}")
    print(f"  Train ratio        : {info['train_size'] / total:.2%}")
    print(f"  Val ratio          : {info['val_size'] / total:.2%}")
    print(f"  Test ratio         : {info['test_size'] / total:.2%}")

    # ---- Vocabulary ----
    _separator("Tokenizer & Vocabulary")
    print(f"  Vocabulary size    : {info['vocab_size']:,}")
    print(f"  Max sequence len   : {info['max_sequence_length']}")
    print(f"  OOV token          : '{OOV_TOKEN}'")
    print(f"  Padding type       : '{PADDING_TYPE}'")
    print(f"  Truncation type    : '{TRUNCATION_TYPE}'")
    print(f"  Num classes        : {info['num_classes']}")

    # ---- Sample tokenised sequences ----
    _separator("Sample Tokenized Sequences (first 3 from training)")
    # Re-load the training CSV to grab the original texts for display.
    if TRAIN_DATASET_PATH.exists():
        _train_df = pd.read_csv(TRAIN_DATASET_PATH)
        sample_texts = _train_df[COL_TEXT].head(3).tolist()
    else:
        sample_texts = ["(training CSV not saved — using placeholder)"]

    for i, text in enumerate(sample_texts):
        seq = tokenizer.texts_to_sequences([text])[0]
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            [seq],
            maxlen=MAX_SEQUENCE_LENGTH,
            padding=PADDING_TYPE,
            truncating=TRUNCATION_TYPE,
        )[0]
        print(f"\n  Sample {i + 1}:")
        print(f"    Text (first 120 chars) : {text[:120]}...")
        print(f"    Sequence length (raw)  : {len(seq)}")
        print(f"    Sequence (first 15)    : {seq[:15]}")
        print(f"    Padded  (first 15)     : {padded[:15].tolist()}")
        print(f"    Padded  (last  15)     : {padded[-15:].tolist()}")

    # ---- Decode a sequence back to text ----
    _separator("Decoded Sequence Verification")
    if TRAIN_DATASET_PATH.exists() and len(sample_texts) > 0:
        first_text = sample_texts[0]
        first_seq = tokenizer.texts_to_sequences([first_text])[0]
        # Build reverse word index.
        reverse_index: Dict[int, str] = {
            idx: word for word, idx in tokenizer.word_index.items()
        }
        decoded_tokens = [
            reverse_index.get(idx, "<?>") for idx in first_seq[:20]
        ]
        print(f"  Original  : {first_text[:120]}...")
        print(f"  Decoded   : {' '.join(decoded_tokens)}...")

    # ---- Batch shapes ----
    _separator("Batch Shapes")
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        for seq_batch, lbl_batch in ds.take(1):
            print(
                f"  {name:<6s} — sequences: {seq_batch.shape}  |  "
                f"labels: {lbl_batch.shape}  |  "
                f"dtype: {seq_batch.dtype}"
            )

    # ---- Pipeline throughput check (iterate one full epoch) ----
    _separator("Pipeline Verification (full iteration)")
    import time

    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        n_batches = 0
        n_samples = 0
        t0 = time.perf_counter()
        for seq_batch, lbl_batch in ds:
            n_batches += 1
            n_samples += seq_batch.shape[0]
        elapsed = time.perf_counter() - t0
        print(
            f"  {name:<6s} — {n_batches:>5,} batches | "
            f"{n_samples:>8,} samples | "
            f"{elapsed:.3f}s"
        )

    # ---- Saved artefacts ----
    _separator("Saved Artefacts")
    for label, path in [
        ("Tokenizer", TOKENIZER_PATH),
        ("Train CSV", TRAIN_DATASET_PATH),
        ("Val CSV", VAL_DATASET_PATH),
        ("Test CSV", TEST_DATASET_PATH),
    ]:
        status = "OK" if path.exists() else "MISSING"
        size_kb = path.stat().st_size / 1024 if path.exists() else 0
        print(f"  [{status}] {label:<12s} : {path}  ({size_kb:,.1f} KB)")

    _separator("Self-Test Complete")
    print("  Dataset pipeline finished successfully.\n")
