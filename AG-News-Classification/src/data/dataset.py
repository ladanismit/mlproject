"""Dataset pipeline module for the AGNews Text Classification project.

This module provides reusable, high-performance, and type-hinted functions
to build, split, shuffle, batch, cache, and prefetch TensorFlow datasets
using the tf.data API. It supports train-validation splitting and integrates
with the project configs, logging, and error handling framework.
"""

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.config import BATCH_SIZE, RANDOM_SEED, VALIDATION_SPLIT
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_tf_dataset(
    features: np.ndarray, labels: Optional[np.ndarray] = None
) -> tf.data.Dataset:
    """Creates a basic tf.data.Dataset from features and optional labels.

    Args:
        features (np.ndarray): Feature array (e.g. tokenized/padded sequences).
        labels (Optional[np.ndarray]): Target labels. Defaults to None.

    Returns:
        tf.data.Dataset: The constructed TensorFlow dataset.

    Raises:
        CustomException: If dataset creation fails.
    """
    try:
        logger.info(f"Creating tf.data.Dataset with features shape: {features.shape}")
        if labels is not None:
            logger.info(f"Adding labels with shape: {labels.shape}")
            # Ensure elements match in length
            if len(features) != len(labels):
                raise ValueError("Features and labels must have the same length.")
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(features)
        return dataset
    except Exception as e:
        raise CustomException(e, sys) from e


def split_train_val(
    features: np.ndarray,
    labels: np.ndarray,
    validation_split: float = VALIDATION_SPLIT,
    random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits training features and labels into training and validation subsets.

    Args:
        features (np.ndarray): Complete training features.
        labels (np.ndarray): Complete training labels.
        validation_split (float): Split ratio (fraction of data for validation).
            Defaults to VALIDATION_SPLIT.
        random_seed (int): Random seed for deterministic shuffling. Defaults to
            RANDOM_SEED.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (train_features, val_features, train_labels, val_labels).

    Raises:
        CustomException: If splitting fails.
    """
    try:
        logger.info(
            f"Splitting dataset with validation_split={validation_split} "
            f"and random_seed={random_seed}"
        )
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("Validation split must be in range [0.0, 1.0).")

        num_samples = len(features)
        if len(labels) != num_samples:
            raise ValueError("Features and labels must have the same length.")

        if validation_split == 0.0:
            logger.info("Validation split is 0.0. Skipping splitting.")
            return features, np.empty((0,) + features.shape[1:], dtype=features.dtype), labels, np.empty((0,), dtype=labels.dtype)

        # Generate a deterministic permuted sequence of indices
        rng = np.random.default_rng(seed=random_seed)
        indices = rng.permutation(num_samples)

        val_size = int(num_samples * validation_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_features = features[train_indices]
        val_features = features[val_indices]
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]

        logger.info(
            f"Split complete. Training: {len(train_features)} samples, "
            f"Validation: {len(val_features)} samples."
        )
        return train_features, val_features, train_labels, val_labels
    except Exception as e:
        raise CustomException(e, sys) from e


def optimize_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    buffer_size: Optional[int] = None,
    cache_filepath: Optional[str] = None,
    random_seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """Applies shuffling, batching, caching, and prefetching optimizations.

    Args:
        dataset (tf.data.Dataset): The source tf.data.Dataset.
        batch_size (int): Batch size. Defaults to BATCH_SIZE.
        shuffle (bool): Whether to shuffle the dataset. Defaults to False.
        buffer_size (Optional[int]): Size of shuffle buffer. If None, buffer
            size is set dynamically.
        cache_filepath (Optional[str]): File path to write cache data. If empty
            string "", caches in memory. If None, skips caching. Defaults to None.
        random_seed (int): Random seed for shuffling. Defaults to RANDOM_SEED.

    Returns:
        tf.data.Dataset: The optimized tf.data.Dataset.

    Raises:
        CustomException: If optimization fails.
    """
    try:
        opt_ds = dataset

        # 1. Shuffle
        if shuffle:
            if buffer_size is None:
                cardinality = int(dataset.cardinality().numpy())
                if cardinality > 0:
                    buffer_size = cardinality
                else:
                    buffer_size = 10000  # Fallback buffer size

            logger.debug(
                f"Applying shuffle with buffer_size={buffer_size} "
                f"and seed={random_seed}"
            )
            opt_ds = opt_ds.shuffle(
                buffer_size=buffer_size,
                seed=random_seed,
                reshuffle_each_iteration=True,
            )

        # 2. Batch
        logger.debug(f"Applying batching with batch_size={batch_size}")
        opt_ds = opt_ds.batch(batch_size, drop_remainder=False)

        # 3. Cache (Optional)
        if cache_filepath is not None:
            if cache_filepath == "":
                logger.debug("Applying in-memory caching")
                opt_ds = opt_ds.cache()
            else:
                logger.debug(f"Applying file-based caching at: {cache_filepath}")
                Path(cache_filepath).parent.mkdir(parents=True, exist_ok=True)
                opt_ds = opt_ds.cache(cache_filepath)

        # 4. Prefetch
        logger.debug("Applying prefetch with AUTOTUNE")
        opt_ds = opt_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return opt_ds
    except Exception as e:
        raise CustomException(e, sys) from e


def create_model_datasets(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: Optional[np.ndarray] = None,
    validation_split: float = VALIDATION_SPLIT,
    batch_size: int = BATCH_SIZE,
    random_seed: int = RANDOM_SEED,
    cache: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Generates training, validation, and testing tf.data.Dataset objects.

    Applies pipeline optimizations to output ready-to-train datasets.

    Args:
        train_features (np.ndarray): Complete training feature array.
        train_labels (np.ndarray): Complete training label array.
        test_features (np.ndarray): Complete testing feature array.
        test_labels (Optional[np.ndarray]): Testing labels. Defaults to None.
        validation_split (float): Portion of training data for validation.
            Defaults to VALIDATION_SPLIT.
        batch_size (int): Size of batches. Defaults to BATCH_SIZE.
        random_seed (int): Random seed for shuffling/splitting. Defaults to
            RANDOM_SEED.
        cache (bool): Whether to cache the dataset in memory. Defaults to False.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
            (train_dataset, validation_dataset, testing_dataset).

    Raises:
        CustomException: If dataset generation fails.
    """
    try:
        logger.info("Initializing model datasets creation pipeline...")

        # 1. Split training and validation arrays
        t_feats, v_feats, t_lbls, v_lbls = split_train_val(
            features=train_features,
            labels=train_labels,
            validation_split=validation_split,
            random_seed=random_seed,
        )

        # 2. Create raw datasets
        train_raw_ds = create_tf_dataset(t_feats, t_lbls)
        val_raw_ds = create_tf_dataset(v_feats, v_lbls)
        test_raw_ds = create_tf_dataset(test_features, test_labels)

        # 3. Optimize datasets
        cache_val = "" if cache else None

        logger.info("Optimizing training dataset (shuffled, batched, prefetched)...")
        train_ds = optimize_dataset(
            train_raw_ds,
            batch_size=batch_size,
            shuffle=True,
            cache_filepath=cache_val,
            random_seed=random_seed,
        )

        logger.info("Optimizing validation dataset (batched, prefetched)...")
        val_ds = optimize_dataset(
            val_raw_ds,
            batch_size=batch_size,
            shuffle=False,
            cache_filepath=cache_val,
        )

        logger.info("Optimizing testing dataset (batched, prefetched)...")
        test_ds = optimize_dataset(
            test_raw_ds,
            batch_size=batch_size,
            shuffle=False,
            cache_filepath=cache_val,
        )

        logger.info("Model datasets successfully created and optimized.")
        return train_ds, val_ds, test_ds
    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info("Starting standalone dataset pipeline verification...")

        # 1. Create mock arrays
        mock_train_features = np.random.randint(low=1, high=100, size=(1000, 10))
        mock_train_labels = np.random.randint(low=0, high=4, size=(1000,))
        mock_test_features = np.random.randint(low=1, high=100, size=(200, 10))
        mock_test_labels = np.random.randint(low=0, high=4, size=(200,))

        # 2. Create datasets
        tr_ds, val_ds, ts_ds = create_model_datasets(
            train_features=mock_train_features,
            train_labels=mock_train_labels,
            test_features=mock_test_features,
            test_labels=mock_test_labels,
            validation_split=0.2,
            batch_size=32,
            cache=True,
        )

        # 3. Verify batch sizes and outputs
        for feat_batch, lbl_batch in tr_ds.take(1):
            print(f"Train batch features shape: {feat_batch.shape}")
            print(f"Train batch labels shape:   {lbl_batch.shape}")

        for feat_batch, lbl_batch in val_ds.take(1):
            print(f"Val batch features shape:   {feat_batch.shape}")
            print(f"Val batch labels shape:     {lbl_batch.shape}")

        for feat_batch, lbl_batch in ts_ds.take(1):
            print(f"Test batch features shape:  {feat_batch.shape}")
            print(f"Test batch labels shape:    {lbl_batch.shape}")

        logger.info("Standalone dataset verification completed successfully.")
    except Exception as error:
        print(f"Verification failed: {error}")
