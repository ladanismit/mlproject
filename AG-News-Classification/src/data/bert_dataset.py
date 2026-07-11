"""BERT dataset pipeline module for the AGNews Text Classification project.

This module provides generic, reusable, and type-hinted functions to create, split,
shuffle, batch, cache, and prefetch TensorFlow datasets specifically formatted for
transformer-based architectures like BERT, using the tf.data API.
"""

import sys
from pathlib import Path
from typing import Any

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


def create_bert_tf_dataset(
    input_ids: np.ndarray,
    attention_masks: np.ndarray,
    token_type_ids: np.ndarray,
    labels: np.ndarray | None = None,
) -> tf.data.Dataset:
    """Creates a basic tf.data.Dataset from BERT features and optional labels.

    Args:
        input_ids (np.ndarray): Array of input token IDs.
        attention_masks (np.ndarray): Array of attention masks.
        token_type_ids (np.ndarray): Array of token type IDs.
        labels (np.ndarray | None): Target labels. Defaults to None.

    Returns:
        tf.data.Dataset: The constructed TensorFlow dataset yielding (features_dict, labels)
            or just features_dict if labels is None.

    Raises:
        CustomException: If dataset creation fails.
    """
    try:
        logger.info(
            f"Creating tf.data.Dataset. Input shapes - "
            f"input_ids: {input_ids.shape}, "
            f"attention_masks: {attention_masks.shape}, "
            f"token_type_ids: {token_type_ids.shape}"
        )

        # Validate matching length of feature components
        if not (len(input_ids) == len(attention_masks) == len(token_type_ids)):
            raise ValueError(
                "input_ids, attention_masks, and token_type_ids must have the same length."
            )

        features = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids,
        }

        if labels is not None:
            logger.info(f"Adding target labels with shape: {labels.shape}")
            if len(labels) != len(input_ids):
                raise ValueError("Features and labels must have the same length.")
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(features)

        return dataset
    except Exception as e:
        raise CustomException(e, sys) from e


def split_bert_data(
    input_ids: np.ndarray,
    attention_masks: np.ndarray,
    token_type_ids: np.ndarray,
    labels: np.ndarray,
    validation_split: float = VALIDATION_SPLIT,
    random_seed: int = RANDOM_SEED,
) -> tuple[
    np.ndarray, np.ndarray,  # train_input_ids, val_input_ids
    np.ndarray, np.ndarray,  # train_attention_masks, val_attention_masks
    np.ndarray, np.ndarray,  # train_token_type_ids, val_token_type_ids
    np.ndarray, np.ndarray,  # train_labels, val_labels
]:
    """Splits BERT feature arrays and label arrays into training and validation sets.

    Args:
        input_ids (np.ndarray): Complete training input IDs.
        attention_masks (np.ndarray): Complete training attention masks.
        token_type_ids (np.ndarray): Complete training token type IDs.
        labels (np.ndarray): Complete training labels.
        validation_split (float): Fraction of data reserved for validation.
            Defaults to VALIDATION_SPLIT.
        random_seed (int): Random seed for deterministic shuffling.
            Defaults to RANDOM_SEED.

    Returns:
        tuple containing:
            - train_input_ids, val_input_ids
            - train_attention_masks, val_attention_masks
            - train_token_type_ids, val_token_type_ids
            - train_labels, val_labels

    Raises:
        CustomException: If splitting fails.
    """
    try:
        logger.info(
            f"Splitting BERT arrays with validation_split={validation_split} "
            f"and random_seed={random_seed}"
        )

        if not 0.0 <= validation_split < 1.0:
            raise ValueError("Validation split must be in range [0.0, 1.0).")

        num_samples = len(input_ids)
        if not (len(attention_masks) == len(token_type_ids) == len(labels) == num_samples):
            raise ValueError("All feature components and labels must have the same length.")

        if validation_split == 0.0:
            logger.info("Validation split is 0.0. Skipping splitting.")
            empty_shape = (0,) + input_ids.shape[1:]
            return (
                input_ids, np.empty(empty_shape, dtype=input_ids.dtype),
                attention_masks, np.empty(empty_shape, dtype=attention_masks.dtype),
                token_type_ids, np.empty(empty_shape, dtype=token_type_ids.dtype),
                labels, np.empty((0,), dtype=labels.dtype)
            )

        # Shuffle indices deterministically
        rng = np.random.default_rng(seed=random_seed)
        indices = rng.permutation(num_samples)

        val_size = int(num_samples * validation_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Split features
        train_input_ids, val_input_ids = input_ids[train_indices], input_ids[val_indices]
        train_attention_masks, val_attention_masks = attention_masks[train_indices], attention_masks[val_indices]
        train_token_type_ids, val_token_type_ids = token_type_ids[train_indices], token_type_ids[val_indices]
        train_labels, val_labels = labels[train_indices], labels[val_indices]

        logger.info(
            f"Split complete. Training: {len(train_input_ids)} samples, "
            f"Validation: {len(val_input_ids)} samples."
        )
        return (
            train_input_ids, val_input_ids,
            train_attention_masks, val_attention_masks,
            train_token_type_ids, val_token_type_ids,
            train_labels, val_labels
        )
    except Exception as e:
        raise CustomException(e, sys) from e


def optimize_bert_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    buffer_size: int | None = None,
    cache_filepath: str | None = None,
    random_seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """Applies shuffling, batching, caching, and prefetching optimizations.

    Args:
        dataset (tf.data.Dataset): The source tf.data.Dataset.
        batch_size (int): Batch size. Defaults to BATCH_SIZE.
        shuffle (bool): Whether to shuffle the dataset. Defaults to False.
        buffer_size (int | None): Size of shuffle buffer. If None, set to dataset size.
        cache_filepath (str | None): File path to write cache data. If empty string "",
            caches in memory. If None, skips caching. Defaults to None.
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
                f"Applying shuffle with buffer_size={buffer_size} and seed={random_seed}"
            )
            opt_ds = opt_ds.shuffle(
                buffer_size=buffer_size,
                seed=random_seed,
                reshuffle_each_iteration=True,
            )

        # 2. Batch
        logger.debug(f"Applying batching with batch_size={batch_size}")
        opt_ds = opt_ds.batch(batch_size, drop_remainder=False)

        # 3. Cache
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


def create_bert_model_datasets(
    train_input_ids: np.ndarray,
    train_attention_masks: np.ndarray,
    train_token_type_ids: np.ndarray,
    train_labels: np.ndarray,
    test_input_ids: np.ndarray,
    test_attention_masks: np.ndarray,
    test_token_type_ids: np.ndarray,
    test_labels: np.ndarray | None = None,
    validation_split: float = VALIDATION_SPLIT,
    batch_size: int = BATCH_SIZE,
    random_seed: int = RANDOM_SEED,
    cache: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Generates training, validation, and testing tf.data.Dataset objects for BERT.

    Args:
        train_input_ids (np.ndarray): Complete training input IDs.
        train_attention_masks (np.ndarray): Complete training attention masks.
        train_token_type_ids (np.ndarray): Complete training token type IDs.
        train_labels (np.ndarray): Complete training label array.
        test_input_ids (np.ndarray): Complete testing input IDs.
        test_attention_masks (np.ndarray): Complete testing attention masks.
        test_token_type_ids (np.ndarray): Complete testing token type IDs.
        test_labels (np.ndarray | None): Testing labels. Defaults to None.
        validation_split (float): Portion of training data for validation.
            Defaults to VALIDATION_SPLIT.
        batch_size (int): Size of batches. Defaults to BATCH_SIZE.
        random_seed (int): Random seed for shuffling/splitting. Defaults to RANDOM_SEED.
        cache (bool): Whether to cache dataset in memory. Defaults to False.

    Returns:
        tuple containing:
            - train_dataset, validation_dataset, testing_dataset

    Raises:
        CustomException: If dataset generation fails.
    """
    try:
        logger.info("Initializing BERT datasets creation pipeline...")

        # 1. Split training and validation arrays
        (
            t_ids, v_ids,
            t_masks, v_masks,
            t_types, v_types,
            t_lbls, v_lbls
        ) = split_bert_data(
            input_ids=train_input_ids,
            attention_masks=train_attention_masks,
            token_type_ids=train_token_type_ids,
            labels=train_labels,
            validation_split=validation_split,
            random_seed=random_seed,
        )

        # 2. Create raw datasets
        train_raw_ds = create_bert_tf_dataset(t_ids, t_masks, t_types, t_lbls)
        val_raw_ds = create_bert_tf_dataset(v_ids, v_masks, v_types, v_lbls)
        test_raw_ds = create_bert_tf_dataset(test_input_ids, test_attention_masks, test_token_type_ids, test_labels)

        # 3. Optimize datasets
        cache_val = "" if cache else None

        logger.info("Optimizing training dataset (shuffled, batched, prefetched)...")
        train_ds = optimize_bert_dataset(
            train_raw_ds,
            batch_size=batch_size,
            shuffle=True,
            cache_filepath=cache_val,
            random_seed=random_seed,
        )

        logger.info("Optimizing validation dataset (batched, prefetched)...")
        val_ds = optimize_bert_dataset(
            val_raw_ds,
            batch_size=batch_size,
            shuffle=False,
            cache_filepath=cache_val,
        )

        logger.info("Optimizing testing dataset (batched, prefetched)...")
        test_ds = optimize_bert_dataset(
            test_raw_ds,
            batch_size=batch_size,
            shuffle=False,
            cache_filepath=cache_val,
        )

        logger.info("BERT model datasets successfully created and optimized.")
        return train_ds, val_ds, test_ds
    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info("Starting standalone BERT dataset pipeline verification...")

        # 1. Create mock arrays
        mock_train_ids = np.random.randint(low=1, high=1000, size=(100, 10), dtype=np.int32)
        mock_train_masks = np.ones((100, 10), dtype=np.int32)
        mock_train_types = np.zeros((100, 10), dtype=np.int32)
        mock_train_labels = np.random.randint(low=0, high=4, size=(100,), dtype=np.int32)

        mock_test_ids = np.random.randint(low=1, high=1000, size=(20, 10), dtype=np.int32)
        mock_test_masks = np.ones((20, 10), dtype=np.int32)
        mock_test_types = np.zeros((20, 10), dtype=np.int32)
        mock_test_labels = np.random.randint(low=0, high=4, size=(20,), dtype=np.int32)

        # 2. Create datasets
        tr_ds, val_ds, ts_ds = create_bert_model_datasets(
            train_input_ids=mock_train_ids,
            train_attention_masks=mock_train_masks,
            train_token_type_ids=mock_train_types,
            train_labels=mock_train_labels,
            test_input_ids=mock_test_ids,
            test_attention_masks=mock_test_masks,
            test_token_type_ids=mock_test_types,
            test_labels=mock_test_labels,
            validation_split=0.2,
            batch_size=16,
            cache=True,
        )

        # 3. Verify batch sizes and outputs
        for feat_batch, lbl_batch in tr_ds.take(1):
            print(f"\nTrain batch - features keys: {list(feat_batch.keys())}")
            print(f"Train batch - input_ids shape: {feat_batch['input_ids'].shape}")
            print(f"Train batch - labels shape:    {lbl_batch.shape}")

        for feat_batch, lbl_batch in val_ds.take(1):
            print(f"\nVal batch - features keys:   {list(feat_batch.keys())}")
            print(f"Val batch - input_ids shape:   {feat_batch['input_ids'].shape}")
            print(f"Val batch - labels shape:      {lbl_batch.shape}")

        for feat_batch, lbl_batch in ts_ds.take(1):
            print(f"\nTest batch - features keys:  {list(feat_batch.keys())}")
            print(f"Test batch - input_ids shape:  {feat_batch['input_ids'].shape}")
            print(f"Test batch - labels shape:     {lbl_batch.shape}")

        logger.info("Standalone BERT dataset verification completed successfully.")
    except Exception as error:
        print(f"Verification failed: {error}")
