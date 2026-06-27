"""DocVision-AI TensorFlow Dataset Module.

This module is responsible for taking the metadata DataFrame from the data loader,
splitting it into train, validation, and test partitions, and wrapping them
into optimized `tf.data.Dataset` pipelines. It handles mapping OpenCV-based
preprocessing dynamically using TensorFlow's python execution capabilities
and ensures shape propagation for downstream models.

This module follows the Single Responsibility Principle, focusing solely on
dataset partitioning and tf.data pipeline creation.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add project root to sys.path to enable running the script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import configuration and preprocessing
from src.config import (
    BATCH_SIZE,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
)
from src.data_loader import load_dataset
from src.preprocessing import preprocess_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_metadata(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the metadata DataFrame into Train, Validation, and Test sets.

    Utilizes stratified splitting to ensure consistent class distributions across
    each set.

    Args:
        df (pd.DataFrame): Input metadata DataFrame containing 'image_path' and 'label'.
        train_size (float): Proportion of the dataset for training. Defaults to 0.70.
        val_size (float): Proportion of the dataset for validation. Defaults to 0.15.
        test_size (float): Proportion of the dataset for testing. Defaults to 0.15.
        random_seed (int): Random seed for reproducibility. Defaults to RANDOM_SEED.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, Val, and Test DataFrames.
    """
    if not (0.99 <= (train_size + val_size + test_size) <= 1.01):
        raise ValueError("Splits must sum up to approximately 1.0")

    logger.info(
        f"Splitting metadata. Ratios - Train: {train_size:.2f}, "
        f"Val: {val_size:.2f}, Test: {test_size:.2f}"
    )

    # First split: Separate training set from the rest (Validation + Test)
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_seed,
        stratify=df["label"] if "label" in df.columns else None,
    )

    # Second split: Divide the remaining into Validation and Test
    # val_ratio_in_temp determines proportion of temp_df that becomes val_df
    val_ratio_in_temp = val_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=random_seed,
        stratify=temp_df["label"] if "label" in temp_df.columns else None,
    )

    logger.info(
        f"Splits generated successfully. "
        f"Train samples: {len(train_df)}, "
        f"Val samples: {len(val_df)}, "
        f"Test samples: {len(test_df)}"
    )

    return train_df, val_df, test_df


def _preprocess_tf_wrapper(image_path_tensor: tf.Tensor) -> tf.Tensor:
    """Helper wrapper function to load and preprocess an image.

    This function runs inside tf.py_function, decoding the string tensor and
    running the OpenCV/NumPy preprocessing pipeline.

    Args:
        image_path_tensor (tf.Tensor): A string tensor containing the image path.

    Returns:
        tf.Tensor: Processed float32 image tensor.
    """
    image_path = image_path_tensor.numpy().decode("utf-8")
    processed_image = preprocess_image(image_path)
    return tf.convert_to_tensor(processed_image, dtype=tf.float32)


def _map_function(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Map function to wrap the custom python preprocessing into tf.data flow.

    Args:
        image_path (tf.Tensor): Path string tensor.
        label (tf.Tensor): Label integer tensor.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Preprocessed image tensor and its label.
    """
    # Use tf.py_function to run numpy/OpenCV code inside the TensorFlow graph
    processed_image = tf.py_function(
        func=_preprocess_tf_wrapper,
        inp=[image_path],
        Tout=tf.float32,
    )

    # Set explicit shape since tf.py_function erases static shape details
    processed_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    return processed_image, label


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    buffer_size: int = 1000,
    random_seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset pipeline from a metadata DataFrame.

    Converts paths and labels to tensors, maps preprocessing, batches, and
    prefetches the dataset for training or evaluation.

    Args:
        df (pd.DataFrame): DataFrame containing 'image_path' and 'label' columns.
        batch_size (int): Target batch size. Defaults to BATCH_SIZE.
        shuffle (bool): If True, shuffles the dataset. Defaults to False.
        buffer_size (int): Shuffle buffer size. Defaults to 1000.
        random_seed (int): Seed for shuffling. Defaults to RANDOM_SEED.

    Returns:
        tf.data.Dataset: Optimized TensorFlow dataset.
    """
    # 1. Create dataset of slices
    paths = df["image_path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    # 2. Shuffle before mapping/batching (recommended for performance)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=random_seed,
            reshuffle_each_iteration=True,
        )

    # 3. Map preprocessing using tf.data.AUTOTUNE for parallel execution
    dataset = dataset.map(_map_function, num_parallel_calls=tf.data.AUTOTUNE)

    # 4. Batch the dataset
    dataset = dataset.batch(batch_size)

    # 5. Prefetch for optimal GPU/CPU pipelining
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_datasets(
    batch_size: int = BATCH_SIZE,
    random_seed: int = RANDOM_SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Generates the Train, Validation, and Test tf.data.Dataset pipelines.

    Orchestrates loading metadata, generating stratified splits, and assembling
    optimized TF pipelines for train, val, and test.

    Args:
        batch_size (int): Batch size. Defaults to BATCH_SIZE.
        random_seed (int): Random seed. Defaults to RANDOM_SEED.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Train, Val, and Test datasets.
    """
    # Load dataset metadata
    df = load_dataset()

    # Split metadata
    train_df, val_df, test_df = split_metadata(
        df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_seed=random_seed,
    )

    # Build tf.data datasets
    logger.info("Building Train, Validation, and Test tf.data pipelines...")
    train_dataset = create_tf_dataset(train_df, batch_size=batch_size, shuffle=True, random_seed=random_seed)
    val_dataset = create_tf_dataset(val_df, batch_size=batch_size, shuffle=False)
    test_dataset = create_tf_dataset(test_df, batch_size=batch_size, shuffle=False)

    logger.info("Dataset pipelines created successfully.")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    print("Initializing DocVision-AI tf.data.Dataset Verification...")
    try:
        # Load and build datasets
        train_ds, val_ds, test_ds = get_datasets()

        # Pull a single batch from the training set to verify shapes
        print("\nFetching a single batch from the Training Dataset...")
        for images_batch, labels_batch in train_ds.take(1):
            print("\nVerification Success Summary:")
            print(f"  - Image Batch Shape: {images_batch.shape}")
            print(f"  - Label Batch Shape: {labels_batch.shape}")
            print(f"  - Image Data Type:   {images_batch.dtype}")
            print(f"  - Label Data Type:   {labels_batch.dtype}")

            # Double check shape is fully defined
            assert images_batch.shape[1:] == (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), \
                "Error: Tensor shape mismatch!"
            print("  - Shape verification: SUCCESS")

    except Exception as err:
        print(f"Verification test failed: {err}")
