"""DocVision-AI TensorFlow Dataset Module.

This module is responsible for taking the metadata DataFrame from the data loader,
splitting it into train, validation, and test partitions, and wrapping them
into optimized `tf.data.Dataset` pipelines. It handles image loading,
aspect-ratio preserving resizing with white padding, and document-safe data
augmentation using pure TensorFlow graph operations, eliminating tf.py_function overhead.

This module follows the Single Responsibility Principle, focusing solely on
dataset partitioning and tf.data pipeline creation.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Add project root to sys.path to enable running the script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import configuration constants
from src.config import (
    BATCH_SIZE,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
)
from src.data_loader import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# 1. DATA AUGMENTATION PIPELINE (Document-Safe)
# ------------------------------------------------------------------------------
# Define native Keras layers for document-safe geometric augmentations.
# Excludes horizontal and vertical flips to avoid rendering text illegible.
augmentation_pipeline = tf.keras.Sequential([
    layers.RandomRotation(factor=0.02, fill_mode="constant", fill_value=1.0),  # Max ~7 degrees rotation
    layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode="constant", fill_value=1.0),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="constant", fill_value=1.0),
], name="document_augmentation")


def augment_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies document-safe spatial and pixel augmentations in batch.

    Args:
        image (tf.Tensor): A batch of float32 image tensors [Batch, H, W, C].
        label (tf.Tensor): A batch of label tensors [Batch].

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Augmented image batch and corresponding labels.
    """
    # Random geometric transformations
    augmented = augmentation_pipeline(image, training=True)
    # Random pixel-level contrast variation
    augmented = tf.image.random_contrast(augmented, lower=0.8, upper=1.2)
    # Ensure values remain strictly in the range [0.0, 1.0]
    augmented = tf.clip_by_value(augmented, 0.0, 1.0)
    return augmented, label


# ------------------------------------------------------------------------------
# 2. PURE TENSORFLOW IMAGE PREPROCESSING
# ------------------------------------------------------------------------------
def tf_preprocess_image(
    image_path: tf.Tensor,
    target_height: int = IMAGE_HEIGHT,
    target_width: int = IMAGE_WIDTH,
) -> tf.Tensor:
    """Loads, decodes, resizes, pads, and normalizes an image using native TF operations.

    Maintains the original aspect ratio and fills the remaining boundary with white pixels.

    Args:
        image_path (tf.Tensor): Scalar string tensor containing absolute image path.
        target_height (int): Desired image height.
        target_width (int): Desired image width.

    Returns:
        tf.Tensor: Preprocessed float32 image tensor of shape [target_height, target_width, 3].
    """
    # 1. Read file
    image_bytes = tf.io.read_file(image_path)

    # 2. Decode image (convert to 3 channels RGB)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)

    # 3. Convert image to float32 and normalize pixels to [0.0, 1.0]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # 4. Aspect-ratio preserving resize and padding
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    th = tf.cast(target_height, tf.float32)
    tw = tf.cast(target_width, tf.float32)

    # Calculate scale factor to fit inside target size without cropping
    scale = tf.minimum(th / h, tw / w)
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)

    # Resize image using area interpolation (optimal for downsampling)
    resized = tf.image.resize(image, [new_h, new_w], method=tf.image.ResizeMethod.AREA)

    # Calculate padding offsets to center the resized image on the canvas
    dy = (target_height - new_h) // 2
    dx = (target_width - new_w) // 2

    pad_top = dy
    pad_bottom = target_height - new_h - dy
    pad_left = dx
    pad_right = target_width - new_w - dx

    # Pad with constant 1.0 (white background in float32 format)
    padded = tf.pad(
        resized,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        mode="CONSTANT",
        constant_values=1.0,
    )

    # Explicitly define shape for TensorFlow compilation
    padded.set_shape([target_height, target_width, 3])
    return padded


def _map_function(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Map wrapper to hook the pure TensorFlow preprocessing pipeline.

    Args:
        image_path (tf.Tensor): Path string tensor.
        label (tf.Tensor): Label integer tensor.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Preprocessed image tensor and label.
    """
    processed_image = tf_preprocess_image(image_path)
    return processed_image, label


# ------------------------------------------------------------------------------
# 3. SPLIT AND DATASET CREATION PIPELINES
# ------------------------------------------------------------------------------
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


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    augment: bool = False,
    buffer_size: int = 1000,
    random_seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset pipeline from a metadata DataFrame.

    Handles loading, decoding, resizing, batching, caching, prefetching, and
    optional document-safe data augmentation entirely in the TensorFlow graph.

    Args:
        df (pd.DataFrame): DataFrame containing 'image_path' and 'label' columns.
        batch_size (int): Target batch size. Defaults to BATCH_SIZE.
        shuffle (bool): If True, shuffles the dataset. Defaults to False.
        augment (bool): If True, applies data augmentation to batch. Defaults to False.
        buffer_size (int): Shuffle buffer size. Defaults to 1000.
        random_seed (int): Seed for shuffling. Defaults to RANDOM_SEED.

    Returns:
        tf.data.Dataset: Optimized, cached, and prefetched TensorFlow dataset.
    """
    paths = df["image_path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=random_seed,
            reshuffle_each_iteration=True,
        )

    # Map preprocessing function natively
    dataset = dataset.map(_map_function, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Apply data augmentation in batch for vectorization speedup
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch for optimal CPU/GPU concurrency
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_datasets(
    batch_size: int = BATCH_SIZE,
    random_seed: int = RANDOM_SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Generates the Train, Validation, and Test tf.data.Dataset pipelines.

    Args:
        batch_size (int): Batch size. Defaults to BATCH_SIZE.
        random_seed (int): Random seed. Defaults to RANDOM_SEED.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Train, Val, and Test datasets.
    """
    df = load_dataset()
    train_df, val_df, test_df = split_metadata(df, random_seed=random_seed)

    logger.info("Building Train, Validation, and Test tf.data pipelines...")
    train_dataset = create_tf_dataset(
        train_df, batch_size=batch_size, shuffle=True, augment=True, random_seed=random_seed
    )
    val_dataset = create_tf_dataset(val_df, batch_size=batch_size, shuffle=False, augment=False)
    test_dataset = create_tf_dataset(test_df, batch_size=batch_size, shuffle=False, augment=False)

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

            assert images_batch.shape[1:] == (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), \
                "Error: Tensor shape mismatch!"
            print("  - Shape verification: SUCCESS")

    except Exception as err:
        print(f"Verification test failed: {err}")
