"""DocVision-AI Preprocessing Module.

This module provides utility functions for loading and preprocessing document
images prior to deep learning model ingestion. It handles reading files, BGR-to-RGB
conversion, grayscale adjustments, resizing, and pixel value normalization.

This module is designed in compliance with the Single Responsibility Principle,
independent of any specific deep learning framework (PyTorch/TensorFlow) or
augmentation packages.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np

# Add project root to sys.path to enable running the script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import configuration constants
from src.config import IMAGE_CHANNELS, IMAGE_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_image(image_path: Union[str, Path]) -> np.ndarray:
    """Reads an image file from the filesystem.

    Args:
        image_path (Union[str, Path]): Path to the target image file.

    Returns:
        np.ndarray: Loaded image array in BGR format.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be read or is empty.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    # cv2.imread expects a string path
    img = cv2.imread(str(path.resolve()))
    if img is None:
        raise ValueError(
            f"Failed to read image. The file might be corrupted or in an unsupported format: {path}"
        )

    return img


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image array (default in OpenCV) to RGB format.

    Args:
        image (np.ndarray): Image array in BGR format.

    Returns:
        np.ndarray: Converted image array in RGB format.
    """
    # If the image is 2D (grayscale), no color channel conversion is required.
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def handle_grayscale(image: np.ndarray, target_channels: int = IMAGE_CHANNELS) -> np.ndarray:
    """Ensures the image matches the target channel count.

    Converts single-channel grayscale to multi-channel RGB, or multi-channel BGR/RGB to
    single-channel grayscale as requested.

    Args:
        image (np.ndarray): Input image array.
        target_channels (int): Target channel count (typically 1 or 3).

    Returns:
        np.ndarray: Image array with the specified number of channels.
    """
    has_channels = len(image.shape) == 3
    num_channels = image.shape[2] if has_channels else 1

    if target_channels == 3 and num_channels == 1:
        # Convert Grayscale to RGB
        if has_channels:
            image = np.squeeze(image, axis=-1)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif target_channels == 1 and num_channels == 3:
        # Convert RGB to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, axis=-1)

    elif target_channels == 1 and not has_channels:
        # Grayscale, ensure it has a channel dimension (H, W, 1)
        return np.expand_dims(image, axis=-1)

    return image


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    pad_color: int = 255,
) -> np.ndarray:
    """Resizes an image preserving its aspect ratio and pads it with a background color.

    Prevents squashing or stretching document layout features. Uses white padding (255)
    by default to match standard document background colors.

    Args:
        image (np.ndarray): Input image array.
        target_size (Tuple[int, int]): A tuple (target_height, target_width) of desired dimensions.
        pad_color (int): Background padding color (0-255). Defaults to 255 (white).

    Returns:
        np.ndarray: Resized and padded image.
    """
    target_height, target_width = target_size
    h, w = image.shape[:2]

    # Calculate scaling factor
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # Resize retaining aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Initialize padded canvas
    if len(image.shape) == 3:
        channels = image.shape[2]
        padded = np.full((target_height, target_width, channels), pad_color, dtype=image.dtype)
    else:
        padded = np.full((target_height, target_width), pad_color, dtype=image.dtype)

    # Offset to center the image
    dy = (target_height - new_h) // 2
    dx = (target_width - new_w) // 2

    # Paste resized image into the canvas
    padded[dy:dy + new_h, dx:dx + new_w] = resized

    return padded


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalizes the image pixels to the range [0.0, 1.0] and converts to float32.

    Args:
        image (np.ndarray): Input image array (usually uint8).

    Returns:
        np.ndarray: Normalized float32 image array.
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    target_channels: int = IMAGE_CHANNELS,
) -> np.ndarray:
    """Performs the complete image preprocessing pipeline.

    Loads the image, converts to RGB, manages grayscale/channel adjustments,
    resizes to target dimensions, and normalizes pixel values to [0.0, 1.0].

    Args:
        image_path (Union[str, Path]): Path to the target image file.
        target_size (Tuple[int, int]): The target (height, width) tuple.
        target_channels (int): Target channel count.

    Returns:
        np.ndarray: Preprocessed float32 NumPy array ready for model consumption.
    """
    try:
        # 1. Read the image
        img = read_image(image_path)

        # 2. Convert from BGR (OpenCV default) to RGB
        img = convert_bgr_to_rgb(img)

        # 3. Handle channel count (e.g., Grayscale to RGB or RGB to Grayscale)
        img = handle_grayscale(img, target_channels=target_channels)

        # 4. Resize to configuration size
        img = resize_image(img, target_size=target_size)

        # 5. Normalize pixel values to [0.0, 1.0]
        preprocessed = normalize_image(img)

        return preprocessed
    except Exception as e:
        logger.error(f"Error during preprocessing for file {image_path}: {e}")
        raise


if __name__ == "__main__":
    # Self-test block: Preprocess an example file from the dataset if available
    from src.data_loader import load_dataset

    print("Initializing DocVision-AI Preprocessing Verification...")
    try:
        # Load dataset metadata to find an existing image path
        df = load_dataset()
        if not df.empty:
            sample_path = df.iloc[0]["image_path"]
            print(f"\nProcessing sample image: {sample_path}")

            # Preprocess the sample
            processed_arr = preprocess_image(sample_path)

            print("\nPreprocessing Success Summary:")
            print(f"  - Output Shape:     {processed_arr.shape}")
            print(f"  - Output Data Type: {processed_arr.dtype}")
            print(f"  - Pixel Value Range: [{processed_arr.min():.4f}, {processed_arr.max():.4f}]")
        else:
            print("No valid dataset images found. Skipping live test.")
    except Exception as err:
        print(f"Verification test failed: {err}")
