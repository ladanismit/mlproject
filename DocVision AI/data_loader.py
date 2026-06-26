"""DocVision-AI Data Loader Module.

This module handles the discovery, validation, and metadata generation for
the document dataset. It recursively scans the raw data directory for supported
images, validates that they are readable using OpenCV, and structures the metadata
into a clean pandas DataFrame.

This module adheres to the Single Responsibility Principle and is completely
independent of modeling frameworks (PyTorch/TensorFlow) and preprocessing steps.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Set
import cv2
import pandas as pd

# Add project root to sys.path to enable running the script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import configurations
from src.config import CLASSES, CLASS_TO_IDX, RAW_DATA_DIR, SUPPORTED_EXTENSIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_valid_image(image_path: Path) -> bool:
    """Validates whether an image file is readable and not corrupted.

    Uses OpenCV (cv2) to read the image file and check that it is non-empty.

    Args:
        image_path (Path): Absolute or relative path to the image file.

    Returns:
        bool: True if the image is readable and valid, False otherwise.
    """
    try:
        # Resolve path to handle special characters or symbolic links
        resolved_path = str(image_path.resolve())
        # cv2.imread returns None if the image is corrupted or cannot be read
        img = cv2.imread(resolved_path)
        if img is None:
            logger.warning(
                f"Image file is unreadable or corrupted: {image_path.name} "
                f"at {image_path.parent}"
            )
            return False

        # Verify that dimensions are greater than zero
        if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            logger.warning(
                f"Image has invalid/empty dimensions: {image_path.name} "
                f"at {image_path.parent}"
            )
            return False

        return True
    except Exception as e:
        logger.warning(
            f"Exception occurred while validating image {image_path.name}: {e}"
        )
        return False


def create_metadata(
    raw_dir: Path = RAW_DATA_DIR,
    document_classes: List[str] = CLASSES,
    class_mappings: Dict[str, int] = CLASS_TO_IDX,
    allowed_extensions: Set[str] = SUPPORTED_EXTENSIONS,
) -> pd.DataFrame:
    """Discovers all supported document images and builds a metadata DataFrame.

    Scans the raw data folder recursively for files matching the allowed
    extensions, filters out any corrupted/unreadable files, and maps their
    labels to integers.

    Args:
        raw_dir (Path): Base directory containing class-specific folders.
        document_classes (List[str]): List of active document class names.
        class_mappings (Dict[str, int]): Mapping of class name to integer index.
        allowed_extensions (Set[str]): Supported file extensions.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - `image_path`: Absolute path to the validated image (str)
            - `class_name`: The category/folder name (str)
            - `label`: Integer index mapping for the category (int)
    """
    records = []
    # Convert extensions to lower case for case-insensitive matching
    lower_extensions = {ext.lower() for ext in allowed_extensions}

    logger.info(f"Scanning directory: {raw_dir}")

    for class_name in document_classes:
        class_path = raw_dir / class_name
        if not class_path.exists():
            logger.error(f"Class directory does not exist: {class_path}")
            continue

        # Recursively search for all files under the class directory
        all_files = list(class_path.rglob("*"))
        logger.info(
            f"Found {len(all_files)} files/folders in class folder: '{class_name}'"
        )

        class_count = 0
        skipped_count = 0
        corrupted_count = 0

        for file_path in all_files:
            if not file_path.is_file():
                continue

            # Case-insensitive extension check
            suffix = file_path.suffix.lower()
            if suffix not in lower_extensions:
                skipped_count += 1
                continue

            # Skip checking/processing PDFs directly with cv2.imread since they
            # need an explicit PDF converter. However, check file existence.
            if suffix == ".pdf":
                if file_path.stat().st_size > 0:
                    records.append(
                        {
                            "image_path": str(file_path.resolve()),
                            "class_name": class_name,
                            "label": class_mappings[class_name],
                        }
                    )
                    class_count += 1
                else:
                    logger.warning(f"Empty PDF file encountered: {file_path}")
                    corrupted_count += 1
            else:
                # Standard image validation
                if is_valid_image(file_path):
                    records.append(
                        {
                            "image_path": str(file_path.resolve()),
                            "class_name": class_name,
                            "label": class_mappings[class_name],
                        }
                    )
                    class_count += 1
                else:
                    corrupted_count += 1

        logger.info(
            f"Class '{class_name}' Summary: {class_count} valid, "
            f"{skipped_count} unsupported files ignored, {corrupted_count} corrupted/empty skipped."
        )

    # Build and return the DataFrame
    df = pd.DataFrame(records, columns=["image_path", "class_name", "label"])
    return df


def print_dataset_statistics(df: pd.DataFrame) -> None:
    """Prints diagnostic statistics and class distribution of the loaded dataset.

    Args:
        df (pd.DataFrame): Metadata DataFrame of the dataset.
    """
    print("=" * 60)
    print("Dataset Metadata Statistics")
    print("=" * 60)
    print(f"Total Valid Samples: {len(df)}")

    if df.empty:
        print("Dataset is empty. No valid samples found.")
        print("=" * 60)
        return

    # Count of images per class
    class_counts = df["class_name"].value_counts()
    percentage = df["class_name"].value_counts(normalize=True) * 100

    print("\nClass Distribution:")
    for cls in class_counts.index:
        count = class_counts[cls]
        pct = percentage[cls]
        print(f"  - {cls:<18} : {count:<5} ({pct:.2f}%)")

    print("\nData Types:")
    print(df.dtypes)
    print("=" * 60)


def load_dataset(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """High-level Orchestrator function to discover and load metadata.

    Args:
        raw_dir (Path): Base directory containing class-specific folders.

    Returns:
        pd.DataFrame: Pandas DataFrame with validated image paths and labels.
    """
    df = create_metadata(raw_dir=raw_dir)
    print_dataset_statistics(df)
    return df


if __name__ == "__main__":
    # Test dataset loading and display metadata summary
    print("Initializing DocVision-AI Data Loader Verification...")
    dataset_df = load_dataset()
    if not dataset_df.empty:
        print("\nFirst 5 Rows of Metadata:")
        print(dataset_df.head())
