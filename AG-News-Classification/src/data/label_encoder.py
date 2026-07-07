"""Label encoder module for the AGNews Text Classification project.

This module provides generic, reusable classes and functions to convert
classification labels to 0-indexed integers and back. It integrates with
the project's configuration, logging, and error handling framework.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.config import CLASS_LABELS
from src.utils.exception import CustomException
from src.utils.file_io import load_pickle, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CustomLabelEncoder:
    """A generic label encoder to convert classification labels to 0-indexed integers.

    Supports decoding them back to their original values or mapped class names.
    """

    def __init__(self, class_labels_map: Optional[dict[Any, str]] = None) -> None:
        """Initializes the label encoder.

        Args:
            class_labels_map (Optional[dict[Any, str]]): Mapping from raw label
                to class name string. Defaults to None.
        """
        self.class_to_idx: dict[Any, int] = {}
        self.idx_to_class: dict[int, Any] = {}
        self.class_labels_map: dict[Any, str] = class_labels_map or {}

    def fit(self, labels: list[Any]) -> "CustomLabelEncoder":
        """Fits the encoder on a list of raw labels.

        Args:
            labels (list[Any]): A list of labels to map.

        Returns:
            CustomLabelEncoder: The fitted encoder instance.
        """
        unique_labels = sorted(list(set(labels)))
        self.class_to_idx = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        self.idx_to_class = {
            idx: label for label, idx in self.class_to_idx.items()
        }
        logger.info(
            f"Fitted CustomLabelEncoder on {len(unique_labels)} unique classes: "
            f"{unique_labels}"
        )
        return self

    def encode(self, labels: list[Any]) -> list[int]:
        """Encodes a list of raw labels to 0-indexed integers.

        Args:
            labels (list[Any]): List of raw labels.

        Returns:
            list[int]: Encoded 0-indexed labels.
        """
        encoded_labels = []
        for label in labels:
            if label not in self.class_to_idx:
                raise ValueError(
                    f"Label '{label}' is not fitted. Known labels: "
                    f"{list(self.class_to_idx.keys())}"
                )
            encoded_labels.append(self.class_to_idx[label])
        return encoded_labels

    def decode(self, indices: list[int], return_names: bool = True) -> list[Any]:
        """Decodes 0-indexed integer indices back to original labels or class names.

        Args:
            indices (list[int]): List of 0-indexed integers.
            return_names (bool): If True, returns class name strings (if mapped).
                If False, returns raw label values. Defaults to True.

        Returns:
            list[Any]: Decoded labels or class names.
        """
        decoded_labels = []
        for idx in indices:
            if idx not in self.idx_to_class:
                raise ValueError(
                    f"Index {idx} is unknown. Known indices: "
                    f"{list(self.idx_to_class.keys())}"
                )
            raw_label = self.idx_to_class[idx]
            if return_names:
                decoded_labels.append(
                    self.class_labels_map.get(
                        raw_label, str(raw_label)
                    )
                )
            else:
                decoded_labels.append(raw_label)
        return decoded_labels


def create_and_initialize_encoder(
    labels: Optional[list[Any]] = None,
    class_labels_map: Optional[dict[Any, str]] = CLASS_LABELS,
) -> CustomLabelEncoder:
    """Creates and initializes a CustomLabelEncoder.

    If labels are provided, fits the encoder. Otherwise, fits using the keys of
    class_labels_map if present.

    Args:
        labels (Optional[list[Any]]): A list of raw labels to fit the encoder.
        class_labels_map (Optional[dict[Any, str]]): Mapping from raw label to
            friendly class names. Defaults to CLASS_LABELS.

    Returns:
        CustomLabelEncoder: Initialized label encoder.

    Raises:
        CustomException: If initialization fails.
    """
    try:
        encoder = CustomLabelEncoder(class_labels_map=class_labels_map)
        if labels is not None:
            encoder.fit(labels)
        elif class_labels_map:
            # Fit using keys of class_labels_map
            encoder.fit(list(class_labels_map.keys()))
        else:
            logger.warning(
                "CustomLabelEncoder initialized without labels or mapping."
            )
        return encoder
    except Exception as e:
        raise CustomException(e, sys) from e


def validate_labels(encoder: CustomLabelEncoder, labels: list[Any]) -> None:
    """Validates that all labels exist in the fitted encoder.

    Args:
        encoder (CustomLabelEncoder): Fitted label encoder.
        labels (list[Any]): List of labels to validate.

    Raises:
        CustomException: If validation fails or any label is unknown.
    """
    try:
        unknown_labels = [
            label for label in labels if label not in encoder.class_to_idx
        ]
        if unknown_labels:
            error_msg = (
                f"Validation failed: Unknown labels {unknown_labels} found. "
                f"Fitted labels: {list(encoder.class_to_idx.keys())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Successfully validated {len(labels)} labels.")
    except Exception as e:
        raise CustomException(e, sys) from e


def encode_labels(encoder: CustomLabelEncoder, labels: list[Any]) -> list[int]:
    """Encodes a list of raw labels into 0-indexed integers.

    Args:
        encoder (CustomLabelEncoder): Fitted label encoder.
        labels (list[Any]): List of raw labels to encode.

    Returns:
        list[int]: Encoded 0-indexed integers.

    Raises:
        CustomException: If encoding fails.
    """
    try:
        logger.info(f"Encoding {len(labels)} labels...")
        return encoder.encode(labels)
    except Exception as e:
        raise CustomException(e, sys) from e


def decode_labels(
    encoder: CustomLabelEncoder,
    indices: list[int],
    return_names: bool = True,
) -> list[Any]:
    """Decodes 0-indexed integer indices back to original labels or class names.

    Args:
        encoder (CustomLabelEncoder): Fitted label encoder.
        indices (list[int]): List of 0-indexed integer indices.
        return_names (bool): If True, returns class name strings (if mapped).
            If False, returns raw label values. Defaults to True.

    Returns:
        list[Any]: List of original label values or class name strings.

    Raises:
        CustomException: If decoding fails.
    """
    try:
        logger.info(
            f"Decoding {len(indices)} indices (return_names={return_names})..."
        )
        return encoder.decode(indices, return_names=return_names)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_label_encoder(encoder: CustomLabelEncoder, file_path: Path) -> None:
    """Saves the label encoder to disk.

    Args:
        encoder (CustomLabelEncoder): CustomLabelEncoder instance to save.
        file_path (Path): Destination file path to save the pickle file.

    Raises:
        CustomException: If saving fails.
    """
    try:
        logger.info(f"Saving label encoder to: {file_path}")
        save_pickle(path=file_path, data=encoder)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_label_encoder(file_path: Path) -> CustomLabelEncoder:
    """Loads a CustomLabelEncoder from disk.

    Args:
        file_path (Path): Path to the saved label encoder file.

    Returns:
        CustomLabelEncoder: Loaded label encoder instance.

    Raises:
        CustomException: If loading fails.
    """
    try:
        logger.info(f"Loading label encoder from: {file_path}")
        encoder = load_pickle(path=file_path)
        if not isinstance(encoder, CustomLabelEncoder):
            raise TypeError(
                f"Loaded object is not of type CustomLabelEncoder: "
                f"{type(encoder)}"
            )
        return encoder
    except Exception as e:
        raise CustomException(e, sys) from e


def get_class_names_and_indices(
    encoder: CustomLabelEncoder,
) -> tuple[list[str], list[int]]:
    """Retrieves class names and their corresponding 0-indexed indices.

    Returns:
        tuple[list[str], list[int]]: A tuple containing a list of class names
            and a list of indices.

    Raises:
        CustomException: If retrieving names/indices fails.
    """
    try:
        indices = sorted(list(encoder.idx_to_class.keys()))
        class_names = []
        for idx in indices:
            raw_label = encoder.idx_to_class[idx]
            name = encoder.class_labels_map.get(raw_label, str(raw_label))
            class_names.append(name)
        return class_names, indices
    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info("Starting standalone label encoder verification...")

        # Setup paths
        test_dir = Path(__file__).resolve().parent / "test_output"
        test_dir.mkdir(exist_ok=True)
        encoder_path = test_dir / "test_label_encoder.pkl"

        # Mock inputs matching AG News labels
        raw_labels = [3, 4, 2, 1, 3, 2, 4, 1]

        # 1. Create and initialize encoder
        enc = create_and_initialize_encoder(labels=raw_labels)

        # 2. Validate labels
        validate_labels(enc, [1, 2, 3, 4])

        # 3. Encode labels
        encoded = encode_labels(enc, raw_labels)
        print(f"Raw labels:     {raw_labels}")
        print(f"Encoded labels: {encoded}")

        # 4. Save label encoder
        save_label_encoder(enc, encoder_path)

        # 5. Load label encoder
        loaded_enc = load_label_encoder(encoder_path)

        # 6. Return class names and indices
        names, idxs = get_class_names_and_indices(loaded_enc)
        print(f"Class Names:   {names}")
        print(f"Class Indices: {idxs}")

        # 7. Decode predictions back to names
        decoded_names = decode_labels(loaded_enc, encoded, return_names=True)
        decoded_raw = decode_labels(loaded_enc, encoded, return_names=False)
        print(f"Decoded (names): {decoded_names}")
        print(f"Decoded (raw):   {decoded_raw}")

        # Clean up
        if encoder_path.exists():
            encoder_path.unlink()
        if test_dir.exists():
            test_dir.rmdir()

    except Exception as error:
        print(f"Verification failed: {error}")
