"""BERT tokenizer utilities module for the AGNews Text Classification project.

This module provides generic, reusable functions to load, save, serialize, and run tokenization
using HuggingFace Transformers for BERT and other transformer-based architectures.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.config import BERT_MAX_LENGTH, BERT_MODEL_NAME
from src.utils.exception import CustomException
from src.utils.file_io import load_pickle, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_bert_tokenizer(
    model_name: str = BERT_MODEL_NAME, **kwargs: Any
) -> PreTrainedTokenizerBase:
    """Loads a pretrained HuggingFace tokenizer.

    Args:
        model_name (str): HuggingFace pretrained model identifier.
            Defaults to BERT_MODEL_NAME.
        **kwargs (Any): Additional options passed to AutoTokenizer.from_pretrained.

    Returns:
        PreTrainedTokenizerBase: Pretrained HuggingFace tokenizer instance.

    Raises:
        CustomException: If tokenizer loading fails.
    """
    try:
        logger.info(f"Loading pretrained tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        logger.info("Pretrained tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        raise CustomException(e, sys) from e


def tokenize_text_data(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = BERT_MAX_LENGTH,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Tokenizes a collection of texts into model inputs (IDs, masks, token types).

    Args:
        tokenizer (PreTrainedTokenizerBase): Loaded HuggingFace tokenizer.
        texts (list[str]): List of input text documents.
        max_length (int): Maximum token sequence length. Defaults to BERT_MAX_LENGTH.
        **kwargs (Any): Additional options passed to the tokenizer.

    Returns:
        dict[str, np.ndarray]: Dictionary containing:
            - "input_ids": Token ID sequence array of shape (num_samples, max_length).
            - "attention_mask": Mask array indicating real vs padding tokens.
            - "token_type_ids": Segment identification array.

    Raises:
        CustomException: If tokenization fails.
    """
    try:
        logger.info(f"Tokenizing {len(texts)} text sequences...")
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
            **kwargs,
        )
        # Convert BatchEncoding to a standard dict of numpy arrays
        result = {key: val for key, val in tokenized.items()}
        logger.info("Text sequence tokenization completed.")
        return result
    except Exception as e:
        raise CustomException(e, sys) from e


def generate_input_ids(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = BERT_MAX_LENGTH,
    **kwargs: Any,
) -> np.ndarray:
    """Generates only the input IDs for the given texts.

    Args:
        tokenizer (PreTrainedTokenizerBase): Loaded HuggingFace tokenizer.
        texts (list[str]): List of input text documents.
        max_length (int): Maximum token sequence length. Defaults to BERT_MAX_LENGTH.
        **kwargs (Any): Additional options passed to the tokenizer.

    Returns:
        np.ndarray: Input token IDs array of shape (num_samples, max_length).

    Raises:
        CustomException: If generation of input IDs fails.
    """
    try:
        tokenized = tokenize_text_data(
            tokenizer, texts, max_length=max_length, **kwargs
        )
        return tokenized["input_ids"]
    except Exception as e:
        raise CustomException(e, sys) from e


def generate_attention_masks(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = BERT_MAX_LENGTH,
    **kwargs: Any,
) -> np.ndarray:
    """Generates only the attention masks for the given texts.

    Args:
        tokenizer (PreTrainedTokenizerBase): Loaded HuggingFace tokenizer.
        texts (list[str]): List of input text documents.
        max_length (int): Maximum token sequence length. Defaults to BERT_MAX_LENGTH.
        **kwargs (Any): Additional options passed to the tokenizer.

    Returns:
        np.ndarray: Attention mask array of shape (num_samples, max_length).

    Raises:
        CustomException: If generation of attention masks fails.
    """
    try:
        tokenized = tokenize_text_data(
            tokenizer, texts, max_length=max_length, **kwargs
        )
        return tokenized["attention_mask"]
    except Exception as e:
        raise CustomException(e, sys) from e


def generate_token_type_ids(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = BERT_MAX_LENGTH,
    **kwargs: Any,
) -> np.ndarray:
    """Generates only the token type IDs for the given texts.

    Args:
        tokenizer (PreTrainedTokenizerBase): Loaded HuggingFace tokenizer.
        texts (list[str]): List of input text documents.
        max_length (int): Maximum token sequence length. Defaults to BERT_MAX_LENGTH.
        **kwargs (Any): Additional options passed to the tokenizer.

    Returns:
        np.ndarray: Token type IDs array of shape (num_samples, max_length).

    Raises:
        CustomException: If generation of token type IDs fails.
    """
    try:
        tokenized = tokenize_text_data(
            tokenizer, texts, max_length=max_length, **kwargs
        )
        # Note: Some models (like DistilBERT, RoBERTa) do not output token_type_ids.
        # Fallback to an array of zeros if missing.
        if "token_type_ids" in tokenized:
            return tokenized["token_type_ids"]
        else:
            logger.warning(
                "token_type_ids not found in tokenized output; defaulting to zeros."
            )
            num_samples = len(texts)
            return np.zeros((num_samples, max_length), dtype=np.int32)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_tokenizer(
    tokenizer: PreTrainedTokenizerBase, file_path: Path
) -> None:
    """Saves the tokenizer object to disk using file_io utilities.

    Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer instance to save.
        file_path (Path): Destination file path to save the pickle file.

    Raises:
        CustomException: If saving the tokenizer fails.
    """
    try:
        logger.info(f"Saving tokenizer artifact to: {file_path}")
        save_pickle(path=file_path, data=tokenizer)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_tokenizer(file_path: Path) -> PreTrainedTokenizerBase:
    """Loads a tokenizer object from disk using file_io utilities.

    Args:
        file_path (Path): Path to the saved tokenizer file.

    Returns:
        PreTrainedTokenizerBase: Loaded HuggingFace tokenizer instance.

    Raises:
        CustomException: If loading the tokenizer fails.
    """
    try:
        logger.info(f"Loading tokenizer artifact from: {file_path}")
        tokenizer = load_pickle(path=file_path)
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(
                f"Loaded object is not of type PreTrainedTokenizerBase: {type(tokenizer)}"
            )
        return tokenizer
    except Exception as e:
        raise CustomException(e, sys) from e


def tokenize_single_text_for_inference(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int = BERT_MAX_LENGTH,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Tokenizes a single text string for model inference.

    Formats the output token structures by adding the batch dimension (batch_size=1).

    Args:
        text (str): Input text string.
        tokenizer (PreTrainedTokenizerBase): Loaded HuggingFace tokenizer.
        max_length (int): Maximum sequence length. Defaults to BERT_MAX_LENGTH.
        **kwargs (Any): Additional options passed to the tokenizer.

    Returns:
        dict[str, np.ndarray]: Dictionary containing:
            - "input_ids": Array of shape (1, max_length).
            - "attention_mask": Array of shape (1, max_length).
            - "token_type_ids": Array of shape (1, max_length).

    Raises:
        CustomException: If converting text for inference fails.
    """
    try:
        return tokenize_text_data(
            tokenizer=tokenizer,
            texts=[text],
            max_length=max_length,
            **kwargs,
        )
    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info("Starting standalone tokenizer verification...")

        # Setup paths
        test_dir = Path(__file__).resolve().parent / "test_output"
        test_dir.mkdir(exist_ok=True)
        tokenizer_path = test_dir / "test_bert_tokenizer.pkl"

        # Initialize
        tok = load_bert_tokenizer()

        # Tokenize single text
        mock_text = "Checking the HuggingFace BERT Tokenizer utility functions."
        tokenized_dict = tokenize_single_text_for_inference(tok, mock_text)
        print(f"Mock Input: '{mock_text}'")
        for key, arr in tokenized_dict.items():
            print(f"{key} (shape: {arr.shape}): {arr[0][:10]}...")

        # Save & Load roundtrip
        save_tokenizer(tok, tokenizer_path)
        loaded_tok = load_tokenizer(tokenizer_path)
        print(f"Successfully serialized and deserialized: {loaded_tok.__class__.__name__}")

        # Clean up
        if tokenizer_path.exists():
            tokenizer_path.unlink()
        if test_dir.exists():
            test_dir.rmdir()
        logger.info("Verification completed successfully.")
    except Exception as error:
        print(f"Verification failed: {error}")
