"""Tokenizer utilities module for the AGNews Text Classification project.

This module provides reusable functions to create, fit, serialize, and load a
Keras Tokenizer, as well as functions to tokenize and pad input sequences for
both training and inference phases across RNN, LSTM, and BiLSTM models.
"""

import sys
from pathlib import Path
from typing import Any, List, Union

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.config import (
    MAX_SEQUENCE_LENGTH,
    OOV_TOKEN,
    PADDING_TYPE,
    TRUNCATING_TYPE,
    VOCAB_SIZE,
)
from src.utils.exception import CustomException
from src.utils.file_io import load_pickle, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_tokenizer(
    vocab_size: int = VOCAB_SIZE, oov_token: str = OOV_TOKEN
) -> Tokenizer:
    """Creates a standard Keras Tokenizer instance.

    Args:
        vocab_size (int): Max number of words to keep. Defaults to VOCAB_SIZE.
        oov_token (str): Out of vocabulary token. Defaults to OOV_TOKEN.

    Returns:
        Tokenizer: Initialized Keras Tokenizer.

    Raises:
        CustomException: If tokenizer creation fails.
    """
    try:
        logger.info(
            f"Creating Keras Tokenizer with vocab_size={vocab_size}, "
            f"oov_token='{oov_token}'"
        )
        return Tokenizer(num_words=vocab_size, oov_token=oov_token)
    except Exception as e:
        raise CustomException(e, sys) from e


def fit_tokenizer_on_text(
    tokenizer: Tokenizer, texts: Union[List[str], Any]
) -> Tokenizer:
    """Fits the given Tokenizer on training text data.

    Args:
        tokenizer (Tokenizer): Tokenizer instance to fit.
        texts (Union[List[str], Any]): List or series of text documents.

    Returns:
        Tokenizer: The fitted Tokenizer instance.

    Raises:
        CustomException: If fitting the tokenizer fails.
    """
    try:
        logger.info(f"Fitting tokenizer on {len(texts)} text sequences...")
        tokenizer.fit_on_texts(texts)
        vocab_size = len(tokenizer.word_index) + 1
        logger.info(
            f"Tokenizer fitting complete. Calculated vocab size: {vocab_size}"
        )
        return tokenizer
    except Exception as e:
        raise CustomException(e, sys) from e


def convert_texts_to_sequences(
    tokenizer: Tokenizer, texts: Union[List[str], Any]
) -> List[List[int]]:
    """Converts a collection of text documents into integer sequences.

    Args:
        tokenizer (Tokenizer): Fitted Tokenizer instance.
        texts (Union[List[str], Any]): List or series of text documents.

    Returns:
        List[List[int]]: List of token index sequences.

    Raises:
        CustomException: If text to sequence conversion fails.
    """
    try:
        logger.info(f"Converting {len(texts)} texts to sequences...")
        sequences = tokenizer.texts_to_sequences(texts)
        return sequences
    except Exception as e:
        raise CustomException(e, sys) from e


def pad_text_sequences(
    sequences: List[List[int]],
    maxlen: int = MAX_SEQUENCE_LENGTH,
    padding: str = PADDING_TYPE,
    truncating: str = TRUNCATING_TYPE,
) -> np.ndarray:
    """Pads or truncates sequences to a fixed target length.

    Args:
        sequences (List[List[int]]): List of token index sequences.
        maxlen (int): Maximum length of all sequences. Defaults to
            MAX_SEQUENCE_LENGTH.
        padding (str): 'pre' or 'post' padding. Defaults to PADDING_TYPE.
        truncating (str): 'pre' or 'post' truncating. Defaults to
            TRUNCATING_TYPE.

    Returns:
        np.ndarray: Padded/truncated 2D numpy array of shape (num_samples, maxlen).

    Raises:
        CustomException: If sequence padding fails.
    """
    try:
        logger.debug(
            f"Padding sequences: maxlen={maxlen}, padding={padding}, "
            f"truncating={truncating}"
        )
        padded_sequences = pad_sequences(
            sequences, maxlen=maxlen, padding=padding, truncating=truncating
        )
        return padded_sequences
    except Exception as e:
        raise CustomException(e, sys) from e


def save_tokenizer(tokenizer: Tokenizer, file_path: Path) -> None:
    """Saves the Tokenizer object to disk.

    Args:
        tokenizer (Tokenizer): Tokenizer instance to save.
        file_path (Path): Destination file path to save the pickle file.

    Raises:
        CustomException: If saving the tokenizer fails.
    """
    try:
        logger.info(f"Saving tokenizer to: {file_path}")
        save_pickle(path=file_path, data=tokenizer)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_tokenizer(file_path: Path) -> Tokenizer:
    """Loads a Tokenizer object from disk.

    Args:
        file_path (Path): Path to the saved tokenizer file.

    Returns:
        Tokenizer: The loaded Tokenizer instance.

    Raises:
        CustomException: If loading the tokenizer fails.
    """
    try:
        logger.info(f"Loading tokenizer from: {file_path}")
        tokenizer = load_pickle(path=file_path)
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError(
                f"Loaded object is not of type Tokenizer: {type(tokenizer)}"
            )
        return tokenizer
    except Exception as e:
        raise CustomException(e, sys) from e


def text_to_padded_sequence_for_inference(
    text: str,
    tokenizer: Tokenizer,
    maxlen: int = MAX_SEQUENCE_LENGTH,
    padding: str = PADDING_TYPE,
    truncating: str = TRUNCATING_TYPE,
) -> np.ndarray:
    """Converts a single raw text string into a padded sequence for model inference.

    Args:
        text (str): Input text string.
        tokenizer (Tokenizer): Fitted Tokenizer instance.
        maxlen (int): Maximum length of the output sequence. Defaults to
            MAX_SEQUENCE_LENGTH.
        padding (str): 'pre' or 'post' padding. Defaults to PADDING_TYPE.
        truncating (str): 'pre' or 'post' truncating. Defaults to
            TRUNCATING_TYPE.

    Returns:
        np.ndarray: Padded 2D numpy array of shape (1, maxlen).

    Raises:
        CustomException: If converting text for inference fails.
    """
    try:
        # Enclose single text in list
        sequences = convert_texts_to_sequences(tokenizer, [text])
        padded = pad_text_sequences(
            sequences, maxlen=maxlen, padding=padding, truncating=truncating
        )
        return padded
    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info("Starting standalone tokenizer verification...")

        # Setup paths
        test_dir = Path(__file__).resolve().parent / "test_output"
        test_dir.mkdir(exist_ok=True)
        tokenizer_path = test_dir / "test_tokenizer.pkl"

        # Mock texts
        mock_texts = [
            "the quick brown fox jumps over the lazy dog",
            "artificial intelligence is changing the software industry",
            "sports statistics and predictions are highly analytical",
        ]

        # 1. Create Tokenizer
        tok = create_tokenizer(vocab_size=100, oov_token="<OOV>")

        # 2. Fit Tokenizer
        tok = fit_tokenizer_on_text(tok, mock_texts)

        # 3. Save Tokenizer
        save_tokenizer(tok, tokenizer_path)

        # 4. Load Tokenizer
        loaded_tok = load_tokenizer(tokenizer_path)

        # 5. Convert to sequence for inference
        inference_text = "the quick dog plays sports"
        padded_seq = text_to_padded_sequence_for_inference(
            text=inference_text,
            tokenizer=loaded_tok,
            maxlen=10
        )

        print(f"\nInference Text: '{inference_text}'")
        print(f"Padded Sequence:\n{padded_seq}")
        print(f"Padded Sequence Shape: {padded_seq.shape}")

        # Clean up
        if tokenizer_path.exists():
            tokenizer_path.unlink()
        if test_dir.exists():
            test_dir.rmdir()

    except Exception as error:
        print(f"Verification failed: {error}")
