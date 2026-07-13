"""Inference pipeline for the AGNews Text Classification project.

This module provides a production-grade `NLPPredictor` class that accepts raw text,
applies the standard cleaning/preprocessing pipeline, performs neural network inference,
and returns structured predictions (predicted class, confidence score, probability distribution).
It supports RNN, LSTM, BiLSTM + Self-Attention, and BERT architectures.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tensorflow as tf

from configs.config import BATCH_SIZE, BERT_MAX_LENGTH, MAX_SEQUENCE_LENGTH, SAVED_MODELS_DIR
from src.data.label_encoder import CustomLabelEncoder, load_label_encoder
from src.data.preprocessing import clean_text
from src.utils.exception import CustomException
from src.utils.file_io import load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NLPPredictor:
    """Production-grade Predictor Pipeline for NLP Text Classification.

    Manages model initialization, tokenizer and label encoder loading, raw text cleaning,
    tokenization/padding, batch inference execution, and response formatting.
    """

    def __init__(
        self,
        model_type: str,
        model_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        label_encoder_path: Optional[Union[str, Path]] = None,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        bert_max_length: int = BERT_MAX_LENGTH,
    ) -> None:
        """Initializes the NLPPredictor and caches model resources.

        Args:
            model_type (str): Architecture key ('rnn', 'lstm', 'attention', 'bert').
            model_path (Optional[Union[str, Path]]): File path to saved model checkpoint.
                If None, resolves path automatically from default checkpoints.
            tokenizer_path (Optional[Union[str, Path]]): File path to saved Tokenizer file.
            label_encoder_path (Optional[Union[str, Path]]): File path to saved Label Encoder.
            max_sequence_length (int): Padding length for standard models. Defaults to MAX_SEQUENCE_LENGTH.
            bert_max_length (int): Padding length for BERT model. Defaults to BERT_MAX_LENGTH.

        Raises:
            ValueError: If the model_type is invalid.
            CustomException: If asset loading fails.
        """
        self.logger = get_logger(self.__class__.__name__)
        try:
            self.model_type = model_type.lower()
            self.max_sequence_length = max_sequence_length
            self.bert_max_length = bert_max_length

            valid_models = ["rnn", "lstm", "attention", "bert"]
            if self.model_type not in valid_models:
                raise ValueError(f"Invalid model_type: '{self.model_type}'. Expected one of {valid_models}")

            # Pre-configure environment settings if BERT
            if self.model_type == "bert":
                os.environ["TF_USE_LEGACY_KERAS"] = "1"

            # 1. Resolve and validate file paths
            self.model_path = self._resolve_model_path(model_path)
            self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else SAVED_MODELS_DIR / f"{self.model_type}_tokenizer.pkl"
            self.label_encoder_path = Path(label_encoder_path) if label_encoder_path else SAVED_MODELS_DIR / f"{self.model_type}_label_encoder.pkl"

            # 2. Load helper objects
            self.label_encoder = self._load_label_encoder()
            self.tokenizer = self._load_tokenizer()

            # 3. Load Keras model
            self.model = self._load_model()

            self.logger.info(f"NLPPredictor successfully initialized for model: '{self.model_type}'")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Path:
        """Resolves model path to a valid file, checking final and best model candidates.

        Args:
            model_path (Optional[Union[str, Path]]): Provided model path.

        Returns:
            Path: Resolved Path instance.

        Raises:
            FileNotFoundError: If no checkpoint is found.
        """
        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found at: {path}")
            return path

        # Try default locations
        candidates = []
        if self.model_type == "attention":
            candidates = ["attention_final_model.h5", "bilstm_attention_best_model.h5"]
        elif self.model_type == "bert":
            candidates = ["bert_final_model.h5", "bert_best_model.h5"]
        else:
            candidates = [f"{self.model_type}_final_model.h5", f"{self.model_type}_best_model.h5"]

        for filename in candidates:
            path = SAVED_MODELS_DIR / filename
            if path.exists():
                self.logger.info(f"Resolved model checkpoint dynamically: {path}")
                return path

        raise FileNotFoundError(
            f"Could not locate any default model checkpoints for type '{self.model_type}' in {SAVED_MODELS_DIR}. "
            f"Checked candidates: {candidates}"
        )

    def _load_label_encoder(self) -> CustomLabelEncoder:
        """Loads CustomLabelEncoder from files.

        Returns:
            CustomLabelEncoder: Loaded label encoder.

        Raises:
            FileNotFoundError: If encoder file doesn't exist.
            CustomException: If loading fails.
        """
        try:
            if not self.label_encoder_path.exists():
                raise FileNotFoundError(f"Label encoder file not found at: {self.label_encoder_path}")
            return load_label_encoder(self.label_encoder_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_tokenizer(self) -> Any:
        """Loads Keras or HuggingFace Tokenizer.

        Returns:
            Any: The loaded tokenizer instance.

        Raises:
            CustomException: If loading fails.
        """
        try:
            if self.model_type == "bert":
                from src.data.bert_tokenizer import load_bert_tokenizer
                if self.tokenizer_path.exists():
                    self.logger.info(f"Loading BERT tokenizer from file: {self.tokenizer_path}")
                    return load_pickle(self.tokenizer_path)
                else:
                    self.logger.info("BERT tokenizer file missing. Loading pretrained tokenizer dynamically...")
                    return load_bert_tokenizer()
            else:
                if not self.tokenizer_path.exists():
                    raise FileNotFoundError(f"Tokenizer file not found at: {self.tokenizer_path}")
                return load_pickle(self.tokenizer_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_model(self) -> tf.keras.Model:
        """Loads checkpoint and maps Custom Layer classes.

        Returns:
            tf.keras.Model: Loaded Keras model instance.

        Raises:
            CustomException: If model loading fails.
        """
        try:
            self.logger.info(f"Loading Keras model from: {self.model_path}")
            custom_objects: Dict[str, Any] = {}

            if self.model_type == "attention":
                from src.models.custom_attention import CustomSelfAttention
                custom_objects["CustomSelfAttention"] = CustomSelfAttention
            elif self.model_type == "bert":
                from transformers import TFBertModel
                custom_objects["TFBertModel"] = TFBertModel

            model = tf.keras.models.load_model(
                str(self.model_path),
                custom_objects=custom_objects
            )
            return model
        except Exception as e:
            raise CustomException(e, sys) from e

    def preprocess(self, texts: List[str]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Preprocesses raw texts into standard numerical inputs.

        Applies text cleaning and sequence padding.

        Args:
            texts (List[str]): Raw input text documents.

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Model inputs.

        Raises:
            CustomException: If preprocessing fails.
        """
        try:
            # Apply modular cleaning steps to raw text
            cleaned_texts = [clean_text(t) for t in texts]

            if self.model_type == "bert":
                from src.data.bert_tokenizer import tokenize_text_data
                tokenized = tokenize_text_data(
                    self.tokenizer,
                    cleaned_texts,
                    max_length=self.bert_max_length
                )
                return {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "token_type_ids": tokenized["token_type_ids"],
                }
            else:
                from src.data.tokenizer_utils import convert_texts_to_sequences, pad_text_sequences
                sequences = convert_texts_to_sequences(self.tokenizer, cleaned_texts)
                padded = pad_text_sequences(sequences, maxlen=self.max_sequence_length)
                return padded
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(
        self,
        text_input: Union[str, List[str]],
        batch_size: int = BATCH_SIZE
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Runs inference on single or multiple text documents.

        Args:
            text_input (Union[str, List[str]]): Raw text input string or list of text strings.
            batch_size (int): Inference batch size. Defaults to BATCH_SIZE.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Prediction details.
                If input is a string, returns a single dictionary.
                If input is a list, returns a list of dictionaries.
                Each dictionary contains:
                    - "predicted_class": String name of the predicted category.
                    - "confidence_score": Probability float value of predicted category.
                    - "probability_distribution": Dictionary mapping categories to probabilities.

        Raises:
            CustomException: If prediction fails.
        """
        try:
            # 1. Determine if single or batch input
            is_single = isinstance(text_input, str)
            texts = [text_input] if is_single else text_input

            if not texts:
                return [] if not is_single else {}

            # 2. Preprocess raw text input
            inputs = self.preprocess(texts)

            # 3. Model Prediction
            probabilities = self.model.predict(inputs, batch_size=batch_size, verbose=0)
            predictions = np.argmax(probabilities, axis=1)

            # 4. Resolve target label maps
            sorted_indices = sorted(list(self.label_encoder.idx_to_class.keys()))
            class_names = [
                self.label_encoder.class_labels_map.get(
                    self.label_encoder.idx_to_class[idx],
                    str(self.label_encoder.idx_to_class[idx])
                )
                for idx in sorted_indices
            ]

            # 5. Format results
            results = []
            for i, probs in enumerate(probabilities):
                pred_idx = predictions[i]
                predicted_class = class_names[pred_idx]
                confidence_score = float(probs[pred_idx])

                # Construct probability distribution mapping
                prob_dist = {
                    class_name: float(probs[idx])
                    for idx, class_name in enumerate(class_names)
                }

                results.append({
                    "predicted_class": predicted_class,
                    "confidence_score": confidence_score,
                    "probability_distribution": prob_dist
                })

            return results[0] if is_single else results

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Classification Model Prediction CLI.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["rnn", "lstm", "attention", "bert"],
        help="Model architecture type to use for prediction."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Raw text string to classify."
    )

    try:
        args = parser.parse_args()
        predictor = NLPPredictor(model_type=args.model)
        result = predictor.predict(args.text)
        
        import json
        print("\n" + "=" * 80)
        print("NLP INFERENCE RESULTS (JSON)")
        print("=" * 80)
        print(json.dumps(result, indent=4))
        print("=" * 80 + "\n")

    except Exception as error:
        logger.error(f"Prediction script failed: {error}")
        sys.exit(1)
