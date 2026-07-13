"""NLP Model Evaluator module for the AGNews Text Classification project.

This module provides an enterprise-grade `NLPEvaluator` class to load trained models,
preprocess test data, generate predictions, compute classification metrics,
and export evaluation metrics to JSON. It supports RNN, LSTM, BiLSTM + Attention, and BERT models.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Configure legacy Keras for BERT model compatibility before importing TensorFlow
# Set it globally or locally depending on model type
if "--model" in sys.argv:
    try:
        model_idx = sys.argv.index("--model")
        if model_idx + 1 < len(sys.argv) and sys.argv[model_idx + 1] == "bert":
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
    except ValueError:
        pass

import tensorflow as tf

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.config import BATCH_SIZE, BERT_MAX_LENGTH, MAX_SEQUENCE_LENGTH, LABEL_COLUMN, TEXT_COLUMN
from src.data.label_encoder import CustomLabelEncoder, encode_labels, load_label_encoder
from src.utils.exception import CustomException
from src.utils.file_io import load_pickle, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NLPEvaluator:
    """Enterprise-level NLP Model Evaluator.

    Supports loading trained neural networks (RNN, LSTM, BiLSTM + Self-Attention, BERT),
    generating predictions, computing standard classification metrics, and exporting
    results in a JSON format.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        model_type: str,
        tokenizer_path: Optional[Union[str, Path]] = None,
        label_encoder_path: Optional[Union[str, Path]] = None,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        bert_max_length: int = BERT_MAX_LENGTH,
    ) -> None:
        """Initializes the NLPEvaluator with configuration paths.

        Args:
            model_path (Union[str, Path]): Path to the trained Keras model file (.h5).
            model_type (str): Type of model architecture ('rnn', 'lstm', 'attention', 'bert').
            tokenizer_path (Optional[Union[str, Path]]): Path to saved Keras/BERT tokenizer.
            label_encoder_path (Optional[Union[str, Path]]): Path to saved Label Encoder pickle file.
            max_sequence_length (int): Max sequence length for standard models. Defaults to MAX_SEQUENCE_LENGTH.
            bert_max_length (int): Max sequence length for BERT model. Defaults to BERT_MAX_LENGTH.

        Raises:
            ValueError: If the model_type is invalid.
            CustomException: If initialization or file loading fails.
        """
        self.logger = get_logger(self.__class__.__name__)
        try:
            self.model_path = Path(model_path)
            self.model_type = model_type.lower()
            self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
            self.label_encoder_path = Path(label_encoder_path) if label_encoder_path else None
            self.max_sequence_length = max_sequence_length
            self.bert_max_length = bert_max_length

            valid_models = ["rnn", "lstm", "attention", "bert"]
            if self.model_type not in valid_models:
                raise ValueError(
                    f"Invalid model_type: '{self.model_type}'. Expected one of {valid_models}"
                )

            # Pre-configure environment settings if BERT
            if self.model_type == "bert":
                os.environ["TF_USE_LEGACY_KERAS"] = "1"

            # Load helper objects
            self.label_encoder: Optional[CustomLabelEncoder] = self._load_label_encoder()
            self.tokenizer: Any = self._load_tokenizer()
            self.model: tf.keras.Model = self.load_model()

            self.logger.info(f"NLPEvaluator successfully initialized for model: '{self.model_type}'")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_label_encoder(self) -> Optional[CustomLabelEncoder]:
        """Loads CustomLabelEncoder from the configured path.

        Returns:
            Optional[CustomLabelEncoder]: The loaded encoder or None if no path provided.

        Raises:
            CustomException: If loading fails.
        """
        try:
            if not self.label_encoder_path:
                self.logger.warning("No label encoder path provided. Evaluator will run without target decoding.")
                return None
            
            if not self.label_encoder_path.exists():
                raise FileNotFoundError(f"Label encoder not found at: {self.label_encoder_path}")

            self.logger.info(f"Loading label encoder from: {self.label_encoder_path}")
            return load_label_encoder(self.label_encoder_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_tokenizer(self) -> Any:
        """Loads Tokenizer from the configured path.

        For BERT, imports HuggingFace's tokenizer dynamically if no path is provided.

        Returns:
            Any: The loaded tokenizer instance.

        Raises:
            ValueError: If tokenizer path is missing for standard models.
            CustomException: If loading fails.
        """
        try:
            if self.model_type == "bert":
                from src.data.bert_tokenizer import load_bert_tokenizer
                if self.tokenizer_path and self.tokenizer_path.exists():
                    self.logger.info(f"Loading BERT tokenizer from: {self.tokenizer_path}")
                    return load_pickle(self.tokenizer_path)
                else:
                    self.logger.info("Loading default pretrained BERT tokenizer...")
                    return load_bert_tokenizer()
            else:
                if not self.tokenizer_path:
                    raise ValueError(f"tokenizer_path must be provided for model type '{self.model_type}'")
                
                if not self.tokenizer_path.exists():
                    raise FileNotFoundError(f"Tokenizer not found at: {self.tokenizer_path}")
                
                self.logger.info(f"Loading Keras Tokenizer from: {self.tokenizer_path}")
                return load_pickle(self.tokenizer_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def load_model(self) -> tf.keras.Model:
        """Loads the compiled Keras model from disk.

        Injects Custom Layer objects (such as `CustomSelfAttention` and `TFBertModel`)
        dynamically based on the selected architecture.

        Returns:
            tf.keras.Model: Loaded Keras model instance.

        Raises:
            FileNotFoundError: If model file does not exist.
            CustomException: If model loading fails.
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Trained model file not found at: {self.model_path}")

            self.logger.info(f"Loading Keras model from: {self.model_path}")
            custom_objects: Dict[str, Any] = {}

            if self.model_type == "attention":
                from src.models.custom_attention import CustomSelfAttention
                custom_objects["CustomSelfAttention"] = CustomSelfAttention
                self.logger.info("Registered 'CustomSelfAttention' class in custom_objects.")

            elif self.model_type == "bert":
                from transformers import TFBertModel
                custom_objects["TFBertModel"] = TFBertModel
                self.logger.info("Registered 'TFBertModel' class in custom_objects.")

            model = tf.keras.models.load_model(
                str(self.model_path),
                custom_objects=custom_objects
            )
            self.logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            raise CustomException(e, sys) from e

    def preprocess_data(self, texts: List[str]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Preprocesses raw texts into standard numerical representations for inference.

        Args:
            texts (List[str]): Raw input text documents.

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Formatted model inputs.

        Raises:
            CustomException: If pre-processing fails.
        """
        try:
            self.logger.info(f"Preprocessing {len(texts)} samples...")
            
            if self.model_type == "bert":
                from src.data.bert_tokenizer import tokenize_text_data
                tokenized = tokenize_text_data(
                    self.tokenizer,
                    texts,
                    max_length=self.bert_max_length
                )
                # Keep inputs formatted as a dictionary of tensors/arrays
                return {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "token_type_ids": tokenized["token_type_ids"],
                }
            else:
                from src.data.tokenizer_utils import convert_texts_to_sequences, pad_text_sequences
                sequences = convert_texts_to_sequences(self.tokenizer, texts)
                padded_features = pad_text_sequences(sequences, maxlen=self.max_sequence_length)
                return padded_features
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, texts: List[str], batch_size: int = BATCH_SIZE) -> Tuple[np.ndarray, np.ndarray]:
        """Generates predictions and class probability scores for test texts.

        Args:
            texts (List[str]): Raw text sequences.
            batch_size (int): Inference batch size. Defaults to BATCH_SIZE.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted class indices and probability distributions.

        Raises:
            CustomException: If prediction fails.
        """
        try:
            inputs = self.preprocess_data(texts)
            self.logger.info("Running batch inference...")
            probabilities = self.model.predict(inputs, batch_size=batch_size, verbose=0)
            predictions = np.argmax(probabilities, axis=1)
            return predictions, probabilities
        except Exception as e:
            raise CustomException(e, sys) from e

    def _prepare_labels(self, labels: List[Any]) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Resolves raw labels to 0-indexed values and fetches friendly target names.

        Args:
            labels (List[Any]): List of true labels.

        Returns:
            Tuple[np.ndarray, Optional[List[str]]]: Encoded integer labels and list of class names.

        Raises:
            CustomException: If labels preparation fails.
        """
        try:
            if not self.label_encoder:
                self.logger.warning("No label encoder loaded. Class labels will be evaluated as-is.")
                y_true = np.array(labels)
                unique_classes = sorted(list(set(labels)))
                target_names = [str(c) for c in unique_classes]
                return y_true, target_names

            # Examine the first label to determine if encoding is required
            first_label = labels[0]
            if first_label in self.label_encoder.class_to_idx:
                self.logger.info("Encoding raw class labels using CustomLabelEncoder...")
                y_true = np.array(encode_labels(self.label_encoder, labels))
            else:
                self.logger.info("True labels appear to be pre-encoded integers.")
                y_true = np.array(labels)

            # Collect unique class names ordered by their 0-indexed positions
            sorted_indices = sorted(list(self.label_encoder.idx_to_class.keys()))
            target_names = [
                self.label_encoder.class_labels_map.get(
                    self.label_encoder.idx_to_class[idx],
                    str(self.label_encoder.idx_to_class[idx])
                )
                for idx in sorted_indices
            ]
            return y_true, target_names
        except Exception as e:
            raise CustomException(e, sys) from e

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Computes NLP classification metrics via scikit-learn.

        Args:
            y_true (np.ndarray): 0-indexed true labels.
            y_pred (np.ndarray): 0-indexed predicted labels.
            target_names (Optional[List[str]]): Target names mapping.

        Returns:
            Dict[str, Any]: Metrics dictionary containing accuracy, precision, recall, f1, etc.

        Raises:
            CustomException: If metrics generation fails.
        """
        try:
            self.logger.info("Computing classification metrics...")
            
            # Compute Accuracy
            accuracy = float(accuracy_score(y_true, y_pred))

            # Compute Precision, Recall, F1
            p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0
            )
            p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )

            # Generate classification reports
            report_dict = classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
            report_str = classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                output_dict=False,
                zero_division=0
            )
            
            # Print report to logs
            self.logger.info(
                f"\n"
                f"============================================================\n"
                f"CLASSIFICATION REPORT:\n"
                f"============================================================\n"
                f"{report_str}"
                f"============================================================\n"
            )

            # Compute Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)

            metrics = {
                "accuracy": accuracy,
                "precision_macro": float(p_macro),
                "recall_macro": float(r_macro),
                "f1_macro": float(f_macro),
                "precision_micro": float(p_micro),
                "recall_micro": float(r_micro),
                "f1_micro": float(f_micro),
                "precision_weighted": float(p_weighted),
                "recall_weighted": float(r_weighted),
                "f1_weighted": float(f_weighted),
                "classification_report": report_dict,
                "confusion_matrix": cm.tolist(),  # Serialized list for JSON output
            }
            return metrics
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate(
        self,
        texts: List[str],
        labels: List[Any],
        batch_size: int = BATCH_SIZE,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs the complete NLP evaluation pipeline.

        Args:
            texts (List[str]): Input texts.
            labels (List[Any]): Target true labels.
            batch_size (int): inference batch size. Defaults to BATCH_SIZE.
            save_path (Optional[Union[str, Path]]): File path to save evaluation results in JSON format.

        Returns:
            Dict[str, Any]: Calculated metrics.

        Raises:
            CustomException: If evaluation fails.
        """
        try:
            self.logger.info(f"Starting NLP model evaluation for {len(texts)} samples.")
            
            # 1. Predictions
            y_pred, _ = self.predict(texts, batch_size=batch_size)

            # 2. Resolve label shapes and classes
            y_true, target_names = self._prepare_labels(labels)

            # 3. Metrics calculation
            metrics = self.compute_metrics(y_true, y_pred, target_names=target_names)

            # 4. Save results to disk
            if save_path:
                save_path = Path(save_path)
                save_json(save_path, metrics)
                self.logger.info(f"Successfully saved evaluation metrics to: {save_path}")

            return metrics
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = TEXT_COLUMN,
        label_column: str = LABEL_COLUMN,
        batch_size: int = BATCH_SIZE,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs evaluation directly on a pandas DataFrame.

        Args:
            df (pd.DataFrame): Dataframe containing texts and labels.
            text_column (str): Text column name. Defaults to configs.config.TEXT_COLUMN.
            label_column (str): Label column name. Defaults to configs.config.LABEL_COLUMN.
            batch_size (int): inference batch size. Defaults to BATCH_SIZE.
            save_path (Optional[Union[str, Path]]): File path to save evaluation results in JSON format.

        Returns:
            Dict[str, Any]: Calculated metrics.

        Raises:
            ValueError: If text_column or label_column is missing from the dataframe.
            CustomException: If dataframe evaluation fails.
        """
        try:
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in DataFrame. Available: {list(df.columns)}")
            
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in DataFrame. Available: {list(df.columns)}")

            texts = df[text_column].tolist()
            labels = df[label_column].tolist()

            return self.evaluate(
                texts=texts,
                labels=labels,
                batch_size=batch_size,
                save_path=save_path,
            )
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    # Standard entry point execution warning
    logger.info("This module is built for enterprise imports. Run verify_evaluator.py for validations.")
