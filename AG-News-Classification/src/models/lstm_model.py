"""LSTM model module for the AGNews Text Classification project.

This module implements a Long Short-Term Memory (LSTM) model for text
classification, inheriting from the abstract `BaseModel` class.
"""

import sys
from pathlib import Path
from typing import Any

import tensorflow as tf

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.config import (
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EMBEDDING_DIM,
    LSTM_DROPOUT,
    LSTM_RECURRENT_DROPOUT,
    LSTM_UNITS,
    MAX_SEQUENCE_LENGTH,
    NUM_CLASSES,
    SAVED_MODELS_DIR,
    VOCAB_SIZE,
)
from src.models.base_model import BaseModel
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LSTMModel(BaseModel):
    """LSTM model class for text classification.

    This class builds, compiles, and configures callbacks for an LSTM-based
    model architecture utilizing Keras embedding and LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        embedding_dim: int = EMBEDDING_DIM,
        lstm_units: int = LSTM_UNITS,
        lstm_dropout: float = LSTM_DROPOUT,
        lstm_recurrent_dropout: float = LSTM_RECURRENT_DROPOUT,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Initializes the LSTM model with architecture parameters.

        Args:
            vocab_size (int): Size of the vocabulary. Defaults to VOCAB_SIZE.
            max_sequence_length (int): Maximum input sequence length.
                Defaults to MAX_SEQUENCE_LENGTH.
            embedding_dim (int): Dimensionality of the dense embedding.
                Defaults to EMBEDDING_DIM.
            lstm_units (int): Dimensionality of the LSTM output space.
                Defaults to LSTM_UNITS.
            lstm_dropout (float): Dropout rate for linear transformations of input.
                Defaults to LSTM_DROPOUT.
            lstm_recurrent_dropout (float): Dropout rate for recurrent state.
                Defaults to LSTM_RECURRENT_DROPOUT.
            num_classes (int): Number of target classification classes.
                Defaults to NUM_CLASSES.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.lstm_recurrent_dropout = lstm_recurrent_dropout
        self.num_classes = num_classes

    def build_model(self) -> tf.keras.Model:
        """Builds and returns the LSTM model architecture.

        Returns:
            tf.keras.Model: The constructed Keras model instance.

        Raises:
            CustomException: If model building fails.
        """
        try:
            self.logger.info("Building LSTM model architecture...")

            inputs = tf.keras.Input(
                shape=(self.max_sequence_length,),
                dtype=tf.int32,
                name="input_layer",
            )

            x = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                name="embedding_layer",
            )(inputs)

            x = tf.keras.layers.LSTM(
                units=self.lstm_units,
                dropout=self.lstm_dropout,
                recurrent_dropout=self.lstm_recurrent_dropout,
                name="lstm_layer",
            )(x)

            x = tf.keras.layers.Dropout(
                rate=self.lstm_dropout,
                name="dropout_layer",
            )(x)

            outputs = tf.keras.layers.Dense(
                units=self.num_classes,
                activation="softmax",
                name="output_layer",
            )(x)

            self.model = tf.keras.Model(
                inputs=inputs,
                outputs=outputs,
                name="LSTMModel",
            )

            self.logger.info("LSTM model architecture built successfully.")
            return self.model
        except Exception as e:
            raise CustomException(e, sys) from e

    def compile_model(
        self,
        optimizer: str | tf.keras.optimizers.Optimizer = "adam",
        loss: str | tf.keras.losses.Loss = "sparse_categorical_crossentropy",
        metrics: list[str | tf.keras.metrics.Metric] | None = None,
        **kwargs: Any,
    ) -> None:
        """Compiles the Keras model with optimizer, loss, and metrics.

        Args:
            optimizer (str | tf.keras.optimizers.Optimizer): Optimizer instance or name.
                Defaults to "adam".
            loss (str | tf.keras.losses.Loss): Loss function instance or name.
                Defaults to "sparse_categorical_crossentropy".
            metrics (list[str | tf.keras.metrics.Metric] | None): List of metrics to monitor.
                Defaults to None, which uses ["accuracy"].
            **kwargs (Any): Additional keyword arguments passed to Keras compile.

        Raises:
            CustomException: If model compilation fails.
        """
        try:
            if self.model is None:
                raise ValueError(
                    "Model is not built. Call build_model() before compile_model()."
                )

            if metrics is None:
                metrics = ["accuracy"]

            self.logger.info(
                f"Compiling LSTM model (optimizer={optimizer}, loss={loss}, metrics={metrics})"
            )
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                **kwargs,
            )
            self.logger.info("LSTM model compiled successfully.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_model(self) -> tf.keras.Model:
        """Retrieves the Keras model instance.

        Returns:
            tf.keras.Model: The compiled or uncompiled Keras model.

        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            self.logger.error("Attempted to get model before building it.")
            raise ValueError(
                "Model has not been built. Please call build_model() first."
            )
        return self.model

    def get_callbacks(self, **kwargs: Any) -> list[tf.keras.callbacks.Callback]:
        """Generates standard Keras callbacks for model training.

        Includes EarlyStopping and ModelCheckpoint.

        Args:
            **kwargs (Any): Overrides for callback settings:
                - patience (int): EarlyStopping patience. Defaults to EARLY_STOPPING_PATIENCE.
                - min_delta (float): EarlyStopping min_delta. Defaults to EARLY_STOPPING_MIN_DELTA.
                - checkpoint_dir (str | Path): Directory to save model checkpoint.
                  Defaults to SAVED_MODELS_DIR.

        Returns:
            list[tf.keras.callbacks.Callback]: A list of Keras callback instances.

        Raises:
            CustomException: If callback creation fails.
        """
        try:
            patience = kwargs.get("patience", EARLY_STOPPING_PATIENCE)
            min_delta = kwargs.get("min_delta", EARLY_STOPPING_MIN_DELTA)

            checkpoint_dir = Path(kwargs.get("checkpoint_dir", SAVED_MODELS_DIR))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "lstm_best_model.h5"

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    min_delta=min_delta,
                    restore_best_weights=True,
                    verbose=1,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path),
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                ),
            ]
            self.logger.info(
                f"Generated {len(callbacks)} callbacks for LSTM training."
            )
            return callbacks
        except Exception as e:
            raise CustomException(e, sys) from e

    def summary(self) -> None:
        """Prints the model summary.

        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            self.logger.error(
                "Attempted to print summary before building the model."
            )
            raise ValueError(
                "Model has not been built. Please call build_model() first."
            )
        self.model.summary()
