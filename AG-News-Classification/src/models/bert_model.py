"""BERT model module for the AGNews Text Classification project.

This module implements a Bidirectional Encoder Representations from Transformers (BERT)
model for text classification, inheriting from the abstract `BaseModel` class.
"""

import sys
from pathlib import Path
from typing import Any

import tensorflow as tf
from transformers import TFBertModel

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.config import (
    BERT_FINE_TUNE_LAYERS,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    NUM_CLASSES,
    SAVED_MODELS_DIR,
)
from src.models.base_model import BaseModel
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BERTModel(BaseModel):
    """BERT-based Transformer model class for text classification.

    This class builds, compiles, and configures callbacks for a BERT-based
    model architecture utilizing the HuggingFace `TFBertModel` representation,
    followed by a dropout layer and a dense classification head.
    """

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        max_length: int = BERT_MAX_LENGTH,
        fine_tune_layers: int = BERT_FINE_TUNE_LAYERS,
        dropout_rate: float = 0.2,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Initializes the BERTModel with architecture parameters.

        Args:
            model_name (str): HuggingFace pretrained model identifier.
                Defaults to BERT_MODEL_NAME ("bert-base-uncased").
            max_length (int): Maximum input sequence length.
                Defaults to BERT_MAX_LENGTH.
            fine_tune_layers (int): Number of top encoder layers to fine-tune.
                Defaults to BERT_FINE_TUNE_LAYERS.
            dropout_rate (float): Dropout rate applied to BERT output representation.
                Defaults to 0.2.
            num_classes (int): Number of target classification classes.
                Defaults to NUM_CLASSES.
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.fine_tune_layers = fine_tune_layers
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

    def build_model(self) -> tf.keras.Model:
        """Builds and returns the BERT text classification model architecture.

        This method initializes the pretrained HuggingFace TFBertModel, configures
        layer freeze/unfreeze based on the `fine_tune_layers` parameter, constructs
        the Keras input and classification heads, and registers the final Model.

        Returns:
            tf.keras.Model: The constructed Keras model instance.

        Raises:
            CustomException: If model building fails.
        """
        try:
            self.logger.info(
                f"Building BERT model architecture (model_name={self.model_name})..."
            )

            # Input layer for token IDs
            # # Shape matches max sequence length expected by the model
            # inputs = tf.keras.Input(
            #     shape=(self.max_length,),
            #     dtype=tf.int32,
            #     name="input_layer",
            # )

            # Load HuggingFace Pretrained BERT Layer
            # We load the weights of 'bert-base-uncased'
            # We use use_safetensors=False to avoid safetensors loading errors in Keras/TF environments
            bert = TFBertModel.from_pretrained(
                self.model_name,
                from_pt=True
            )
            # Configure fine-tuning parameters
            if self.fine_tune_layers > 0:
                bert.trainable = True
                
                # Freeze embeddings
                if hasattr(bert, "bert") and hasattr(bert.bert, "embeddings"):
                    bert.bert.embeddings.trainable = False
                
                # Freeze encoder layers except the last N
                if hasattr(bert, "bert") and hasattr(bert.bert, "encoder") and hasattr(bert.bert.encoder, "layer"):
                    encoder_layers = bert.bert.encoder.layer
                    num_layers = len(encoder_layers)
                    for i in range(num_layers - self.fine_tune_layers):
                        encoder_layers[i].trainable = False
                    for i in range(num_layers - self.fine_tune_layers, num_layers):
                        encoder_layers[i].trainable = True
                    self.logger.info(
                        f"Configured top {self.fine_tune_layers} encoder layers of "
                        f"BERT as trainable. Rest of the layers are frozen."
                    )
                else:
                    self.logger.warning(
                        "Could not locate internal BERT encoder layers for fine-tuning configuration. "
                        "Defaulting to training the entire BERT model."
                    )
            elif self.fine_tune_layers == 0:
                bert.trainable = False
                self.logger.info("All layers of the BERT model are frozen.")
            else:
                bert.trainable = True
                self.logger.info("All layers of the BERT model are trainable.")

            # Compute attention mask dynamically based on padding tokens (where value != 0)
            # This is essential since standard Keras tokenization output is padded with 0
            # attention_mask = tf.keras.layers.Lambda(
            #     lambda x: tf.cast(tf.math.not_equal(x, 0), tf.int32),
            #     name="attention_mask_layer",
            # )(inputs)
            input_ids = tf.keras.Input(
                shape=(self.max_length,),
                dtype=tf.int32,
                name="input_ids",
            )

            attention_mask = tf.keras.Input(
                shape=(self.max_length,),
                dtype=tf.int32,
                name="attention_mask",
            )

            token_type_ids = tf.keras.Input(
                shape=(self.max_length,),
                dtype=tf.int32,
                name="token_type_ids",
            )

            # Pass inputs through BERT positionally (input_ids, attention_mask)
            # This avoids Keras 3 KerasTensor type-checking errors in HuggingFace keyword unpacking
            bert_outputs = bert(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
            )
            # Extract pooler output for classification
            if isinstance(bert_outputs, tuple) or isinstance(bert_outputs, list):
                pooled_output = bert_outputs[1]
            else:
                pooled_output = bert_outputs.pooler_output

            # Add Dropout for regularization
            x = tf.keras.layers.Dropout(
                rate=self.dropout_rate,
                name="dropout_layer",
            )(pooled_output)

            # Dense output layer with softmax activation
            outputs = tf.keras.layers.Dense(
                units=self.num_classes,
                activation="softmax",
                name="output_layer",
            )(x)

            self.model = tf.keras.Model(
                inputs={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
                outputs=outputs,
                name="BERTModel",
            )

            self.logger.info("BERT model architecture built successfully.")
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
                f"Compiling BERT model (optimizer={optimizer}, loss={loss}, metrics={metrics})"
            )
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                **kwargs,
            )
            self.logger.info("BERT model compiled successfully.")
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
            checkpoint_path = checkpoint_dir / "bert_best_model.h5"

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
                    save_weights_only=False,
                    verbose=1,
                ),
            ]
            self.logger.info(
                f"Generated {len(callbacks)} callbacks for BERT training."
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
