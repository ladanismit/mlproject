"""Custom self-attention layer module for enterprise NLP models.

This module implements a reusable, strongly-typed attention pooling layer (self-attention
with context) designed to process sequence outputs from recurrent layers (e.g., LSTM, RNN, BiLSTM).
It computes a weighted sum of sequence states to generate a fixed-size context vector,
along with the normalized attention weights.
"""

import sys
from pathlib import Path
from typing import Any

import tensorflow as tf

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class CustomSelfAttention(tf.keras.layers.Layer):
    """Custom self-attention layer (Attention with Context) for sequence outputs.

    This layer accepts a 3D tensor representing sequence outputs from recurrent layers
    of shape `(batch_size, sequence_length, feature_dim)` and computes attention scores
    for each time step. These scores are normalized using softmax to obtain attention
    weights, which are then used to produce a weighted representation of the sequence
    (the context vector).

    Attributes:
        attention_dim (int): Dimensionality of the attention space.
        kernel_initializer (tf.keras.initializers.Initializer): Initializer for weight matrices.
        bias_initializer (tf.keras.initializers.Initializer): Initializer for bias vectors.
        kernel_regularizer (tf.keras.regularizers.Regularizer | None): Regularizer for weights.
        bias_regularizer (tf.keras.regularizers.Regularizer | None): Regularizer for biases.
    """

    def __init__(
        self,
        attention_dim: int = 64,
        kernel_initializer: str | tf.keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | tf.keras.initializers.Initializer = "zeros",
        kernel_regularizer: str | tf.keras.regularizers.Regularizer | None = None,
        bias_regularizer: str | tf.keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the CustomSelfAttention layer.

        Args:
            attention_dim (int): Dimensionality of the attention projection. Defaults to 64.
            kernel_initializer (str | tf.keras.initializers.Initializer): Initializer for weight
                matrices. Defaults to "glorot_uniform".
            bias_initializer (str | tf.keras.initializers.Initializer): Initializer for bias
                vectors. Defaults to "zeros".
            kernel_regularizer (str | tf.keras.regularizers.Regularizer | None): Regularizer
                applied to weight matrices. Defaults to None.
            bias_regularizer (str | tf.keras.regularizers.Regularizer | None): Regularizer
                applied to bias vectors. Defaults to None.
            **kwargs (Any): Additional Keras layer keyword arguments (e.g., name, dtype).
        """
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.supports_masking = True

    def build(self, input_shape: tf.TensorShape) -> None:
        """Creates the weight and bias variables of the layer.

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.

        Raises:
            ValueError: If input tensor rank is not exactly 3.
            CustomException: If weight allocation fails.
        """
        try:
            if len(input_shape) != 3:
                raise ValueError(
                    f"CustomSelfAttention layer requires a 3D input tensor of shape "
                    f"(batch_size, sequence_length, feature_dim), but got shape {input_shape} "
                    f"with rank {len(input_shape)}."
                )

            feature_dim = input_shape[-1]

            # Weight mapping from feature space to attention space
            self.W_1 = self.add_weight(
                name="attention_weight_dense",
                shape=(feature_dim, self.attention_dim),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
            )

            # Bias for the attention space mapping
            self.b_1 = self.add_weight(
                name="attention_bias_dense",
                shape=(self.attention_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
            )

            # Projection vector to compute attention scores
            self.W_2 = self.add_weight(
                name="attention_weight_score",
                shape=(self.attention_dim, 1),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
            )

            super().build(input_shape)
        except Exception as e:
            logger.error("Failed to build CustomSelfAttention layer.")
            raise CustomException(e, sys) from e

    def call(
        self, inputs: tf.Tensor, mask: tf.Tensor | None = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Executes the attention pooling forward pass.

        Args:
            inputs (tf.Tensor): A 3D tensor of shape `(batch_size, sequence_length, feature_dim)`.
            mask (tf.Tensor | None): A boolean tensor of shape `(batch_size, sequence_length)`
                indicating valid tokens/steps. Defaults to None.

        Returns:
            tuple[tf.Tensor, tf.Tensor]:
                - context_vector: Weighted sum representation of shape `(batch_size, feature_dim)`.
                - attention_weights: Softmax-normalized attention scores of shape
                  `(batch_size, sequence_length, 1)`.

        Raises:
            CustomException: If execution of forward pass fails.
        """
        try:
            # 1. Project inputs: (batch_size, sequence_length, feature_dim) @ (feature_dim, attention_dim) -> (batch_size, sequence_length, attention_dim)
            u = tf.tanh(tf.matmul(inputs, self.W_1) + self.b_1)

            # 2. Compute attention scores: (batch_size, sequence_length, attention_dim) @ (attention_dim, 1) -> (batch_size, sequence_length, 1)
            scores = tf.matmul(u, self.W_2)

            # 3. Apply masking to ignore padded tokens
            if mask is not None:
                mask_bool = tf.cast(mask, tf.bool)
                if len(mask_bool.shape) == 2:
                    mask_bool = tf.expand_dims(mask_bool, axis=-1)
                
                # Apply mask: assign a large negative score to masked elements to nullify their softmax probability
                scores = tf.where(mask_bool, scores, tf.constant(-1e9, dtype=scores.dtype))

            # 4. Softmax normalization over the sequence length axis: (batch_size, sequence_length, 1)
            attention_weights = tf.nn.softmax(scores, axis=1)

            # 5. Weighted sum computation: (batch_size, sequence_length, feature_dim) * (batch_size, sequence_length, 1) -> sum over axis 1 -> (batch_size, feature_dim)
            context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)

            return context_vector, attention_weights
        except Exception as e:
            logger.error("Error encountered during CustomSelfAttention forward pass.")
            raise CustomException(e, sys) from e

    def compute_output_shape(
        self, input_shape: tf.TensorShape
    ) -> tuple[tf.TensorShape, tf.TensorShape]:
        """Computes the output shapes of the context vector and attention weights.

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.

        Returns:
            tuple[tf.TensorShape, tf.TensorShape]:
                - Output shape of the context vector: `(batch_size, feature_dim)`.
                - Output shape of the attention weights: `(batch_size, sequence_length, 1)`.
        """
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        feature_dim = input_shape[2]

        context_vector_shape = tf.TensorShape([batch_size, feature_dim])
        attention_weights_shape = tf.TensorShape([batch_size, sequence_length, 1])

        return context_vector_shape, attention_weights_shape

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self) -> dict[str, Any]:
        """Serializes the layer configuration parameters for model saving and loading.

        Returns:
            dict[str, Any]: Layer configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "attention_dim": self.attention_dim,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
