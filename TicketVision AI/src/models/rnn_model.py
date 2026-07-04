"""
rnn_model.py — Simple RNN Text Classifier for TicketVision-AI
=============================================================

Builds a baseline text-classification model using one or two
``tf.keras.layers.SimpleRNN`` layers with dropout regularisation.

Architecture
------------
::

    Input (MAX_SEQUENCE_LENGTH,)
      │
      ▼
    Embedding (VOCAB_SIZE × EMBEDDING_DIM, mask_zero)
      │
      ▼
    SimpleRNN-1 (RNN_UNITS, return_sequences=True)
      │
      ▼
    Dropout (DROPOUT_RATE)
      │
      ▼
    SimpleRNN-2 (RNN_UNITS, return_sequences=False)
      │
      ▼
    Dropout (DROPOUT_RATE)
      │
      ▼
    Dense (RNN_UNITS, relu)
      │
      ▼
    Dropout (DROPOUT_RATE)
      │
      ▼
    Dense (NUM_CLASSES, softmax)

This module is responsible **only** for model creation and compilation —
no training, data loading, or evaluation logic lives here.

Usage
-----
>>> from src.models.rnn_model import build_rnn_model
>>> model = build_rnn_model()
>>> model.summary()

Author : TicketVision-AI Team
Created: 2026-07-04
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Input,
    SimpleRNN,
)
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# Ensure project root is importable (same pattern as data_loader.py).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (  # noqa: E402
    DROPOUT_RATE,
    EMBEDDING_DIM,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    NUM_CLASSES,
    RECURRENT_DROPOUT,
    RNN_UNITS,
    VOCAB_SIZE,
    setup_logging,
)

# Module-level logger.
logger: logging.Logger = logging.getLogger("ticketvision.models.rnn")


# ============================================================================
# 1. EMBEDDING LAYER
# ============================================================================

def _build_embedding_layer(
    vocab_size: int = VOCAB_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
    input_length: int = MAX_SEQUENCE_LENGTH,
) -> Embedding:
    """Create a trainable word-embedding layer.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (including OOV).
    embedding_dim : int
        Dimensionality of each embedding vector.
    input_length : int
        Fixed length of every input sequence.

    Returns
    -------
    Embedding
        Configured Keras ``Embedding`` layer with zero-masking enabled.
    """
    return Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=input_length,
        mask_zero=True,
        name="embedding",
    )


# ============================================================================
# 2. RECURRENT BLOCK
# ============================================================================

def _build_rnn_block(
    x: tf.Tensor,
    *,
    units: int = RNN_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    recurrent_dropout: float = RECURRENT_DROPOUT,
    num_layers: int = 2,
) -> tf.Tensor:
    """Stack one or more ``SimpleRNN`` layers with inter-layer dropout.

    Parameters
    ----------
    x : tf.Tensor
        Output tensor from the embedding layer,
        shape ``(batch, seq_len, embedding_dim)``.
    units : int
        Number of units in each RNN cell.
    dropout_rate : float
        Dropout rate applied between layers.
    recurrent_dropout : float
        Dropout applied inside the recurrent cells.
    num_layers : int
        Total number of stacked SimpleRNN layers (1 or 2).

    Returns
    -------
    tf.Tensor
        Output of the final RNN layer, shape ``(batch, units)``.
    """
    for i in range(num_layers):
        return_sequences = i < (num_layers - 1)  # True for all but the last
        x = SimpleRNN(
            units=units,
            return_sequences=return_sequences,
            recurrent_dropout=recurrent_dropout,
            name=f"simple_rnn_{i + 1}",
        )(x)
        x = Dropout(rate=dropout_rate, name=f"rnn_dropout_{i + 1}")(x)
    return x


# ============================================================================
# 3. CLASSIFICATION HEAD
# ============================================================================

def _build_classification_head(
    x: tf.Tensor,
    *,
    dense_units: int = RNN_UNITS,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
) -> tf.Tensor:
    """Attach a dense classification head to the RNN output.

    Parameters
    ----------
    x : tf.Tensor
        Output from the recurrent block, shape ``(batch, units)``.
    dense_units : int
        Width of the hidden dense layer.
    num_classes : int
        Number of target categories.
    dropout_rate : float
        Dropout rate before the final projection.

    Returns
    -------
    tf.Tensor
        Probability distribution over classes, shape ``(batch, num_classes)``.
    """
    x = Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = Dropout(rate=dropout_rate, name="dense_dropout")(x)
    x = Dense(num_classes, activation="softmax", name="output")(x)
    return x


# ============================================================================
# 4. MODEL COMPILATION
# ============================================================================

def _compile_model(
    model: Model,
    *,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """Compile the model with Adam, sparse cross-entropy, and accuracy.

    Parameters
    ----------
    model : Model
        Uncompiled Keras model.
    learning_rate : float
        Learning rate for the Adam optimiser.

    Returns
    -------
    Model
        The same model, now compiled and ready for ``.fit()``.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(
        "Model compiled — optimizer=Adam(lr=%s), "
        "loss=SparseCategoricalCrossentropy, metrics=[accuracy].",
        learning_rate,
    )
    return model


# ============================================================================
# 5. PUBLIC API
# ============================================================================

def build_rnn_model(
    vocab_size: int = VOCAB_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    rnn_units: int = RNN_UNITS,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    recurrent_dropout: float = RECURRENT_DROPOUT,
    learning_rate: float = LEARNING_RATE,
    num_rnn_layers: int = 2,
    compile_model: bool = True,
) -> Model:
    """Build and optionally compile the Simple-RNN text classifier.

    All parameters default to their ``config.py`` values, but can be
    overridden for hyper-parameter sweeps or experimentation.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    embedding_dim : int
        Word-embedding dimensionality.
    max_sequence_length : int
        Fixed input sequence length (after padding / truncation).
    rnn_units : int
        Number of units per SimpleRNN cell.
    num_classes : int
        Number of target categories.
    dropout_rate : float
        Dropout rate between layers.
    recurrent_dropout : float
        Dropout inside recurrent cells.
    learning_rate : float
        Adam learning rate.
    num_rnn_layers : int
        Number of stacked SimpleRNN layers (1 or 2).
    compile_model : bool
        If ``True`` (default), compile before returning.

    Returns
    -------
    Model
        A Keras ``Model`` ready for training.

    Raises
    ------
    ValueError
        If ``num_rnn_layers`` is not 1 or 2, or if any numeric hyper-
        parameter is non-positive.
    """
    # --- Input validation ---
    if num_rnn_layers not in (1, 2):
        raise ValueError(
            f"num_rnn_layers must be 1 or 2, got {num_rnn_layers}."
        )
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(
            f"dropout_rate must be in [0, 1), got {dropout_rate}."
        )

    logger.info("Building Simple-RNN model …")
    logger.info(
        "  vocab=%s  embed=%d  seq_len=%d  rnn_units=%d  "
        "layers=%d  classes=%d  dropout=%.2f  rec_dropout=%.2f",
        f"{vocab_size:,}", embedding_dim, max_sequence_length,
        rnn_units, num_rnn_layers, num_classes, dropout_rate,
        recurrent_dropout,
    )

    # --- Functional API graph ---
    inputs = Input(
        shape=(max_sequence_length,),
        dtype="int32",
        name="input_sequences",
    )

    x = _build_embedding_layer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        input_length=max_sequence_length,
    )(inputs)

    x = _build_rnn_block(
        x,
        units=rnn_units,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        num_layers=num_rnn_layers,
    )

    outputs = _build_classification_head(
        x,
        dense_units=rnn_units,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )

    model = Model(inputs=inputs, outputs=outputs, name="SimpleRNN_Classifier")
    logger.info("Model graph constructed successfully.")

    if compile_model:
        _compile_model(model, learning_rate=learning_rate)

    return model


# ============================================================================
# 6. SELF-TEST
# ============================================================================

def _separator(title: str, width: int = 60) -> None:
    """Print a formatted section header (self-test only)."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


if __name__ == "__main__":
    setup_logging()

    _separator("TicketVision-AI  —  Simple RNN Model Self-Test")

    # ---- Build the model ----
    try:
        model = build_rnn_model()
    except (ValueError, RuntimeError) as err:
        logger.critical("Model build FAILED: %s", err)
        sys.exit(1)

    # ---- Architecture summary ----
    _separator("Model Summary")
    model.summary(line_length=90)

    # ---- Parameter counts ----
    _separator("Parameter Counts")
    total_params: int = model.count_params()
    trainable_params: int = sum(
        int(tf.keras.backend.count_params(w)) for w in model.trainable_weights
    )
    non_trainable_params: int = total_params - trainable_params

    print(f"  Total parameters       : {total_params:>12,}")
    print(f"  Trainable parameters   : {trainable_params:>12,}")
    print(f"  Non-trainable params   : {non_trainable_params:>12,}")

    # ---- Hyperparameters used ----
    _separator("Hyperparameters (from config.py)")
    print(f"  VOCAB_SIZE             : {VOCAB_SIZE:,}")
    print(f"  EMBEDDING_DIM          : {EMBEDDING_DIM}")
    print(f"  MAX_SEQUENCE_LENGTH    : {MAX_SEQUENCE_LENGTH}")
    print(f"  RNN_UNITS              : {RNN_UNITS}")
    print(f"  NUM_CLASSES            : {NUM_CLASSES}")
    print(f"  DROPOUT_RATE           : {DROPOUT_RATE}")
    print(f"  RECURRENT_DROPOUT      : {RECURRENT_DROPOUT}")
    print(f"  LEARNING_RATE          : {LEARNING_RATE}")

    # ---- I/O shape verification ----
    _separator("Input / Output Shape Verification")
    import numpy as np

    dummy_input = np.zeros((1, MAX_SEQUENCE_LENGTH), dtype=np.int32)
    dummy_output = model.predict(dummy_input, verbose=0)

    print(f"  Input shape            : {dummy_input.shape}")
    print(f"  Output shape           : {dummy_output.shape}")
    print(f"  Output dtype           : {dummy_output.dtype}")
    print(f"  Sum of probabilities   : {dummy_output.sum():.6f}  (expected ≈ 1.0)")
    print(f"  Predicted class        : {np.argmax(dummy_output, axis=-1)[0]}")

    assert dummy_output.shape == (1, NUM_CLASSES), (
        f"Expected output shape (1, {NUM_CLASSES}), got {dummy_output.shape}"
    )
    assert abs(dummy_output.sum() - 1.0) < 1e-5, (
        f"Softmax probabilities should sum to ≈1.0, got {dummy_output.sum()}"
    )

    # ---- Compilation verification ----
    _separator("Compilation Verification")
    print(f"  Optimizer              : {model.optimizer.__class__.__name__}")
    print(f"  Learning rate          : {model.optimizer.learning_rate.numpy():.1e}")
    print(f"  Loss function          : {model.loss}")
    print(f"  Metrics                : {[m.name for m in model.metrics]}")

    # ---- Layer-by-layer breakdown ----
    _separator("Layer-by-Layer Breakdown")
    print(f"  {'Layer Name':<25s}  {'Type':<20s}  {'Output Shape':<22s}  {'Params':>10s}")
    print(f"  {'-' * 25}  {'-' * 20}  {'-' * 22}  {'-' * 10}")
    for layer in model.layers:
        output_shape = str(layer.output.shape) if hasattr(layer, "output") else "N/A"
        params = layer.count_params()
        print(
            f"  {layer.name:<25s}  {layer.__class__.__name__:<20s}  "
            f"{output_shape:<22s}  {params:>10,}"
        )

    # ---- Save-path reminder ----
    _separator("Model Save Path")
    from config import MODEL_SAVE_PATHS
    rnn_path = MODEL_SAVE_PATHS.get("rnn", "NOT CONFIGURED")
    print(f"  Configured path : {rnn_path}")

    _separator("Self-Test Complete")
    print("  Simple RNN model built and verified successfully.\n")
