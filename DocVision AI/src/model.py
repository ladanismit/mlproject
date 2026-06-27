"""DocVision-AI Keras Model Definition and Compilation Module.

This module provides reusable functions to build and compile the convolutional
neural network (CNN) model for document classification (Resume vs. Invoice).
It imports hyperparameters and configurations from `config.py` to ensure
consistent input shapes, class counts, and optimization parameters.

This module follows the Single Responsibility Principle, focusing solely on
model definition, compilation, and utility output (summary printing and diagram plotting).
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models

# Add project root to sys.path to enable running the script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import configuration constants
from src.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
    NUM_CLASSES,
    LEARNING_RATE,
    OUTPUTS_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_model(
    input_shape: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
    num_classes: int = NUM_CLASSES,
) -> models.Sequential:
    """Builds a custom CNN model using the Keras Sequential API.

    The architecture consists of:
    - Input Layer (input_shape)
    - Conv Block 1: Conv2D (32 filters, 3x3, same) -> BatchNormalization -> ReLU -> MaxPooling2D
    - Conv Block 2: Conv2D (64 filters, 3x3, same) -> BatchNormalization -> ReLU -> MaxPooling2D
    - Conv Block 3: Conv2D (128 filters, 3x3, same) -> BatchNormalization -> ReLU -> MaxPooling2D
    - GlobalAveragePooling2D
    - Dense (128 units, relu)
    - Dropout (0.5)
    - Output Dense (num_classes, softmax)

    Args:
        input_shape (Tuple[int, int, int]): The shape of input image tensors.
            Defaults to (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).
        num_classes (int): The number of target classification categories.
            Defaults to NUM_CLASSES.

    Returns:
        models.Sequential: The constructed, uncompiled Keras Sequential model.

    Raises:
        ValueError: If input_shape is not a 3-tuple or num_classes is not positive.
    """
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must have exactly 3 dimensions (H, W, C), got {input_shape}")
    if num_classes <= 0:
        raise ValueError(f"Number of classes must be a positive integer, got {num_classes}")

    logger.info(
        f"Building custom CNN model. Input shape: {input_shape}, Number of classes: {num_classes}"
    )

    try:
        model = models.Sequential([
            # Input Layer
            layers.Input(shape=input_shape, name="input_layer"),

            # Conv Block 1
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", name="conv_block1_conv"),
            layers.LayerNormalization(axis=-1, name="conv_block1_ln"),
            layers.ReLU(name="conv_block1_relu"),
            layers.MaxPooling2D(pool_size=(2, 2), name="conv_block1_pool"),

            # Conv Block 2
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="conv_block2_conv"),
            layers.LayerNormalization(axis=-1, name="conv_block2_ln"),
            layers.ReLU(name="conv_block2_relu"),
            layers.MaxPooling2D(pool_size=(2, 2), name="conv_block2_pool"),

            # Conv Block 3
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", name="conv_block3_conv"),
            layers.LayerNormalization(axis=-1, name="conv_block3_ln"),
            layers.ReLU(name="conv_block3_relu"),
            layers.MaxPooling2D(pool_size=(2, 2), name="conv_block3_pool"),

            # Classifier Head
            layers.GlobalAveragePooling2D(name="global_avg_pool"),
            layers.Dense(units=128, activation="relu", name="dense_fc"),
            layers.Dropout(rate=0.5, name="dropout_fc"),
            layers.Dense(units=num_classes, activation="softmax", name="output_layer"),
        ], name="docvision_cnn")

        logger.info("Model architecture constructed successfully.")
        return model
    except Exception as e:
        logger.error(f"Error during model architecture construction: {e}")
        raise


def compile_model(
    model: models.Sequential,
    learning_rate: float = LEARNING_RATE,
) -> models.Sequential:
    """Compiles the model with the Adam optimizer, Sparse Categorical Crossentropy loss, and Accuracy metric.

    Args:
        model (models.Sequential): The Keras model to compile.
        learning_rate (float): The learning rate for the Adam optimizer.
            Defaults to LEARNING_RATE from config.

    Returns:
        models.Sequential: The compiled Keras model.

    Raises:
        ValueError: If model is None or learning_rate is not positive.
    """
    if model is None:
        raise ValueError("Model parameter cannot be None.")
    if learning_rate <= 0.0:
        raise ValueError(f"Learning rate must be a positive float, got {learning_rate}")

    logger.info(
        f"Compiling model with Adam optimizer (learning_rate={learning_rate}) "
        "and SparseCategoricalCrossentropy loss."
    )

    try:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        logger.info("Model compiled successfully.")
        return model
    except Exception as e:
        logger.error(f"Error during model compilation: {e}")
        raise


def print_model_summary(model: models.Sequential) -> None:
    """Prints a summary of the model representation to the console.

    Args:
        model (models.Sequential): The Keras model to summarize.

    Raises:
        ValueError: If model is None.
    """
    if model is None:
        raise ValueError("Model parameter cannot be None.")

    logger.info("Generating model summary:")
    model.summary()


def save_model_diagram(
    model: models.Sequential,
    output_path: Optional[Path] = None,
) -> bool:
    """Attempts to save the model architecture diagram to a file.

    Requires pydot and graphviz. Gracefully logs a warning if they are not installed.

    Args:
        model (models.Sequential): The Keras model to plot.
        output_path (Optional[Path]): File path to save the diagram image.
            Defaults to saving as 'model_architecture.png' inside the outputs directory.

    Returns:
        bool: True if the diagram was saved successfully, False otherwise.
    """
    if model is None:
        logger.error("Model parameter cannot be None for diagram generation.")
        return False

    if output_path is None:
        output_path = OUTPUTS_DIR / "model_architecture.png"
    else:
        output_path = Path(output_path)

    logger.info(f"Attempting to save model diagram to {output_path}")

    try:
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tf.keras.utils.plot_model(
            model,
            to_file=str(output_path),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=96,
        )
        
        # Check if the file was actually written to disk, as plot_model might fail
        # silently and output error messages to stdout/stderr in some Keras versions.
        if output_path.exists():
            logger.info(f"Model diagram successfully saved to: {output_path}")
            return True
        else:
            logger.warning(
                "Could not save model diagram: diagram file was not created. "
                "This typically occurs when 'pydot' or 'graphviz' dependencies are missing."
            )
            return False
    except ImportError as e:
        logger.warning(
            f"Could not save model diagram: dependencies 'pydot' or 'graphviz' are missing. "
            "Please install them if you want to generate the architecture diagram. "
            f"Error: {e}"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to save model diagram due to an unexpected error: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting model.py self-test...")
    try:
        # 1. Build model
        model = build_model()

        # 2. Print summary
        print_model_summary(model)

        # 3. Compile model
        compiled_model = compile_model(model)

        # 4. Save diagram
        diagram_saved = save_model_diagram(compiled_model)

        logger.info(f"Self-test completed successfully. Diagram saved: {diagram_saved}")
    except Exception as err:
        logger.error(f"Self-test failed: {err}")
        sys.exit(1)
