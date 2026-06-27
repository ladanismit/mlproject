"""DocVision-AI Keras Model Training Module.

This module manages the complete model training lifecycle, handling dataset retrieval,
model instantiation, callback configuration (EarlyStopping, ModelCheckpoint,
ReduceLROnPlateau, CSVLogger), model training, and final model serialization.

This module follows the Single Responsibility Principle, focusing solely on
the orchestration of training and checkpointing.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf

# Add project root to sys.path to enable running the script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import configuration constants
from src.config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    LOGS_DIR,
)

# Import dataset loader and model builder/compiler
from src.dataset import get_datasets
from src.model import build_model, compile_model, print_model_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_callbacks(
    checkpoint_filepath: Path,
    log_filepath: Path,
) -> list:
    """Configures and returns production-ready TensorFlow/Keras callbacks.

    Includes EarlyStopping, ModelCheckpoint (saving the best model),
    ReduceLROnPlateau, and CSVLogger.

    Args:
        checkpoint_filepath (Path): Filepath where the best model checkpoint will be saved.
        log_filepath (Path): Filepath where the training log CSV will be saved.

    Returns:
        list: A list of configured tf.keras.callbacks.Callback objects.
    """
    logger.info(
        f"Configuring callbacks. Checkpoint path: {checkpoint_filepath}, "
        f"Log path: {log_filepath}"
    )

    # Ensure parent directories exist
    checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)
    log_filepath.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(log_filepath),
            separator=",",
            append=False,
        ),
    ]
    return callbacks


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    epochs: int = EPOCHS,
    callbacks: list = None,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Trains the model using model.fit() with the provided datasets and callbacks.

    Args:
        model (tf.keras.Model): The compiled Keras model.
        train_dataset (tf.data.Dataset): The training dataset.
        validation_dataset (tf.data.Dataset): The validation dataset.
        epochs (int): Number of epochs to train. Defaults to EPOCHS from config.
        callbacks (list): List of Keras callbacks to apply.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: The trained model and history.

    Raises:
        ValueError: If model, train_dataset, or validation_dataset is None.
        RuntimeError: If training fails.
    """
    if model is None:
        raise ValueError("Model parameter cannot be None.")
    if train_dataset is None:
        raise ValueError("Train dataset cannot be None.")
    if validation_dataset is None:
        raise ValueError("Validation dataset cannot be None.")

    logger.info(f"Starting model training for {epochs} epochs...")
    try:
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("Model training completed successfully.")
        return model, history
    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        raise RuntimeError(f"Training failed: {e}") from e


def run_training_pipeline(
    epochs: Optional[int] = None,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Orchestrates the entire training pipeline.

    Loads data, builds and compiles the model, configures callbacks, trains the
    model, saves the final model, and returns results.

    Args:
        epochs (Optional[int]): Number of epochs to train. If None, uses EPOCHS from config.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: The trained model and history.
    """
    logger.info("Initializing DocVision-AI Training Pipeline...")

    # 1. Resolve paths (use .keras suffix if config path has .pth PyTorch extension)
    checkpoint_filepath = BEST_MODEL_PATH
    if checkpoint_filepath.suffix == ".pth":
        checkpoint_filepath = checkpoint_filepath.with_suffix(".keras")

    final_filepath = FINAL_MODEL_PATH
    if final_filepath.suffix == ".pth":
        final_filepath = final_filepath.with_suffix(".keras")

    log_filepath = LOGS_DIR / "training_log.csv"
    
    target_epochs = epochs if epochs is not None else EPOCHS

    try:
        # 2. Get datasets
        logger.info("Loading training and validation datasets...")
        train_ds, val_ds, _ = get_datasets(batch_size=BATCH_SIZE)

        # 3. Build and compile model
        logger.info("Building and compiling model...")
        model = build_model()
        print_model_summary(model)
        compiled_model = compile_model(model, learning_rate=LEARNING_RATE)

        # 4. Get callbacks
        callbacks = get_callbacks(checkpoint_filepath, log_filepath)

        # 5. Train model
        trained_model, history = train_model(
            model=compiled_model,
            train_dataset=train_ds,
            validation_dataset=val_ds,
            epochs=target_epochs,
            callbacks=callbacks,
        )

        # 6. Save final model
        logger.info(f"Saving final trained model to: {final_filepath}")
        final_filepath.parent.mkdir(parents=True, exist_ok=True)
        trained_model.save(str(final_filepath))
        logger.info("Final model saved successfully.")

        return trained_model, history

    except Exception as e:
        logger.error(f"Training pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting trainer.py self-test...")
    try:
        # Run pipeline with epochs=1 for testing
        trained_model, history = run_training_pipeline(epochs=1)
        logger.info("trainer.py self-test completed successfully.")
    except Exception as err:
        logger.error(f"trainer.py self-test failed: {err}")
        sys.exit(1)
