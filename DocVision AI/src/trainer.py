"""DocVision-AI Keras Model Training Module.

This module manages the complete model training lifecycle, handling dataset retrieval,
model instantiation, callback configuration (EarlyStopping, ModelCheckpoint,
ReduceLROnPlateau, CSVLogger, TensorBoard), model training, final model serialization,
and comprehensive model verification and reporting (history logs, JSON metadata,
confusion matrices, accuracy/loss curves).
"""

import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------------------
# 0. DESERIALIZATION MONKEYPATCH FOR KERAS 3 / TF 2.16+
# ------------------------------------------------------------------------------
# Pop quantization_config argument during Dense initialization to prevent
# "TypeError: Unrecognized keyword arguments passed to Dense" when loading models.
original_dense_init = tf.keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    original_dense_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = patched_dense_init


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
    METADATA_PATH,
    HISTORY_PATH,
    CONFUSION_MATRIX_PATH,
    CLASSIFICATION_REPORT_PATH,
    ACCURACY_CURVE_PATH,
    LOSS_CURVE_PATH,
    LOGS_DIR,
    CLASSES,
)

# Import dataset loader and model builder/compiler
from src.dataset import get_datasets
from src.data_loader import load_dataset
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
    ReduceLROnPlateau, CSVLogger, and TensorBoard logger.

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
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_filepath.parent / "tensorboard"),
            histogram_freq=1,
            update_freq="epoch",
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


def evaluate_and_log_metrics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    dataset_name: str = "Validation",
) -> Tuple[float, float]:
    """Evaluates model performance and prints loss, accuracy, confusion matrix, and F1 metrics.

    Args:
        model (tf.keras.Model): The model to evaluate.
        dataset (tf.data.Dataset): The dataset to evaluate.
        dataset_name (str): Label describing the dataset (e.g. "Validation").

    Returns:
        Tuple[float, float]: Calculated (loss, accuracy).
    """
    logger.info(f"Evaluating model performance on {dataset_name} set...")

    # 1. Evaluate general loss and accuracy
    loss, accuracy = model.evaluate(dataset, verbose=0)
    logger.info(f"[{dataset_name}] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return float(loss), float(accuracy)


def save_reports_and_plots(
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    history: tf.keras.callbacks.History,
) -> None:
    """Plots and writes metrics artifacts: curves, reports, and confusion matrix.

    Args:
        model (tf.keras.Model): The reloaded best model for report validation.
        val_dataset (tf.data.Dataset): Validation dataset.
        history (tf.keras.callbacks.History): Training history object.
    """
    logger.info("Generating and saving reports and visual plots...")
    
    # 1. Get predictions for validation set
    y_true = []
    y_pred = []
    for images, labels in val_dataset:
        preds = model.predict(images, verbose=0)
        pred_indices = np.argmax(preds, axis=-1)
        y_true.extend(labels.numpy())
        y_pred.extend(pred_indices)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 2. Save Confusion Matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Validation Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(CONFUSION_MATRIX_PATH), dpi=150)
    plt.close()
    logger.info(f"Confusion Matrix saved to: {CONFUSION_MATRIX_PATH}")

    # 3. Save Textual Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write(report)
    logger.info(f"Classification report saved to: {CLASSIFICATION_REPORT_PATH}")

    # 4. Save Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Model Training and Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(ACCURACY_CURVE_PATH), dpi=150)
    plt.close()
    logger.info(f"Accuracy curve saved to: {ACCURACY_CURVE_PATH}")

    # 5. Save Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Training and Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(LOSS_CURVE_PATH), dpi=150)
    plt.close()
    logger.info(f"Loss curve saved to: {LOSS_CURVE_PATH}")


def run_training_pipeline(
    epochs: Optional[int] = None,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Orchestrates the entire training pipeline.

    Loads data, builds and compiles the model, configures callbacks, trains the
    model, evaluates performance, and saves/verifies the serialized checkpoints.

    Args:
        epochs (Optional[int]): Number of epochs to train. If None, uses EPOCHS from config.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: The trained model and history.
    """
    logger.info("Initializing DocVision-AI Training Pipeline...")

    checkpoint_filepath = BEST_MODEL_PATH
    final_filepath = FINAL_MODEL_PATH
    log_filepath = LOGS_DIR / "training_log.csv"
    
    target_epochs = epochs if epochs is not None else EPOCHS

    try:
        # 1. Get datasets
        logger.info("Loading training, validation, and test datasets...")
        train_ds, val_ds, test_ds = get_datasets(batch_size=BATCH_SIZE)

        # 2. Build and compile model
        logger.info("Building and compiling model...")
        model = build_model()
        print_model_summary(model)
        compiled_model = compile_model(model, learning_rate=LEARNING_RATE)

        # 3. Get callbacks
        callbacks = get_callbacks(checkpoint_filepath, log_filepath)

        # 4. Train model
        trained_model, history = train_model(
            model=compiled_model,
            train_dataset=train_ds,
            validation_dataset=val_ds,
            epochs=target_epochs,
            callbacks=callbacks,
        )

        # 5. Save final model
        logger.info(f"Saving final trained model to: {final_filepath}")
        final_filepath.parent.mkdir(parents=True, exist_ok=True)
        trained_model.save(str(final_filepath))
        logger.info("Final model saved successfully.")

        # 6. Save history log CSV
        logger.info(f"Saving training history to: {HISTORY_PATH}")
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(HISTORY_PATH, index=True, index_label="epoch")

        # 7. Model Verification Process
        logger.info("=" * 60)
        logger.info("STARTING POST-TRAINING MODEL VERIFICATION")
        logger.info("=" * 60)
        
        # Load best checkpoint model from disk directly
        logger.info(f"Loading best checkpoint model directly from disk: {checkpoint_filepath}")
        reloaded_model = tf.keras.models.load_model(str(checkpoint_filepath))
        
        # Evaluate both models on the test dataset
        loss_orig, acc_orig = evaluate_and_log_metrics(trained_model, test_ds, dataset_name="In-Memory Model")
        loss_reloaded, acc_reloaded = evaluate_and_log_metrics(reloaded_model, test_ds, dataset_name="Reloaded Model")
        
        # Compare prediction distributions using np.allclose
        logger.info("Extracting prediction tensors for similarity comparison...")
        preds_orig = trained_model.predict(test_ds, verbose=0)
        preds_reloaded = reloaded_model.predict(test_ds, verbose=0)
        match_status = bool(np.allclose(preds_orig, preds_reloaded, atol=1e-5))
        
        integrity_status = "VERIFIED" if match_status else "FAILED"
        
        # Print Verification Summary
        print("\n" + "=" * 60)
        print("MODEL VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Original Model Test Accuracy:  {acc_orig*100:.2f}% (Loss: {loss_orig:.4f})")
        print(f"Reloaded Model Test Accuracy:  {acc_reloaded*100:.2f}% (Loss: {loss_reloaded:.4f})")
        print(f"Prediction Outputs Match:      {match_status}")
        print(f"Model Integrity Status:        {integrity_status}")
        print("=" * 60 + "\n")
        
        logger.info(f"Model verification finished. Status: {integrity_status}")

        # 8. Save Reports, Confusion Matrix, and Curves
        save_reports_and_plots(reloaded_model, val_ds, history)

        # 9. Save Metadata JSON file
        logger.info(f"Saving training metadata to: {METADATA_PATH}")
        metadata = {
            "model_name": trained_model.name,
            "dataset_size": len(load_dataset()),
            "epochs": target_epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "train_accuracy": float(history.history["accuracy"][-1]),
            "validation_accuracy": float(history.history["val_accuracy"][-1]),
            "test_accuracy": acc_reloaded,
            "train_loss": float(history.history["loss"][-1]),
            "validation_loss": float(history.history["val_loss"][-1]),
            "test_loss": loss_reloaded,
            "timestamp": datetime.datetime.now().isoformat(),
            "class_names": CLASSES,
            "integrity_status": integrity_status,
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info("Metadata saved successfully.")

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
