"""
trainer.py — Model-Agnostic Training Engine for TicketVision-AI
===============================================================

Trains **any** compiled TensorFlow / Keras model (Simple RNN, LSTM,
BiLSTM + Attention, Transformer) on the datasets produced by
``dataset.py``.

Responsibilities
----------------
* Configure callbacks: ``ModelCheckpoint``, ``EarlyStopping``,
  ``ReduceLROnPlateau``, ``CSVLogger``, ``TensorBoard``.
* Train the model for ``EPOCHS`` epochs.
* Save best *and* final model weights.
* Persist training history as JSON.
* Generate and save accuracy / loss plots.

This module is **training-only** — no model creation or evaluation
logic lives here.

Usage
-----
>>> from src.trainer import train_model
>>> from src.models.rnn_model import build_rnn_model
>>> model = build_rnn_model()
>>> trained_model, history = train_model(model, model_name="rnn")

Author : TicketVision-AI Team
Created: 2026-07-04
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import tensorflow as tf                     # noqa: E402
from tensorflow.keras import Model          # noqa: E402
from tensorflow.keras.callbacks import (    # noqa: E402
    CSVLogger,
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

# ---------------------------------------------------------------------------
# Ensure project root is importable (same pattern as data_loader.py).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    EXPERIMENTS_DIR,
    MODEL_HISTORY_PATHS,
    MODEL_SAVE_PATHS,
    PLOTS_DIR,
    SEED,
    setup_logging,
)
from src.dataset import prepare_datasets  # noqa: E402

# Module-level logger.
logger: logging.Logger = logging.getLogger("ticketvision.trainer")

# Mapping from model name → experiment sub-directory.
_EXP_DIRS: Dict[str, Path] = {
    "rnn":         EXPERIMENTS_DIR / "rnn",
    "lstm":        EXPERIMENTS_DIR / "lstm",
    "attention":   EXPERIMENTS_DIR / "attention",
    "transformer": EXPERIMENTS_DIR / "transformer",
}


# ============================================================================
# 1. CALLBACK CONFIGURATION
# ============================================================================

def _build_callbacks(
    model_name: str,
    *,
    checkpoint_path: Path,
    csv_log_path: Path,
    tensorboard_dir: Path,
    patience_early_stop: int = 5,
    patience_reduce_lr: int = 3,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> List[Callback]:
    """Assemble the training callback stack.

    Parameters
    ----------
    model_name : str
        Architecture identifier (``"rnn"``, ``"lstm"``, etc.).
    checkpoint_path : Path
        Filepath for ``ModelCheckpoint`` (best weights).
    csv_log_path : Path
        Filepath for epoch-level CSV log.
    tensorboard_dir : Path
        Directory for TensorBoard event files.
    patience_early_stop : int
        Epochs without val-loss improvement before stopping.
    patience_reduce_lr : int
        Epochs without val-loss improvement before reducing LR.
    reduce_lr_factor : float
        Multiplicative factor for LR reduction.
    min_lr : float
        Lower bound on learning rate.

    Returns
    -------
    list[Callback]
        Ordered list of Keras callbacks.
    """
    # Ensure parent directories exist.
    for p in (checkpoint_path, csv_log_path, tensorboard_dir):
        (p if p.is_dir() else p.parent).mkdir(parents=True, exist_ok=True)

    callbacks: List[Callback] = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=patience_reduce_lr,
            min_lr=min_lr,
            verbose=1,
        ),
        CSVLogger(
            filename=str(csv_log_path),
            append=False,
        ),
        TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
    ]

    logger.info("Callbacks configured for '%s':", model_name)
    for cb in callbacks:
        logger.info("  • %s", cb.__class__.__name__)

    return callbacks


# ============================================================================
# 2. TRAINING HISTORY PERSISTENCE
# ============================================================================

def _save_history(
    history: Dict[str, List[float]],
    save_path: Path,
) -> Path:
    """Serialise the training history dict to a JSON file.

    Parameters
    ----------
    history : dict
        Keras ``History.history`` dict (metric name → list of values).
    save_path : Path
        Destination JSON path.

    Returns
    -------
    Path
        The path the file was written to.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python for JSON serialisation.
    serialisable: Dict[str, List[float]] = {
        key: [float(v) for v in values]
        for key, values in history.items()
    }
    save_path.write_text(
        json.dumps(serialisable, indent=2),
        encoding="utf-8",
    )
    logger.info("Training history saved to: %s", save_path)
    return save_path


# ============================================================================
# 3. TRAINING PLOTS
# ============================================================================

def _plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    *,
    plots_dir: Path = PLOTS_DIR,
) -> Path:
    """Generate and save accuracy + loss curves side by side.

    Parameters
    ----------
    history : dict
        Keras ``History.history`` dict.
    model_name : str
        Architecture name, used in the plot title and filename.
    plots_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history["loss"]) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Accuracy ----
    ax_acc.plot(epochs_range, history["accuracy"],
                label="Train Accuracy", linewidth=2, marker="o", markersize=4)
    ax_acc.plot(epochs_range, history["val_accuracy"],
                label="Val Accuracy", linewidth=2, marker="s", markersize=4)
    ax_acc.set_title(f"{model_name.upper()} — Accuracy", fontsize=13, fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    # ---- Loss ----
    ax_loss.plot(epochs_range, history["loss"],
                 label="Train Loss", linewidth=2, marker="o", markersize=4)
    ax_loss.plot(epochs_range, history["val_loss"],
                 label="Val Loss", linewidth=2, marker="s", markersize=4)
    ax_loss.set_title(f"{model_name.upper()} — Loss", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    fig.suptitle(
        f"Training Curves — {model_name.upper()}",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    save_path: Path = plots_dir / f"{model_name}_training_curves.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Training plots saved to: %s", save_path)
    return save_path


# ============================================================================
# 4. PUBLIC API
# ============================================================================

def train_model(
    model: Model,
    model_name: str,
    *,
    train_ds: Optional[tf.data.Dataset] = None,
    val_ds: Optional[tf.data.Dataset] = None,
    epochs: int = EPOCHS,
    save: bool = True,
    patience_early_stop: int = 5,
    patience_reduce_lr: int = 3,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> Tuple[Model, Dict[str, List[float]]]:
    """Train a compiled Keras model end-to-end.

    Parameters
    ----------
    model : Model
        A **compiled** Keras model.
    model_name : str
        Architecture key (``"rnn"``, ``"lstm"``, ``"attention"``,
        ``"transformer"``).  Determines save paths and plot labels.
    train_ds : tf.data.Dataset, optional
        Training dataset.  If ``None``, calls ``prepare_datasets()``.
    val_ds : tf.data.Dataset, optional
        Validation dataset.  If ``None``, calls ``prepare_datasets()``.
    epochs : int
        Maximum number of training epochs.
    save : bool
        Persist models, history, and plots to disk.
    patience_early_stop : int
        ``EarlyStopping`` patience.
    patience_reduce_lr : int
        ``ReduceLROnPlateau`` patience.
    reduce_lr_factor : float
        Multiplicative LR reduction factor.
    min_lr : float
        Minimum learning rate floor.

    Returns
    -------
    tuple[Model, dict]
        ``(trained_model, history_dict)``

    Raises
    ------
    ValueError
        If ``model_name`` is not a recognised architecture key.
    RuntimeError
        If training fails due to an internal TensorFlow error.
    """
    # ------------------------------------------------------------------
    # 0. Validate model_name.
    # ------------------------------------------------------------------
    if model_name not in MODEL_SAVE_PATHS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Expected one of: {sorted(MODEL_SAVE_PATHS.keys())}"
        )

    logger.info("=" * 60)
    logger.info("  Training: %s", model_name.upper())
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Obtain datasets if not provided.
    # ------------------------------------------------------------------
    if train_ds is None or val_ds is None:
        logger.info("Datasets not provided — running prepare_datasets().")
        train_ds, val_ds, _, _, info = prepare_datasets()
        logger.info(
            "Datasets loaded — Train: %s | Val: %s | Classes: %d",
            f"{info['train_size']:,}", f"{info['val_size']:,}",
            info["num_classes"],
        )

    # ------------------------------------------------------------------
    # 2. Resolve paths.
    # ------------------------------------------------------------------
    best_model_path: Path = MODEL_SAVE_PATHS[model_name]
    final_model_path: Path = best_model_path.with_stem(
        best_model_path.stem + "_final"
    )
    csv_log_path: Path = MODEL_HISTORY_PATHS[model_name]
    history_json_path: Path = best_model_path.with_suffix(".history.json")
    exp_dir: Path = _EXP_DIRS.get(model_name, EXPERIMENTS_DIR / model_name)
    tensorboard_dir: Path = exp_dir / "tensorboard"

    logger.info("Best model will be saved to  : %s", best_model_path)
    logger.info("Final model will be saved to : %s", final_model_path)
    logger.info("CSV log                      : %s", csv_log_path)
    logger.info("History JSON                 : %s", history_json_path)
    logger.info("TensorBoard logs             : %s", tensorboard_dir)

    # ------------------------------------------------------------------
    # 3. Build callbacks.
    # ------------------------------------------------------------------
    callbacks = _build_callbacks(
        model_name,
        checkpoint_path=best_model_path,
        csv_log_path=csv_log_path,
        tensorboard_dir=tensorboard_dir,
        patience_early_stop=patience_early_stop,
        patience_reduce_lr=patience_reduce_lr,
        reduce_lr_factor=reduce_lr_factor,
        min_lr=min_lr,
    )

    # ------------------------------------------------------------------
    # 4. Train.
    # ------------------------------------------------------------------
    logger.info(
        "Starting training — epochs=%d, seed=%d.",
        epochs, SEED,
    )
    t_start = time.perf_counter()

    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    except Exception as exc:
        logger.error("Training FAILED: %s", exc)
        raise RuntimeError(
            f"Training failed for model '{model_name}': {exc}"
        ) from exc

    elapsed = time.perf_counter() - t_start
    n_epochs_completed: int = len(history.history["loss"])
    best_val_loss: float = min(history.history["val_loss"])
    best_epoch: int = int(np.argmin(history.history["val_loss"])) + 1

    logger.info("Training complete in %.1fs.", elapsed)
    logger.info(
        "  Epochs completed : %d / %d", n_epochs_completed, epochs,
    )
    logger.info("  Best val_loss    : %.5f  (epoch %d)", best_val_loss, best_epoch)
    logger.info(
        "  Final train_acc  : %.4f  |  val_acc : %.4f",
        history.history["accuracy"][-1],
        history.history["val_accuracy"][-1],
    )

    history_dict: Dict[str, List[float]] = history.history

    # ------------------------------------------------------------------
    # 5. Save artefacts.
    # ------------------------------------------------------------------
    if save:
        # Best model is already saved by ModelCheckpoint.
        # Save the final-epoch model as well.
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(final_model_path))
        logger.info("Final model saved to: %s", final_model_path)

        _save_history(history_dict, history_json_path)
        _plot_training_curves(history_dict, model_name)

    logger.info("=" * 60)
    logger.info("  Training pipeline complete: %s", model_name.upper())
    logger.info("=" * 60)

    return model, history_dict


# ============================================================================
# 5. SELF-TEST
# ============================================================================

def _separator(title: str, width: int = 60) -> None:
    """Print a formatted section header (self-test only)."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


if __name__ == "__main__":
    setup_logging()

    _separator("TicketVision-AI  —  Trainer Self-Test")
    print("  Running a quick 1-epoch smoke test with the Simple RNN model.\n")

    # ---- 1. Prepare data ----
    _separator("Step 1 / 5 — Preparing Datasets")
    try:
        train_ds, val_ds, test_ds, tokenizer, info = prepare_datasets()
    except (FileNotFoundError, KeyError, ValueError) as err:
        logger.critical("Dataset preparation FAILED: %s", err)
        sys.exit(1)

    print(f"  Train batches : {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"  Val batches   : {tf.data.experimental.cardinality(val_ds).numpy()}")

    # ---- 2. Build the RNN model ----
    _separator("Step 2 / 5 — Building Simple RNN Model")
    from src.models.rnn_model import build_rnn_model  # noqa: E402

    model = build_rnn_model()
    model.summary(line_length=90)

    # ---- 3. Take a small subset for fast smoke test ----
    _separator("Step 3 / 5 — Creating Small Subset (5 batches)")
    small_train = train_ds.take(5)
    small_val = val_ds.take(2)

    # Confirm shapes.
    for seq_b, lbl_b in small_train.take(1):
        print(f"  Sample batch — sequences: {seq_b.shape}, labels: {lbl_b.shape}")

    # ---- 4. Train for 1 epoch ----
    _separator("Step 4 / 5 — Training (1 Epoch)")
    trained_model, history = train_model(
        model,
        model_name="rnn",
        train_ds=small_train,
        val_ds=small_val,
        epochs=1,
        save=True,
    )

    # ---- 5. Verify artefacts ----
    _separator("Step 5 / 5 — Verifying Output Artefacts")

    artefacts = {
        "Best Model": MODEL_SAVE_PATHS["rnn"],
        "Final Model": MODEL_SAVE_PATHS["rnn"].with_stem(
            MODEL_SAVE_PATHS["rnn"].stem + "_final"
        ),
        "CSV Log": MODEL_HISTORY_PATHS["rnn"],
        "History JSON": MODEL_SAVE_PATHS["rnn"].with_suffix(".history.json"),
        "Training Plot": PLOTS_DIR / "rnn_training_curves.png",
        "TensorBoard Dir": _EXP_DIRS["rnn"] / "tensorboard",
    }

    all_ok = True
    for label, path in artefacts.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        if path.is_file() and exists:
            size_kb = path.stat().st_size / 1024
            print(f"  [{status}]  {label:<16s} : {path}  ({size_kb:,.1f} KB)")
        elif path.is_dir() and exists:
            n_files = sum(1 for _ in path.rglob("*") if _.is_file())
            print(f"  [{status}]  {label:<16s} : {path}  ({n_files} files)")
        else:
            print(f"  [{status}]  {label:<16s} : {path}")
            all_ok = False

    # ---- Training history summary ----
    _separator("Training History (1 Epoch)")
    for metric, values in history.items():
        print(f"  {metric:<20s} : {values[-1]:.6f}")

    # ---- Final status ----
    _separator("Self-Test Result")
    if all_ok:
        print("  All artefacts generated successfully.")
        print("  Trainer pipeline verified. ✓\n")
    else:
        print("  WARNING: Some artefacts are missing!")
        print("  Review the log output above.\n")
        sys.exit(1)
