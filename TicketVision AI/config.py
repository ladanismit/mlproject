"""
config.py — Single Source of Truth for TicketVision-AI
======================================================

Central configuration module for the TicketVision-AI project.
Manages every tuneable knob — directory layout, dataset schema, NLP
preprocessing, training hyper-parameters, per-model hyper-parameters,
model-saving paths, and logging — in one auditable location.

Design Principles
-----------------
* **Immutable at runtime** — all values are module-level constants.
* **pathlib-first** — every filesystem path is a ``pathlib.Path``.
* **Auto-provisioning** — required directories are created on first import.
* **Extensible** — adding a fifth model requires only a new entry in
  ``MODEL_SAVE_PATHS`` and its hyper-parameter block.

Usage
-----
>>> from config import *          # quick scripts / notebooks
>>> from config import SEED, LR   # explicit imports (preferred)

Author : TicketVision-AI Team
Created: 2026-07-03
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Final, List

# ============================================================================
# 1. PROJECT ROOT & DIRECTORY LAYOUT
# ============================================================================

# Resolve project root relative to this file so it works regardless of cwd.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

# --- Data directories ---
DATA_DIR: Final[Path]           = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path]      = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

# --- Top-level model directory (trained weights & checkpoints) ---
MODELS_DIR: Final[Path]        = PROJECT_ROOT / "models"
MODELS_RNN_DIR: Final[Path]    = MODELS_DIR / "rnn"
MODELS_LSTM_DIR: Final[Path]   = MODELS_DIR / "lstm"
MODELS_ATT_DIR: Final[Path]    = MODELS_DIR / "attention"
MODELS_TRANS_DIR: Final[Path]  = MODELS_DIR / "transformer"

# --- Experiments directory (training runs, hyper-parameter sweeps) ---
EXPERIMENTS_DIR: Final[Path]       = PROJECT_ROOT / "experiments"
EXP_RNN_DIR: Final[Path]          = EXPERIMENTS_DIR / "rnn"
EXP_LSTM_DIR: Final[Path]         = EXPERIMENTS_DIR / "lstm"
EXP_ATT_DIR: Final[Path]          = EXPERIMENTS_DIR / "attention"
EXP_TRANS_DIR: Final[Path]        = EXPERIMENTS_DIR / "transformer"

# --- Output directories ---
OUTPUT_DIR: Final[Path]         = PROJECT_ROOT / "outputs"
LOGS_DIR: Final[Path]          = OUTPUT_DIR / "logs"
REPORTS_DIR: Final[Path]       = OUTPUT_DIR / "reports"
PLOTS_DIR: Final[Path]         = OUTPUT_DIR / "plots"
COMPARISONS_DIR: Final[Path]   = OUTPUT_DIR / "comparisons"

# --- Additional project directories ---
ASSETS_DIR: Final[Path]        = PROJECT_ROOT / "assets"
DOCS_DIR: Final[Path]          = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR: Final[Path]    = PROJECT_ROOT / "notebooks"
SRC_DIR: Final[Path]           = PROJECT_ROOT / "src"
SRC_MODELS_DIR: Final[Path]   = SRC_DIR / "models"
TESTS_DIR: Final[Path]        = PROJECT_ROOT / "tests"

# Aggregate list for auto-creation (§10).
_ALL_DIRECTORIES: Final[List[Path]] = [
    # Data
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    # Models
    MODELS_RNN_DIR,
    MODELS_LSTM_DIR,
    MODELS_ATT_DIR,
    MODELS_TRANS_DIR,
    # Experiments
    EXP_RNN_DIR,
    EXP_LSTM_DIR,
    EXP_ATT_DIR,
    EXP_TRANS_DIR,
    # Outputs
    LOGS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    COMPARISONS_DIR,
    # Project scaffolding
    ASSETS_DIR,
    DOCS_DIR,
    NOTEBOOKS_DIR,
    SRC_MODELS_DIR,
    TESTS_DIR,
]

# ============================================================================
# 2. DATASET PATHS
# ============================================================================

# Raw dataset (shipped with the repo).
RAW_DATASET_PATH: Final[Path] = (
    RAW_DATA_DIR / "customer_support_tickets_200k.csv"
)

# Processed artefacts produced by the preprocessing pipeline.
# Train / validation / test splits all reside under data/processed/.
PROCESSED_DATASET_PATH: Final[Path] = PROCESSED_DATA_DIR / "processed_tickets.csv"
TRAIN_DATASET_PATH: Final[Path]     = PROCESSED_DATA_DIR / "train.csv"
VAL_DATASET_PATH: Final[Path]       = PROCESSED_DATA_DIR / "validation.csv"
TEST_DATASET_PATH: Final[Path]      = PROCESSED_DATA_DIR / "test.csv"

# Tokeniser / label-encoder artefacts.
TOKENIZER_PATH: Final[Path]        = PROCESSED_DATA_DIR / "tokenizer.json"
LABEL_ENCODER_PATH: Final[Path]    = PROCESSED_DATA_DIR / "label_encoder.json"

# ============================================================================
# 3. DATASET COLUMN NAMES
# ============================================================================

# Column that concatenates product context (used as auxiliary feature).
COL_PRODUCT: Final[str]           = "product"

# Free-text column that the models will learn from.
COL_ISSUE_DESCRIPTION: Final[str] = "issue_description"

# Target label column.
COL_CATEGORY: Final[str]          = "category"

#Text to be created during the preprocessing.
COL_TEXT: Final[str]              = "text"  

# ============================================================================
# 4. NLP CONFIGURATION
# ============================================================================

VOCAB_SIZE: Final[int]       = 20_000    # Maximum vocabulary size for tokeniser.
MAX_SEQUENCE_LENGTH: Final[int] = 150    # Pad / truncate every sequence to this length.
EMBEDDING_DIM: Final[int]    = 128       # Dimensionality of word embeddings.
OOV_TOKEN: Final[str]        = "<OOV>"   # Out-of-vocabulary placeholder token.
PADDING_TYPE: Final[str]     = "post"    # Pad sequences at the end ("post") or start ("pre").
TRUNCATION_TYPE: Final[str]  = "post"    # Truncate sequences at the end ("post") or start ("pre").

# ============================================================================
# 5. TRAINING HYPERPARAMETERS
# ============================================================================

BATCH_SIZE: Final[int]        = 64
EPOCHS: Final[int]            = 20
LEARNING_RATE: Final[float]   = 1e-3     # Adam default; BERT fine-tuning may override.
VALIDATION_SPLIT: Final[float] = 0.15    # Fraction of data reserved for validation.
SEED: Final[int]              = 42       # Global random seed for reproducibility.

# ============================================================================
# 6. MODEL HYPERPARAMETERS
# ============================================================================

# --- Simple RNN ---
RNN_UNITS: Final[int]              = 128

# --- LSTM / BiLSTM ---
LSTM_UNITS: Final[int]             = 128

# --- Self-Attention (used with BiLSTM + Attention) ---
ATTENTION_UNITS: Final[int]        = 64

# --- Regularisation ---
DROPOUT_RATE: Final[float]         = 0.3   # Applied after dense / attention layers.
RECURRENT_DROPOUT: Final[float]    = 0.2   # Applied inside recurrent cells.

# ============================================================================
# 7. MODEL SAVING PATHS
# ============================================================================

# Each architecture gets its own sub-directory under the top-level models/ dir.
# Keras ``model.save()`` will write the full SavedModel / .keras here.

MODEL_SAVE_PATHS: Final[Dict[str, Path]] = {
    "rnn":         MODELS_RNN_DIR / "simple_rnn_model.keras",
    "lstm":        MODELS_LSTM_DIR / "lstm_model.keras",
    "attention":   MODELS_ATT_DIR / "bilstm_attention_model.keras",
    "transformer": MODELS_TRANS_DIR / "transformer_model.keras",
}

# Training-history CSVs (loss, accuracy per epoch).
MODEL_HISTORY_PATHS: Final[Dict[str, Path]] = {
    name: path.with_suffix(".history.csv")
    for name, path in MODEL_SAVE_PATHS.items()
}

# ============================================================================
# 8. COMPARISON / REPORTING PATHS
# ============================================================================

COMPARISON_REPORT_PATH: Final[Path] = COMPARISONS_DIR / "model_comparison.csv"
COMPARISON_PLOT_PATH: Final[Path]   = COMPARISONS_DIR / "model_comparison.png"

# ============================================================================
# 9. LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL: Final[int]          = logging.INFO
LOG_FORMAT: Final[str]         = (
    "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
)
LOG_DATE_FORMAT: Final[str]    = "%Y-%m-%d %H:%M:%S"
LOG_FILE: Final[Path]          = LOGS_DIR / "ticketvision.log"


def setup_logging(
    level: int = LOG_LEVEL,
    log_file: Path | None = LOG_FILE,
) -> logging.Logger:
    """Configure the project-wide root logger.

    Parameters
    ----------
    level : int
        Minimum severity level (default: ``LOG_LEVEL``).
    log_file : Path | None
        If provided, logs are also written to this file.

    Returns
    -------
    logging.Logger
        The configured root logger for the project.
    """
    logger = logging.getLogger("ticketvision")
    logger.setLevel(level)

    # Prevent duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler — always active.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — optional.
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# 10. AUTO-CREATE REQUIRED DIRECTORIES
# ============================================================================

def _create_directories() -> None:
    """Ensure every directory in ``_ALL_DIRECTORIES`` exists.

    Also creates parent directories for each model-save path so that
    ``model.save(...)`` never fails due to a missing folder.
    """
    for directory in _ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

    for model_path in MODEL_SAVE_PATHS.values():
        model_path.parent.mkdir(parents=True, exist_ok=True)


# Run on first import — directories are ready before any other module needs them.
_create_directories()


# ============================================================================
# 11. SELF-TEST
# ============================================================================

def _print_section(title: str) -> None:
    """Print a formatted section header to stdout."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


if __name__ == "__main__":
    _print_section("TicketVision-AI  -  Configuration Summary")

    # --- Paths ---
    _print_section("Project Paths")
    print(f"  Project Root          : {PROJECT_ROOT}")
    print(f"  Raw Data Dir          : {RAW_DATA_DIR}")
    print(f"  Processed Data Dir    : {PROCESSED_DATA_DIR}")
    print(f"  Assets Dir            : {ASSETS_DIR}")
    print(f"  Docs Dir              : {DOCS_DIR}")
    print(f"  Notebooks Dir         : {NOTEBOOKS_DIR}")
    print(f"  Source Dir            : {SRC_DIR}")
    print(f"  Tests Dir             : {TESTS_DIR}")

    _print_section("Dataset Paths")
    print(f"  Raw Dataset           : {RAW_DATASET_PATH}")
    print(f"  Processed Dataset     : {PROCESSED_DATASET_PATH}")
    print(f"  Train Split           : {TRAIN_DATASET_PATH}")
    print(f"  Validation Split      : {VAL_DATASET_PATH}")
    print(f"  Test Split            : {TEST_DATASET_PATH}")
    print(f"  Tokenizer             : {TOKENIZER_PATH}")
    print(f"  Label Encoder         : {LABEL_ENCODER_PATH}")

    _print_section("Models Directory")
    print(f"  Models Root           : {MODELS_DIR}")
    print(f"    RNN                 : {MODELS_RNN_DIR}")
    print(f"    LSTM                : {MODELS_LSTM_DIR}")
    print(f"    Attention           : {MODELS_ATT_DIR}")
    print(f"    Transformer         : {MODELS_TRANS_DIR}")

    _print_section("Experiments Directory")
    print(f"  Experiments Root      : {EXPERIMENTS_DIR}")
    print(f"    RNN                 : {EXP_RNN_DIR}")
    print(f"    LSTM                : {EXP_LSTM_DIR}")
    print(f"    Attention           : {EXP_ATT_DIR}")
    print(f"    Transformer         : {EXP_TRANS_DIR}")

    _print_section("Output Directories")
    print(f"  Logs Dir              : {LOGS_DIR}")
    print(f"  Reports Dir           : {REPORTS_DIR}")
    print(f"  Plots Dir             : {PLOTS_DIR}")
    print(f"  Comparisons Dir       : {COMPARISONS_DIR}")

    # --- Columns ---
    _print_section("Dataset Columns")
    print(f"  Product Column        : {COL_PRODUCT}")
    print(f"  Issue Description Col : {COL_ISSUE_DESCRIPTION}")
    print(f"  Category Column       : {COL_CATEGORY}")

    # --- NLP ---
    _print_section("NLP Configuration")
    print(f"  Vocabulary Size       : {VOCAB_SIZE:,}")
    print(f"  Max Sequence Length    : {MAX_SEQUENCE_LENGTH}")
    print(f"  Embedding Dimension   : {EMBEDDING_DIM}")
    print(f"  OOV Token             : {OOV_TOKEN}")
    print(f"  Padding Type          : {PADDING_TYPE}")
    print(f"  Truncation Type       : {TRUNCATION_TYPE}")

    # --- Training ---
    _print_section("Training Hyperparameters")
    print(f"  Batch Size            : {BATCH_SIZE}")
    print(f"  Epochs                : {EPOCHS}")
    print(f"  Learning Rate         : {LEARNING_RATE}")
    print(f"  Validation Split      : {VALIDATION_SPLIT}")
    print(f"  Random Seed           : {SEED}")

    # --- Model ---
    _print_section("Model Hyperparameters")
    print(f"  RNN Units             : {RNN_UNITS}")
    print(f"  LSTM Units            : {LSTM_UNITS}")
    print(f"  Attention Units       : {ATTENTION_UNITS}")
    print(f"  Dropout Rate          : {DROPOUT_RATE}")
    print(f"  Recurrent Dropout     : {RECURRENT_DROPOUT}")

    # --- Save Paths ---
    _print_section("Model Save Paths")
    for name, path in MODEL_SAVE_PATHS.items():
        status = "OK" if path.parent.exists() else "MISSING"
        print(f"  {name:<15s} : {path}  [{status}]")

    # --- Logging ---
    _print_section("Logging")
    print(f"  Log Level             : {logging.getLevelName(LOG_LEVEL)}")
    print(f"  Log File              : {LOG_FILE}")
    print(f"  Log Format            : {LOG_FORMAT}")

    # --- Directory check ---
    _print_section("Directory Status")
    all_ok = True
    for d in _ALL_DIRECTORIES:
        exists = d.exists()
        mark = "[OK]" if exists else "[!!]"
        print(f"  {mark}  {d}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n  All directories verified successfully.")
    else:
        print("\n  WARNING: Some directories are missing!")

    print(f"\n{'=' * 60}")
    print("  Self-test complete.")
    print(f"{'=' * 60}\n")
