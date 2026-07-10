"""Configuration module for the AGNews Text Classification project.

This module contains all configuration parameters organized into logical,
strongly-typed configuration classes. It utilizes pathlib.Path for cross-platform
compatibility and contains only constants.
"""

from pathlib import Path
from typing import Final, Dict

# ==============================================================================
# PROJECT INFO
# ==============================================================================
PROJECT_NAME: Final[str] = "AGNews-Text-Classification"
VERSION: Final[str] = "1.0.0"
RANDOM_SEED: Final[int] = 42

# ==============================================================================
# DIRECTORY PATHS
# ==============================================================================
# Root directory of the project (parent of src/)
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

CONFIGS_DIR: Final[Path] = ROOT_DIR / "configs"
ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "artifacts"
LOGS_DIR: Final[Path] = ROOT_DIR / "logs"
SAVED_MODELS_DIR: Final[Path] = ROOT_DIR / "saved_models"

# Ensure crucial directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CONFIGS_DIR, ARTIFACTS_DIR, LOGS_DIR, SAVED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
TRAIN_FILE: Final[Path] = RAW_DATA_DIR / "train.csv"
TEST_FILE: Final[Path] = RAW_DATA_DIR / "test.csv"

TEXT_COLUMN: Final[str] = "description"
LABEL_COLUMN: Final[str] = "class"
NUM_CLASSES: Final[int] = 4

# Class labels mapped to their standard names in AG News dataset
CLASS_LABELS: Final[Dict[int, str]] = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

# ==============================================================================
# TEXT PREPROCESSING
# ==============================================================================
VOCAB_SIZE: Final[int] = 15000
MAX_SEQUENCE_LENGTH: Final[int] = 80
EMBEDDING_DIM: Final[int] = 64
OOV_TOKEN: Final[str] = "<OOV>"
PADDING_TYPE: Final[str] = "post"      # "pre" or "post"
TRUNCATING_TYPE: Final[str] = "post"   # "pre" or "post"

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
BATCH_SIZE: Final[int] = 128
EPOCHS: Final[int] = 10
LEARNING_RATE: Final[float] = 1e-3
VALIDATION_SPLIT: Final[float] = 0.2

# Early Stopping parameters
EARLY_STOPPING_PATIENCE: Final[int] = 3
EARLY_STOPPING_MIN_DELTA: Final[float] = 1e-4

# ==============================================================================
# MODEL CONFIGURATIONS & HYPERPARAMETERS
# ==============================================================================
MODEL_NAME_SIMPLE_RNN: Final[str] = "simple_rnn"
MODEL_NAME_LSTM: Final[str] = "lstm"
MODEL_NAME_BILSTM_ATTENTION: Final[str] = "bilstm_attention"
MODEL_NAME_BERT: Final[str] = "bert_transformer"

# --- Simple RNN Configuration ---
RNN_UNITS: Final[int] = 64
RNN_DROPOUT: Final[float] = 0.2
RNN_RECURRENT_DROPOUT: Final[float] = 0.0

# --- LSTM Configuration ---
LSTM_UNITS: Final[int] = 128
LSTM_DROPOUT: Final[float] = 0.2
LSTM_RECURRENT_DROPOUT: Final[float] = 0.2

# --- BiLSTM + Self-Attention Configuration ---
BILSTM_UNITS: Final[int] = 64
ATTENTION_HEADS: Final[int] = 4
ATTENTION_KEY_DIM: Final[int] = 64
ATTENTION_DROPOUT: Final[float] = 0.1

# --- BERT Transformer Configuration (HuggingFace / TF-Hub base) ---
BERT_MODEL_NAME: Final[str] = "bert-base-uncased"
BERT_MAX_LENGTH: Final[int] = 128
BERT_FINE_TUNE_LAYERS: Final[int] = 2  # Number of top layers to fine-tune
BERT_LEARNING_RATE: Final[float] = 2e-5

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
LOG_FILE: Final[Path] = LOGS_DIR / "project.log"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL: Final[str] = "INFO"
