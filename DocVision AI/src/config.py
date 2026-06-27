"""DocVision-AI Configuration Settings.

This module defines all project-wide constants, path configurations,
data parameters, and training hyperparameters for the DocVision-AI pipeline.
It is designed to be clean, modular, and easily extensible for new document classes.

To add new document classes:
Simply append the new class names to the `CLASSES` list in the Dataset Configuration section.
"""

from pathlib import Path
import torch

# ==============================================================================
# 1. PATH CONFIGURATION (Pathlib based)
# ==============================================================================
# Project root directory (resolves to the directory containing this config's parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Models and outputs
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Development directories
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure all critical directories exist
directories = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    CHECKPOINT_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
    NOTEBOOKS_DIR,
]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. DATASET CONFIGURATION
# ==============================================================================
# Classes to classify documents into.
# Easily extendable by adding new document types here (e.g., "Land Records", "Bank Statements").
CLASSES = [
    "Resume",
    "Invoice",
    # Add new classes here to extend the pipeline
]

# Total number of classes
NUM_CLASSES = len(CLASSES)

# Mapping dictionary for class name to index and vice versa
CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: class_name for idx, class_name in enumerate(CLASSES)}

# Ensure directory for each class exists in raw data directory
for class_name in CLASSES:
    (RAW_DATA_DIR / class_name).mkdir(parents=True, exist_ok=True)



# ==============================================================================
# 3. IMAGE CONFIGURATION
# ==============================================================================
# Standard target size for deep learning models (e.g., ResNet, ViT)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Color channels (3 for RGB, 1 for Grayscale)
IMAGE_CHANNELS = 3

# Allowed file extensions for document images
SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".pdf",  # PDFs will be converted to images during preprocessing
}


# ==============================================================================
# 4. TRAINING CONFIGURATION
# ==============================================================================
# Reproducibility
RANDOM_SEED = 42

# Hardware acceleration setup
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Optimization parameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# Split ratio for train/validation
VALIDATION_SPLIT = 0.2


# ==============================================================================
# 5. MODEL SAVING CONFIGURATION
# ==============================================================================
# Specific paths for storing model checkpoints and final models
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pth"


# ==============================================================================
# 6. CONFIGURATION VERIFICATION (Utility function)
# ==============================================================================
def print_config_summary() -> None:
    """Prints a summary of the current configuration to the console."""
    print("=" * 60)
    print("DocVision-AI Configuration Summary")
    print("=" * 60)
    print(f"Project Root:       {PROJECT_ROOT}")
    print(f"Device:             {DEVICE}")
    print(f"Classes:            {CLASSES} (Total: {NUM_CLASSES})")
    print(f"Image Size:         {IMAGE_SIZE} (Channels: {IMAGE_CHANNELS})")
    print(f"Training Epochs:    {EPOCHS}")
    print(f"Batch Size:         {BATCH_SIZE}")
    print(f"Learning Rate:      {LEARNING_RATE}")
    print(f"Best Model Path:    {BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    # Test directory creation and configuration validity when executed directly
    print_config_summary()
