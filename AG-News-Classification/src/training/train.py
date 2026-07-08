"""Main training orchestrator script for the AGNews Text Classification project.

This script manages the complete training pipeline including argument parsing,
logger initialization, random seed configuration, dataset loading, preprocessing,
tokenizer training and saving, label encoding and saving, TF Dataset optimization,
model building, training, callback orchestration, and saving outputs.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.config import (
    BATCH_SIZE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    OOV_TOKEN,
    RANDOM_SEED,
    SAVED_MODELS_DIR,
    TEXT_COLUMN,
    LABEL_COLUMN,
    VALIDATION_SPLIT,
    VOCAB_SIZE,
)
from src.data.data_loader import load_both_datasets
from src.data.dataset import create_model_datasets
from src.data.label_encoder import (
    create_and_initialize_encoder,
    encode_labels,
    save_label_encoder,
)
from src.data.preprocessing import preprocess_dataframe
from src.data.tokenizer_utils import (
    convert_texts_to_sequences,
    create_tokenizer,
    fit_tokenizer_on_text,
    pad_text_sequences,
    save_tokenizer,
)
from src.utils.exception import CustomException
from src.utils.file_io import save_json
from src.utils.logger import get_logger
from src.utils.seed import set_seed

# Initialize logger
logger = get_logger("training_orchestrator")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for model selection and training parameters.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Enterprise training entrypoint for AGNews text classification models."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["rnn", "lstm", "attention", "bert"],
        help="Target neural network architecture type.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs. Defaults to configs.config.EPOCHS ({EPOCHS}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for datasets. Defaults to configs.config.BATCH_SIZE ({BATCH_SIZE}).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Adam optimizer learning rate. Defaults to configs.config.LEARNING_RATE ({LEARNING_RATE}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f"Patience for Keras EarlyStopping callback. Defaults to {EARLY_STOPPING_PATIENCE}.",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
        help=f"Min delta for Keras EarlyStopping callback. Defaults to {EARLY_STOPPING_MIN_DELTA}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility. Defaults to configs.config.RANDOM_SEED ({RANDOM_SEED}).",
    )
    return parser.parse_args()


def get_model_class(model_type: str) -> type:
    """Dynamically imports and retrieves the selected model class.

    Args:
        model_type (str): The requested model identifier.

    Returns:
        type: The uninstantiated class of the requested model.

    Raises:
        NotImplementedError: If the model architecture is not yet implemented.
        ValueError: If model type is unrecognized.
    """
    logger.info(f"Loading class configuration for model type: '{model_type}'")
    if model_type == "rnn":
        from src.models.rnn_model import RNNModel
        return RNNModel
    elif model_type == "lstm":
        try:
            from src.models.lstm_model import LSTMModel
            return LSTMModel
        except ImportError as e:
            raise NotImplementedError(
                "LSTMModel is not yet implemented. Please implement it in "
                "src/models/lstm_model.py"
            ) from e
    elif model_type == "attention":
        try:
            from src.models.attention_model import AttentionModel
            return AttentionModel
        except ImportError as e:
            raise NotImplementedError(
                "AttentionModel is not yet implemented. Please implement it in "
                "src/models/attention_model.py"
            ) from e
    elif model_type == "bert":
        try:
            from src.models.bert_model import BERTModel
            return BERTModel
        except ImportError as e:
            raise NotImplementedError(
                "BERTModel is not yet implemented. Please implement it in "
                "src/models/bert_model.py"
            ) from e
    else:
        raise ValueError(f"Unrecognized model type: {model_type}")


def main() -> None:
    """Orchestrates the loading, preprocessing, and training pipeline of the model."""
    try:
        # 1. Command Line Parsing
        args = parse_args()
        logger.info("=" * 60)
        logger.info(f"STARTING MODEL TRAINING PIPELINE - MODEL: {args.model.upper()}")
        logger.info("=" * 60)
        logger.info(f"CLI Parameters: {vars(args)}")

        # 2. Reproducibility Configuration
        set_seed(args.seed)

        # 3. Load Dataset
        logger.info("Loading training and testing CSV datasets...")
        train_df, test_df = load_both_datasets()
        logger.info(f"Loaded datasets. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # 4. Preprocessing Pipeline
        logger.info("Executing text cleaning and preprocessing...")
        train_df = preprocess_dataframe(train_df, text_column=TEXT_COLUMN)
        test_df = preprocess_dataframe(test_df, text_column=TEXT_COLUMN)

        # 5. Tokenizer Fitting & Sequence Encoding
        logger.info("Initializing tokenizer training...")
        tokenizer = create_tokenizer(vocab_size=VOCAB_SIZE, oov_token=OOV_TOKEN)
        tokenizer = fit_tokenizer_on_text(tokenizer, train_df[TEXT_COLUMN].tolist())
        
        # Save Tokenizer
        tokenizer_path = SAVED_MODELS_DIR / f"{args.model}_tokenizer.pkl"
        save_tokenizer(tokenizer, tokenizer_path)
        logger.info(f"Fitted tokenizer saved successfully to: {tokenizer_path}")

        # Encode Texts to Sequences
        logger.info("Converting text fields to sequences...")
        train_seqs = convert_texts_to_sequences(tokenizer, train_df[TEXT_COLUMN].tolist())
        test_seqs = convert_texts_to_sequences(tokenizer, test_df[TEXT_COLUMN].tolist())

        # Pad Sequences
        logger.info(f"Padding text sequences to max length of: {MAX_SEQUENCE_LENGTH}")
        train_features = pad_text_sequences(train_seqs, maxlen=MAX_SEQUENCE_LENGTH)
        test_features = pad_text_sequences(test_seqs, maxlen=MAX_SEQUENCE_LENGTH)

        # 6. Label Encoding
        logger.info("Initializing label encoding...")
        raw_train_labels = train_df[LABEL_COLUMN].tolist()
        raw_test_labels = test_df[LABEL_COLUMN].tolist()
        
        label_encoder = create_and_initialize_encoder(labels=raw_train_labels)
        
        # Save Label Encoder
        encoder_path = SAVED_MODELS_DIR / f"{args.model}_label_encoder.pkl"
        save_label_encoder(label_encoder, encoder_path)
        logger.info(f"Label encoder saved successfully to: {encoder_path}")

        # Encode classes
        train_labels = np.array(encode_labels(label_encoder, raw_train_labels))
        test_labels = np.array(encode_labels(label_encoder, raw_test_labels))

        # 7. Create TensorFlow Datasets
        logger.info("Creating optimized tf.data.Dataset pipelines...")
        train_ds, val_ds, test_ds = create_model_datasets(
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            validation_split=VALIDATION_SPLIT,
            batch_size=args.batch_size,
            random_seed=args.seed,
            cache=True,
        )

        # 8. Initialize Model Architecture
        model_class = get_model_class(args.model)
        logger.info(f"Instantiating model class: {model_class.__name__}")
        
        # RNN constructor accepts custom hyperparameters matching config or defaults
        # We dynamically initialize classes inheriting from BaseModel
        model_instance = model_class()
        
        logger.info("Building model layers...")
        model_instance.build_model()
        
        # Compile Model
        logger.info(f"Compiling model with learning_rate={args.learning_rate}...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        model_instance.compile_model(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Display model summary
        model_instance.summary()

        # Retrieve callbacks configuration
        logger.info("Generating training callbacks (EarlyStopping, Checkpoint)...")
        callbacks = model_instance.get_callbacks(
            patience=args.patience,
            min_delta=args.min_delta,
            checkpoint_dir=str(SAVED_MODELS_DIR)
        )

        # 9. Train the Model
        logger.info("Beginning model fitting and training process...")
        keras_model = model_instance.get_model()
        history = keras_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("Model training process finished successfully.")

        # 10. Save Final Model Output
        final_model_path = SAVED_MODELS_DIR / f"{args.model}_final_model.h5"
        logger.info(f"Saving final trained model checkpoint at: {final_model_path}")
        keras_model.save(str(final_model_path))
        logger.info("Final model saved successfully.")

        # 11. Save Training History & Metrics
        history_path = SAVED_MODELS_DIR / f"{args.model}_history.json"
        logger.info(f"Saving training history log to: {history_path}")
        save_json(history_path, history.history)
        logger.info("Training history logged successfully.")
        
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE - {args.model.upper()} MODEL DEPLOYED TO SAVED_MODELS")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error encountered during model training: {e}")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()
