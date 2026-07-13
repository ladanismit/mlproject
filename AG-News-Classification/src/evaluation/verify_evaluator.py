"""Verification script for NLPEvaluator.

This script tests the NLPEvaluator functionality end-to-end using a dummy Keras model,
a mock Keras Tokenizer, and a CustomLabelEncoder. It ensures all metrics calculations,
JSON output generation, and logging work properly.
"""

import sys
from pathlib import Path

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pickle
import tensorflow as tf
from src.data.label_encoder import CustomLabelEncoder
from src.evaluation.evaluator import NLPEvaluator
from src.utils.logger import get_logger

logger = get_logger("verify_evaluator")


def create_dummy_assets(temp_dir: Path) -> tuple[Path, Path, Path]:
    """Creates dummy model, tokenizer, and label encoder for verification.

    Args:
        temp_dir (Path): Path to store the temporary verification assets.

    Returns:
        tuple[Path, Path, Path]: Paths to (model, tokenizer, label_encoder).
    """
    logger.info("Creating dummy assets for evaluation testing...")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build and save a tiny dummy Keras model
    inputs = tf.keras.Input(shape=(80,), dtype=tf.int32, name="input_layer")
    x = tf.keras.layers.Embedding(input_dim=100, output_dim=8, input_length=80)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="output_layer")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DummyVerificationModel")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    model_path = temp_dir / "dummy_model.h5"
    model.save(str(model_path))
    logger.info(f"Dummy model saved to: {model_path}")

    # 2. Build and save a dummy Keras Tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    # Fit on some sample texts
    tokenizer.fit_on_texts([
        "world cup sports football tournament",
        "business stocks shares market finance merger",
        "science space technology engineering AI",
        "international politics world conflict summit"
    ])
    
    tokenizer_path = temp_dir / "dummy_tokenizer.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Dummy tokenizer saved to: {tokenizer_path}")

    # 3. Build and save a CustomLabelEncoder
    class_labels_map = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech"
    }
    label_encoder = CustomLabelEncoder(class_labels_map=class_labels_map)
    label_encoder.fit([1, 2, 3, 4])
    
    label_encoder_path = temp_dir / "dummy_label_encoder.pkl"
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Dummy label encoder saved to: {label_encoder_path}")

    return model_path, tokenizer_path, label_encoder_path


def main() -> None:
    """Orchestrates dummy evaluation verification."""
    temp_dir = Path(_project_root) / "saved_models" / "verify_temp"
    
    try:
        # Create verification assets
        model_path, tokenizer_path, label_encoder_path = create_dummy_assets(temp_dir)

        # Initialize NLPEvaluator
        logger.info("Initializing NLPEvaluator...")
        evaluator = NLPEvaluator(
            model_path=model_path,
            model_type="rnn",
            tokenizer_path=tokenizer_path,
            label_encoder_path=label_encoder_path,
            max_sequence_length=80
        )

        # Sample test data
        test_texts = [
            "the world leaders met at the summit in geneva to discuss politics",
            "the football player scored a last minute goal in the tournament final",
            "shares of tech giants fell dramatically in today's stock market trade",
            "nasa launched a new space exploration telescope to study distant planets"
        ]
        # True labels matching the class map (1: World, 2: Sports, 3: Business, 4: Sci/Tech)
        test_labels = [1, 2, 3, 4]

        # Run evaluation
        metrics_save_path = temp_dir / "evaluation_results.json"
        logger.info("Running evaluation...")
        metrics = evaluator.evaluate(
            texts=test_texts,
            labels=test_labels,
            batch_size=2,
            save_path=metrics_save_path
        )

        # Validation assertions
        assert "accuracy" in metrics, "Accuracy metric is missing!"
        assert "precision_macro" in metrics, "Macro Precision metric is missing!"
        assert "f1_macro" in metrics, "Macro F1 metric is missing!"
        assert "classification_report" in metrics, "Classification report is missing!"
        assert "confusion_matrix" in metrics, "Confusion matrix is missing!"
        assert metrics_save_path.exists(), "Results JSON was not saved!"

        logger.info("=" * 60)
        logger.info("VERIFICATION COMPLETED SUCCESSFULLY!")
        logger.info(f"Generated Metrics: {metrics}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary assets
        logger.info("Cleaning up temporary verification assets...")
        for filename in ["dummy_model.h5", "dummy_tokenizer.pkl", "dummy_label_encoder.pkl", "evaluation_results.json"]:
            filepath = temp_dir / filename
            if filepath.exists():
                filepath.unlink()
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass
        logger.info("Cleanup completed.")


if __name__ == "__main__":
    main()
