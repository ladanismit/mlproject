"""Verification script for NLPPredictor.

This script tests the NLPPredictor class functionality by mocking Keras checkpoints,
Keras tokenizers, and label encoders, executing single and batch predictions,
and verifying the prediction response structure.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pickle
import shutil
import tensorflow as tf
from src.data.label_encoder import CustomLabelEncoder
from src.pipeline.predict import NLPPredictor
from src.utils.logger import get_logger

logger = get_logger("verify_predictor")


def create_dummy_assets(temp_dir: Path) -> tuple[Path, Path, Path]:
    """Creates temporary mock Keras model, tokenizer, and label encoder.

    Args:
        temp_dir (Path): Path to store the mock files.

    Returns:
        tuple[Path, Path, Path]: (model_path, tokenizer_path, label_encoder_path)
    """
    logger.info("Creating mock assets for prediction testing...")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build and save a tiny dummy Keras model
    inputs = tf.keras.Input(shape=(80,), dtype=tf.int32, name="input_layer")
    x = tf.keras.layers.Embedding(input_dim=100, output_dim=8, input_length=80)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="output_layer")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    model_path = temp_dir / "dummy_rnn_final_model.h5"
    model.save(str(model_path))
    logger.info(f"Dummy model saved: {model_path}")

    # 2. Build and save a Keras Tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(["hello world news sport business tech science"])
    
    tokenizer_path = temp_dir / "dummy_rnn_tokenizer.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Dummy tokenizer saved: {tokenizer_path}")

    # 3. Build and save a label encoder
    class_labels_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    label_encoder = CustomLabelEncoder(class_labels_map=class_labels_map)
    label_encoder.fit([1, 2, 3, 4])
    
    label_encoder_path = temp_dir / "dummy_rnn_label_encoder.pkl"
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Dummy label encoder saved: {label_encoder_path}")

    return model_path, tokenizer_path, label_encoder_path


def main() -> None:
    """Orchestrates predictor verification."""
    temp_dir = Path(_project_root) / "saved_models" / "verify_predict_temp"

    try:
        # Create verification assets
        model_path, tokenizer_path, label_encoder_path = create_dummy_assets(temp_dir)

        # Initialize NLPPredictor pointing to the mock assets
        logger.info("Initializing NLPPredictor...")
        predictor = NLPPredictor(
            model_type="rnn",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            label_encoder_path=label_encoder_path,
            max_sequence_length=80
        )

        # Test 1: Single Prediction
        logger.info("Testing Single Prediction...")
        single_text = "Acme Corp stocks surged today after the merger announcement."
        single_result = predictor.predict(single_text)
        
        logger.info(f"Single prediction result:\n{json.dumps(single_result, indent=4)}")
        
        assert isinstance(single_result, dict), "Single prediction output must be a dictionary!"
        assert "predicted_class" in single_result, "predicted_class missing!"
        assert "confidence_score" in single_result, "confidence_score missing!"
        assert "probability_distribution" in single_result, "probability_distribution missing!"
        assert len(single_result["probability_distribution"]) == 4, "Expected 4 class probabilities!"

        # Test 2: Batch Prediction
        logger.info("Testing Batch Prediction...")
        batch_texts = [
            "the soccer tournament matches start tomorrow.",
            "scientists found water molecules on a distant asteroid."
        ]
        batch_results = predictor.predict(batch_texts)

        logger.info(f"Batch prediction result:\n{json.dumps(batch_results, indent=4)}")

        assert isinstance(batch_results, list), "Batch prediction output must be a list!"
        assert len(batch_results) == 2, f"Expected 2 outputs, got {len(batch_results)}"
        for res in batch_results:
            assert isinstance(res, dict), "Batch items must be dictionaries!"
            assert "predicted_class" in res
            assert "confidence_score" in res
            assert "probability_distribution" in res

        logger.info("=" * 80)
        logger.info("VERIFICATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)
    finally:
        # Clean up mock directories
        logger.info("Cleaning up mock files...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.info("Cleanup completed.")


if __name__ == "__main__":
    main()
