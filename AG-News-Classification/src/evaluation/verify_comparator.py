"""Verification script for ModelComparator.

This script tests the ModelComparator functionality by mocking training histories,
evaluation metrics, and tiny Keras model checkpoints. It validates parameter counting,
file size retrieval, metrics consolidation, and final report generation.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import shutil
import tensorflow as tf
from src.evaluation.compare_models import ModelComparator, MODEL_METADATA_MAP
from src.utils.logger import get_logger

logger = get_logger("verify_comparator")


def create_mock_metrics_and_models(temp_dir: Path) -> None:
    """Generates mock evaluation files and mini Keras checkpoints.

    Args:
        temp_dir (Path): Temporary workspace directory.
    """
    logger.info("Setting up mock evaluation files and checkpoints...")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mock metrics
    mock_metrics = {
        "rnn": {
            "accuracy": 0.72,
            "precision_macro": 0.70,
            "recall_macro": 0.71,
            "f1_macro": 0.705,
            "precision_weighted": 0.70,
            "recall_weighted": 0.71,
            "f1_weighted": 0.705,
            "training_time": 120.5
        },
        "lstm": {
            "accuracy": 0.81,
            "precision_macro": 0.80,
            "recall_macro": 0.80,
            "f1_macro": 0.80,
            "precision_weighted": 0.80,
            "recall_weighted": 0.80,
            "f1_weighted": 0.80,
            "training_time": 350.2
        },
        "attention": {
            "accuracy": 0.85,
            "precision_macro": 0.84,
            "recall_macro": 0.85,
            "f1_macro": 0.845,
            "precision_weighted": 0.84,
            "recall_weighted": 0.85,
            "f1_weighted": 0.845,
            "training_time": 480.0
        },
        "bert": {
            "accuracy": 0.91,
            "precision_macro": 0.90,
            "recall_macro": 0.91,
            "f1_macro": 0.905,
            "precision_weighted": 0.90,
            "recall_weighted": 0.91,
            "f1_weighted": 0.905,
            "training_time": 1200.8
        }
    }

    for model_type, metrics in mock_metrics.items():
        # Save metrics JSON
        meta = MODEL_METADATA_MAP[model_type]
        metrics_file = temp_dir / meta["metric_keys"][0]
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Mock metrics saved for {model_type}: {metrics_file}")

        # Save a tiny mock Keras model to count parameters
        inputs = tf.keras.Input(shape=(10,), dtype=tf.int32)
        x = tf.keras.layers.Embedding(input_dim=50, output_dim=4)(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model_file = temp_dir / meta["filenames"][0]
        model.save(str(model_file))
        logger.info(f"Mock checkpoint saved for {model_type}: {model_file}")


def main() -> None:
    """Orchestrates comparator verification."""
    temp_dir = Path(_project_root) / "saved_models" / "verify_comparator_temp"

    try:
        # Create temporary assets
        create_mock_metrics_and_models(temp_dir)

        # Instantiate comparator pointing to the temp directory
        logger.info("Initializing ModelComparator...")
        comparator = ModelComparator(
            saved_models_dir=temp_dir,
            artifacts_dir=temp_dir
        )

        # Run comparison
        logger.info("Running model comparison...")
        comparison_df, best_model = comparator.compare(primary_metric="f1_macro")

        # Save outputs
        output_name = "mock_comparison"
        comparator.save_results(comparison_df, base_filename=output_name)

        # Verify results
        csv_output = temp_dir / f"{output_name}.csv"
        json_output = temp_dir / f"{output_name}.json"

        assert csv_output.exists(), "CSV comparison report is missing!"
        assert json_output.exists(), "JSON comparison report is missing!"
        assert len(comparison_df) == 4, f"Expected 4 models compared, got {len(comparison_df)}"
        assert best_model == "BERT Transformer", f"Expected BERT to be the best performer, got: {best_model}"

        # Output verification details
        print("\n" + "=" * 80)
        print("MOCK MODEL COMPARISON RESULTS:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)
        print(f"VERIFIED BEST MODEL: {best_model}")
        print("=" * 80)
        print("VERIFICATION COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary files
        logger.info("Cleaning up mock files...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.info("Cleanup complete.")


if __name__ == "__main__":
    main()
