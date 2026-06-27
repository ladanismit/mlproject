"""DocVision-AI Single Image Prediction Module.

This script executes inference on individual document images using the best
trained Keras model. It reuses the aspect-ratio preserving OpenCV preprocessing
pipeline from `preprocessing.py` and maps predicted outputs to classes
imported from `config.py`.

It can be run directly from the command line:
    python src/predict.py --image path/to/image.png
"""

import logging
import sys
from pathlib import Path
from typing import Union
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

# Import configuration constants and preprocessing functions
from src.config import (
    BEST_MODEL_PATH,
    CLASSES,
    IDX_TO_CLASS,
    OUTPUTS_DIR,
)
from src.preprocessing import preprocess_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def predict_image(
    image_path: Union[str, Path],
    model: tf.keras.Model,
) -> dict:
    """Validates, preprocesses, and classifies a single document image.

    Args:
        image_path (Union[str, Path]): Path to the target image file.
        model (tf.keras.Model): Loaded Keras classification model.

    Returns:
        dict: A dictionary containing:
            - "class_name": String name of the predicted document class
            - "confidence": Float percentage of the prediction confidence
            - "probabilities": Dict of class name to probability score

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If preprocessing fails or image is corrupted.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Target image file not found: {path}")

    logger.info(f"Preprocessing image for inference: {path.name}")
    try:
        # 1. Run the identical preprocessing used during training
        processed = preprocess_image(path)

        # 2. Add the batch dimension: shape becomes [1, H, W, C]
        batch_input = np.expand_dims(processed, axis=0)

        # 3. Predict class probabilities
        logger.info("Executing model inference...")
        preds = model.predict(batch_input, verbose=0)[0]

        # 4. Map probabilities to classes
        pred_idx = int(np.argmax(preds))
        predicted_class = IDX_TO_CLASS[pred_idx]
        confidence = float(preds[pred_idx]) * 100.0

        probabilities = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

        results = {
            "class_name": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }
        logger.info(f"Prediction success. Result: {predicted_class} ({confidence:.2f}%)")
        return results

    except Exception as e:
        logger.error(f"Error during image prediction: {e}")
        raise ValueError(f"Failed to process and predict image: {e}") from e


def display_prediction(
    image_path: Union[str, Path],
    prediction_results: dict,
) -> None:
    """Displays the input image along with classification labels and confidence scores using Matplotlib.

    If run in a headless environment, saves the plotted figure as an image file.

    Args:
        image_path (Union[str, Path]): Path to the input image file.
        prediction_results (dict): The result dictionary returned by predict_image().
    """
    path = Path(image_path)
    class_name = prediction_results["class_name"]
    confidence = prediction_results["confidence"]

    logger.info("Plotting classification result...")
    try:
        # Load image for display
        img = cv2.imread(str(path.resolve()))
        if img is None:
            raise ValueError("Failed to read image for display.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.axis("off")

        title_text = (
            f"File: {path.name}\n"
            f"Prediction: {class_name} | Confidence: {confidence:.2f}%"
        )
        plt.title(title_text, fontsize=14, fontweight="bold", pad=15)

        # Add probability distribution details as a textbox in the plot
        prob_text = "\n".join(
            [f"{k}: {v*100:.2f}%" for k, v in prediction_results["probabilities"].items()]
        )
        plt.figtext(
            0.15,
            0.05,
            f"Probabilities:\n{prob_text}",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        )

        plt.tight_layout()

        # Handle running in headless environments gracefully
        backend = plt.get_backend()
        if backend.lower() == "agg":
            output_plot = OUTPUTS_DIR / f"{path.stem}_prediction.png"
            plt.savefig(str(output_plot), dpi=150)
            logger.info(f"Running in a headless environment. Plot saved to: {output_plot}")
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Failed to render image plot: {e}")


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Inference CLI script for DocVision-AI Document Classifier."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=r"D:\MLPs\DocVision AI\data\raw\resume raw\london-bd8262b0.jpg",
        help="Path to the document image file (JPG, PNG, BMP, TIFF, etc.)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip displaying/saving the prediction visualization plot.",
    )
    args = parser.parse_args()

    # 1. Check model existence
    if not BEST_MODEL_PATH.exists():
        logger.error(
            f"Trained model not found at {BEST_MODEL_PATH}. "
            "Please run src/trainer.py first to train and save the model."
        )
        sys.exit(1)

    # 2. Load model directly using tf.keras.models.load_model
    logger.info(f"Loading trained Keras model directly from: {BEST_MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model file: {e}")
        sys.exit(1)

    # 3. Predict and Display
    try:
        results = predict_image(args.image, model)

        print("\n" + "=" * 40)
        print("Prediction Results (JSON):")
        print("=" * 40)
        print(json.dumps(results, indent=4))
        print("=" * 40 + "\n")

        if not args.no_plot:
            display_prediction(args.image, results)

    except Exception as err:
        logger.error(f"Inference pipeline failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
