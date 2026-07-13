"""Model Comparison module for the AGNews Text Classification project.

This module consolidates evaluation metrics (accuracy, precision, recall, f1-score),
computational properties (parameter counts, file size), and efficiency metrics (training time)
across all supported model architectures (RNN, LSTM, BiLSTM + Attention, BERT).
It outputs consolidated comparisons as CSV and JSON and identifies the best-performing model.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tensorflow as tf

from configs.config import ARTIFACTS_DIR, SAVED_MODELS_DIR
from src.utils.exception import CustomException
from src.utils.file_io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping from model type identifiers to architecture metadata
MODEL_METADATA_MAP: Dict[str, Dict[str, Any]] = {
    "rnn": {
        "friendly_name": "Simple RNN",
        "filenames": ["rnn_final_model.h5", "rnn_best_model.h5"],
        "metric_keys": ["rnn_metrics.json", "simple_rnn_metrics.json"],
        "history_keys": ["rnn_history.json"]
    },
    "lstm": {
        "friendly_name": "LSTM",
        "filenames": ["lstm_final_model.h5", "lstm_best_model.h5"],
        "metric_keys": ["lstm_metrics.json", "lstm_evaluation.json"],
        "history_keys": ["lstm_history.json"]
    },
    "attention": {
        "friendly_name": "BiLSTM + Self-Attention",
        "filenames": ["attention_final_model.h5", "bilstm_attention_best_model.h5"],
        "metric_keys": ["attention_metrics.json", "bilstm_attention_metrics.json"],
        "history_keys": ["attention_history.json"]
    },
    "bert": {
        "friendly_name": "BERT Transformer",
        "filenames": ["bert_final_model.h5", "bert_best_model.h5"],
        "metric_keys": ["bert_metrics.json", "bert_transformer_metrics.json"],
        "history_keys": ["bert_history.json"]
    }
}


class ModelComparator:
    """Orchestrates model size, parameter counting, and metric comparisons.

    Consolidates multiple models' files and metrics configurations into a single
    comparison dataframe and identifies the best-performing model.
    """

    def __init__(
        self,
        saved_models_dir: Union[str, Path] = SAVED_MODELS_DIR,
        artifacts_dir: Union[str, Path] = ARTIFACTS_DIR
    ) -> None:
        """Initializes the ModelComparator with target directory locations.

        Args:
            saved_models_dir (Union[str, Path]): Directory containing saved Keras models.
            artifacts_dir (Union[str, Path]): Directory containing evaluation output metrics.
        """
        self.saved_models_dir = Path(saved_models_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = get_logger(self.__class__.__name__)

    def _locate_file(self, directories: List[Path], candidates: List[str]) -> Optional[Path]:
        """Searches for a file within multiple directories based on list of naming candidates.

        Args:
            directories (List[Path]): Directories to search.
            candidates (List[str]): Naming candidates.

        Returns:
            Optional[Path]: Found file path or None if not found.
        """
        for directory in directories:
            if not directory.exists():
                continue
            for candidate in candidates:
                path = directory / candidate
                if path.exists() and path.is_file():
                    return path
        return None

    def get_model_size_mb(self, model_path: Path) -> float:
        """Calculates size of the model checkpoint in Megabytes.

        Args:
            model_path (Path): Path to model file.

        Returns:
            float: Size in Megabytes.
        """
        try:
            if model_path.exists():
                size_bytes = os.path.getsize(model_path)
                return round(size_bytes / (1024 * 1024), 2)
            return 0.0
        except Exception as e:
            self.logger.warning(f"Failed to fetch file size for '{model_path}': {e}")
            return 0.0

    def count_model_parameters(self, model_path: Path, model_type: str) -> int:
        """Dynamically loads model checkpoint and returns total parameter counts.

        Args:
            model_path (Path): Path to model file.
            model_type (str): Type of model architecture.

        Returns:
            int: Total model parameters. -1 if loading fails.
        """
        try:
            custom_objects: Dict[str, Any] = {}
            if model_type == "attention":
                from src.models.custom_attention import CustomSelfAttention
                custom_objects["CustomSelfAttention"] = CustomSelfAttention
            elif model_type == "bert":
                os.environ["TF_USE_LEGACY_KERAS"] = "1"
                from transformers import TFBertModel
                custom_objects["TFBertModel"] = TFBertModel

            # Load model without compilation details to optimize loading time
            model = tf.keras.models.load_model(
                str(model_path),
                custom_objects=custom_objects,
                compile=False
            )
            return int(model.count_params())
        except Exception as e:
            self.logger.warning(f"Could not calculate parameters dynamically for model {model_type}: {e}")
            return -1

    def load_metrics_file(self, model_type: str) -> Dict[str, Any]:
        """Loads evaluation metrics JSON file for the model type.

        Searches in both the artifacts and saved models directory.

        Args:
            model_type (str): Model key ('rnn', 'lstm', 'attention', 'bert').

        Returns:
            Dict[str, Any]: Loaded metrics dictionary. Empty dict if missing.
        """
        meta = MODEL_METADATA_MAP[model_type]
        search_dirs = [self.artifacts_dir, self.saved_models_dir]
        
        metrics_path = self._locate_file(search_dirs, meta["metric_keys"])
        if not metrics_path:
            self.logger.warning(f"Metrics file missing for model type '{model_type}'. Attempted search: {meta['metric_keys']}")
            return {}

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(f"Loaded evaluation metrics from: {metrics_path}")
            return data
        except Exception as e:
            self.logger.warning(f"Could not load metrics file {metrics_path}: {e}")
            return {}

    def extract_training_time(self, model_type: str, metrics: Dict[str, Any]) -> float:
        """Retrieves or estimates training time for a model.

        Args:
            model_type (str): Model type name.
            metrics (Dict[str, Any]): Loaded evaluation metrics dictionary.

        Returns:
            float: Training time in seconds. NaN if unavailable.
        """
        # 1. Check in metrics JSON keys first
        for key in ["training_time", "training_time_seconds", "elapsed_time", "duration"]:
            if key in metrics:
                return float(metrics[key])

        # 2. Check in training history metadata JSON if available
        meta = MODEL_METADATA_MAP[model_type]
        history_path = self._locate_file([self.saved_models_dir, self.artifacts_dir], meta["history_keys"])
        if history_path:
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                for key in ["training_time", "elapsed_time", "time_elapsed", "duration"]:
                    if key in history:
                        return float(history[key])
            except Exception as e:
                self.logger.warning(f"Could not parse training history for elapsed time: {e}")

        return float("nan")

    def compare(self, primary_metric: str = "f1_macro") -> tuple[pd.DataFrame, Optional[str]]:
        """Aggregates metrics and physical properties to compare all models.

        Args:
            primary_metric (str): Metric name to decide the best model. Defaults to 'f1_macro'.

        Returns:
            tuple[pd.DataFrame, Optional[str]]: Consolidated DataFrame and best model name.

        Raises:
            CustomException: If execution fails.
        """
        try:
            self.logger.info("Initializing model comparison routine...")
            records = []

            for model_type, meta in MODEL_METADATA_MAP.items():
                self.logger.info(f"Processing details for architecture: {meta['friendly_name']}")
                
                # Check for model checkpoint file
                model_path = self._locate_file([self.saved_models_dir], meta["filenames"])
                
                # Calculate size and param counts
                model_size = 0.0
                params = -1
                if model_path:
                    model_size = self.get_model_size_mb(model_path)
                    params = self.count_model_parameters(model_path, model_type)
                else:
                    self.logger.warning(f"Model checkpoint file not found for: '{model_type}'")

                # Load performance metrics
                metrics = self.load_metrics_file(model_type)
                
                # Extract standard metrics
                accuracy = metrics.get("accuracy", float("nan"))
                p_macro = metrics.get("precision_macro", float("nan"))
                r_macro = metrics.get("recall_macro", float("nan"))
                f_macro = metrics.get("f1_macro", float("nan"))
                p_weighted = metrics.get("precision_weighted", float("nan"))
                r_weighted = metrics.get("recall_weighted", float("nan"))
                f_weighted = metrics.get("f1_weighted", float("nan"))

                # Resolve training time
                training_time = self.extract_training_time(model_type, metrics)

                records.append({
                    "Model Name": meta["friendly_name"],
                    "Model Type": model_type,
                    "Accuracy": accuracy,
                    "Precision (Macro)": p_macro,
                    "Recall (Macro)": r_macro,
                    "F1-Score (Macro)": f_macro,
                    "Precision (Weighted)": p_weighted,
                    "Recall (Weighted)": r_weighted,
                    "F1-Score (Weighted)": f_weighted,
                    "Training Time (Sec)": training_time,
                    "Parameters": params if params >= 0 else float("nan"),
                    "Model Size (MB)": model_size
                })

            df = pd.DataFrame(records)
            
            # Identify best model automatically
            best_model_name = None
            primary_col = None
            
            # Map primary metric string to DataFrame column name
            metric_to_col = {
                "accuracy": "Accuracy",
                "f1_macro": "F1-Score (Macro)",
                "f1_weighted": "F1-Score (Weighted)",
                "precision_macro": "Precision (Macro)",
                "recall_macro": "Recall (Macro)"
            }
            primary_col = metric_to_col.get(primary_metric.lower(), "F1-Score (Macro)")

            # Filter valid models that have the primary column evaluated
            valid_df = df.dropna(subset=[primary_col])
            if not valid_df.empty:
                idx_max = valid_df[primary_col].idxmax()
                best_model_name = valid_df.loc[idx_max, "Model Name"]
                best_value = valid_df.loc[idx_max, primary_col]
                self.logger.info(f"Automatically identified best model: '{best_model_name}' based on {primary_col} ({best_value:.4f})")
            else:
                self.logger.warning("Could not identify the best model automatically because no evaluation metrics are loaded.")

            return df, best_model_name
        except Exception as e:
            raise CustomException(e, sys) from e

    def save_results(self, df: pd.DataFrame, base_filename: str = "model_comparison") -> None:
        """Saves comparison DataFrame as CSV and JSON.

        Args:
            df (pd.DataFrame): Comparison DataFrame.
            base_filename (str): Base filename. Defaults to "model_comparison".

        Raises:
            CustomException: If saving fails.
        """
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # CSV Path
            csv_path = self.artifacts_dir / f"{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Consolidated comparisons saved to CSV: {csv_path}")

            # JSON Path
            json_path = self.artifacts_dir / f"{base_filename}.json"
            # Convert NaN to None for JSON serialization compatibility
            json_compatible_df = df.replace({np.nan: None})
            records = json_compatible_df.to_dict(orient="records")
            save_json(json_path, records)
            self.logger.info(f"Consolidated comparisons saved to JSON: {json_path}")
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Classification Model Comparison Tool.")
    parser.add_argument(
        "--saved_models_dir",
        type=str,
        default=str(SAVED_MODELS_DIR),
        help=f"Path to checkpoints folder. Default: {SAVED_MODELS_DIR}"
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=str(ARTIFACTS_DIR),
        help=f"Path to metrics/eval results folder. Default: {ARTIFACTS_DIR}"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="model_comparison",
        help="Base name of generated report files."
    )
    parser.add_argument(
        "--primary_metric",
        type=str,
        default="f1_macro",
        choices=["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"],
        help="Metric to identify the best model. Default: f1_macro"
    )

    try:
        args = parser.parse_args()
        comparator = ModelComparator(
            saved_models_dir=args.saved_models_dir,
            artifacts_dir=args.artifacts_dir
        )
        comparison_df, best_model = comparator.compare(primary_metric=args.primary_metric)
        comparator.save_results(comparison_df, base_filename=args.output_name)

        print("\n" + "=" * 80)
        print("CONSOLIDATED MODEL COMPARISON DASHBOARD")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)
        if best_model:
            print(f"RECOMMENDED MODEL (Best Performer): {best_model} (Metric: {args.primary_metric})")
            print("=" * 80)
        print()

    except Exception as error:
        logger.error(f"Error occurred in standalone comparison: {error}")
        sys.exit(1)
