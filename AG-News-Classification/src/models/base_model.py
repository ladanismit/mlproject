"""Base model module for the AGNews Text Classification project.

This module defines the abstract base class `BaseModel` that enforces a common
interface for all model architectures used in the project, such as Simple RNN,
LSTM, BiLSTM + Self-Attention, and BERT Transformer.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import tensorflow as tf

# Add project root to sys.path to resolve imports when run directly
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract Base Class for all Deep Learning NLP models.

    This class defines the mandatory interface for implementing neural network
    models within the project's training, evaluation, and inference pipelines.
    It guarantees consistency in model construction, compilation, callback
    generation, and verification.

    Attributes:
        model (Optional[tf.keras.Model]): The compiled or uncompiled Keras model.
    """

    def __init__(self) -> None:
        """Initializes the base model and configures logging."""
        self.logger = get_logger(self.__class__.__name__)
        self.model: tf.keras.Model | None = None
        self.logger.info(f"Initialized {self.__class__.__name__} instance.")

    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Builds and returns the Keras model architecture.

        This method defines the layers, inputs, and outputs of the neural network
        model, assigns it to the internal `self.model` attribute, and returns it.

        Returns:
            tf.keras.Model: The constructed Keras model instance.

        Raises:
            CustomException: If model building fails.
        """
        pass

    @abstractmethod
    def compile_model(
        self,
        optimizer: str | tf.keras.optimizers.Optimizer,
        loss: str | tf.keras.losses.Loss,
        metrics: list[str | tf.keras.metrics.Metric] | None = None,
        **kwargs: Any,
    ) -> None:
        """Compiles the Keras model with optimizer, loss, and metrics.

        Args:
            optimizer (str | tf.keras.optimizers.Optimizer): Optimizer instance or registered name.
            loss (str | tf.keras.losses.Loss): Loss instance or registered name.
            metrics (list[str | tf.keras.metrics.Metric] | None): List of metrics to monitor during training and evaluation. Defaults to None.
            **kwargs (Any): Additional keyword arguments passed to Keras model compile.

        Raises:
            CustomException: If model compilation fails.
        """
        pass

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        """Retrieves the Keras model instance.

        Returns:
            tf.keras.Model: The compiled or uncompiled Keras model.

        Raises:
            ValueError: If the model has not been built yet.
        """
        pass

    @abstractmethod
    def get_callbacks(self, **kwargs: Any) -> list[tf.keras.callbacks.Callback]:
        """Generates standard Keras callbacks for model training.

        This should return a list of standard callbacks (e.g., EarlyStopping,
        ModelCheckpoint, TensorBoard, LearningRateScheduler) to be used during
        training.

        Args:
            **kwargs (Any): Additional configuration for callbacks.

        Returns:
            list[tf.keras.callbacks.Callback]: A list of Keras callback instances.
        """
        pass

    @abstractmethod
    def summary(self) -> None:
        """Prints the model summary.

        Raises:
            ValueError: If the model has not been built yet.
        """
        pass
