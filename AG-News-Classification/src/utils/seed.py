"""Seed configuration module for project-wide reproducibility.

This module provides a unified setup for initializing random seeds across Python's
built-in random, NumPy, and TensorFlow. It also enforces determinism in TensorFlow ops.
"""

import os
import random
import sys
from typing import Optional

import numpy as np
import tensorflow as tf

from configs.config import RANDOM_SEED
from src.utils.exception import CustomException
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def set_seed(seed: Optional[int] = None) -> None:
    """Configures random seeds and environment variables for reproducibility.
    
    Sets seeds for python's native `random`, `numpy.random`, and `tensorflow.random`.
    Additionally, configures environment flags and TensorFlow settings to run
    algorithms deterministically where supported.
    
    Args:
        seed (Optional[int]): Target integer seed. Defaults to RANDOM_SEED from config.py.
        
    Raises:
        CustomException: Wrapped exception if configuration fails.
    """
    try:
        target_seed = seed if seed is not None else RANDOM_SEED
        logger.info(f"Configuring environment with reproducibility seed: {target_seed}")

        # Set environment variables for hash and GPU determinism
        os.environ["PYTHONHASHSEED"] = str(target_seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

        # Set standard library and package seeds
        random.seed(target_seed)
        np.random.seed(target_seed)
        tf.random.set_seed(target_seed)

        # Enable TensorFlow experimental op determinism if supported (TensorFlow 2.9+)
        try:
            tf.config.experimental.enable_op_determinism()
            logger.info("Enforced TensorFlow operation determinism.")
        except AttributeError:
            logger.warning(
                "tf.config.experimental.enable_op_determinism() is not supported in this "
                "TensorFlow version. Falling back to environment variables."
            )

        logger.info("Reproducibility settings successfully configured.")
    except Exception as e:
        raise CustomException(e, sys) from e
