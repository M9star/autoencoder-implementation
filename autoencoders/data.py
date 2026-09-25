"""MNIST loaders and shared image constants."""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_SHAPE = (28, 28, 1)
PIXELS = 28 * 28
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mnist.npz"


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST from data/mnist.npz and scale pixels from 0-255 into 0-1.

    Labels are discarded. The target is the image itself.
    """
    with np.load(DATA_PATH, allow_pickle=False) as data:
        x_train = data["x_train"]
        x_test = data["x_test"]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return x_train, x_test
