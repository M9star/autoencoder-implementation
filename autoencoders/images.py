"""Pixel helpers used by training and figure generation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def add_noise(img: np.ndarray, random_chance: float = 5) -> np.ndarray:
    """Replace about ``random_chance`` percent of pixels with random values in [0, 1]."""
    image = np.asarray(img, dtype=np.float32)
    mask = np.random.rand(*image.shape) <= (random_chance / 100.0)
    noise = np.random.uniform(0.0, 1.0, size=image.shape).astype(np.float32)
    return np.where(mask, noise, image)


def gaussian_noise(images: np.ndarray, std: float = 0.35) -> np.ndarray:
    """Add clipped Gaussian noise. Used to train the denoising autoencoder."""
    noise = np.random.normal(0.0, std, size=images.shape).astype(np.float32)
    return np.clip(images + noise, 0.0, 1.0)


def remove_values(img: np.ndarray, random_chance: float = 5) -> np.ndarray:
    """Set about ``random_chance`` percent of pixels to 0, punching gaps in the digit."""
    image = np.asarray(img, dtype=np.float32)
    mask = np.random.rand(*image.shape) <= (random_chance / 100.0)
    return np.where(mask, np.float32(0.0), image)


def naive_resize(img: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a digit to ``size`` x ``size`` and stretch it back to 28x28."""
    squeezed = np.squeeze(img).astype(np.float32)
    small = tf.image.resize(squeezed[..., None], (size, size), method="nearest").numpy()
    restored = tf.image.resize(small, (28, 28), method="nearest").numpy()
    return np.squeeze(small), np.squeeze(restored)
