"""Compatibility wrapper around the dense autoencoder."""

from autoencoders.data import DATA_PATH, IMAGE_SHAPE, PIXELS, load_mnist
from autoencoders.dense import build as build_autoencoder
from autoencoders.dense import compile as compile_autoencoder
from autoencoders.images import add_noise, naive_resize, remove_values

__all__ = [
    "DATA_PATH",
    "IMAGE_SHAPE",
    "PIXELS",
    "add_noise",
    "build_autoencoder",
    "compile_autoencoder",
    "load_mnist",
    "naive_resize",
    "remove_values",
]
