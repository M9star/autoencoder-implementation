"""Denoising autoencoder.

Same convolutional body as ``convolutional``. Training input is a noisy
digit; the target is the clean digit. Noise is redrawn every epoch.
"""

from __future__ import annotations

from tensorflow import keras

from autoencoders.convolutional import build as build_conv
from autoencoders.images import gaussian_noise
from autoencoders.optim import adam

NAME = "denoising"
DEFAULT_LATENT = 64
SAMPLES = False
SPARSITY = False
NOISE_STD = 0.35


def build(latent_dim: int = DEFAULT_LATENT) -> tuple[keras.Model, keras.Model, keras.Model]:
    encoder, decoder, _ = build_conv(latent_dim)
    autoencoder = keras.Model(encoder.input, decoder(encoder(encoder.input)), name=NAME)
    return encoder, decoder, autoencoder


def compile(autoencoder: keras.Model) -> keras.Model:
    autoencoder.compile(optimizer=adam(), loss="mse")
    return autoencoder


def prepare_xy(images):
    return gaussian_noise(images, std=NOISE_STD), images
