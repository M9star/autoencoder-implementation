"""Convolutional autoencoder. Keeps the 2D layout of the digit.

    img (28, 28, 1)
      -> Conv 32, stride 2     14 x 14 x 32
      -> Conv 64, stride 2      7 x  7 x 64
      -> Flatten / Dense(latent)
      -> Dense / Reshape        7 x  7 x 64
      -> ConvTranspose 64       14 x 14 x 64
      -> ConvTranspose 32       28 x 28 x 32
      -> Conv 1, sigmoid        28 x 28 x 1
"""

from __future__ import annotations

from tensorflow import keras

from autoencoders.data import IMAGE_SHAPE
from autoencoders.optim import adam

NAME = "convolutional"
DEFAULT_LATENT = 64
SAMPLES = False
SPARSITY = False


def build(latent_dim: int = DEFAULT_LATENT) -> tuple[keras.Model, keras.Model, keras.Model]:
    if latent_dim < 1:
        raise ValueError("latent_dim must be at least 1")

    encoder_input = keras.Input(shape=IMAGE_SHAPE, name="img")
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_input)
    x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="encoder_output")(x)
    encoder = keras.Model(encoder_input, latent, name="encoder")

    decoder_input = keras.Input(shape=(latent_dim,), name="encoded")
    x = keras.layers.Dense(7 * 7 * 64, activation="relu")(decoder_input)
    x = keras.layers.Reshape((7, 7, 64))(x)
    x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    decoder_output = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    autoencoder = keras.Model(encoder_input, decoder(encoder(encoder_input)), name=NAME)
    return encoder, decoder, autoencoder


def compile(autoencoder: keras.Model) -> keras.Model:
    autoencoder.compile(optimizer=adam(), loss="mse")
    return autoencoder


def prepare_xy(images):
    return images, images
