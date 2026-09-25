"""Vanilla dense autoencoder.

    img (28, 28, 1)
      -> Flatten
      -> Dense(256, relu)
      -> Dense(latent, relu)
      -> Dense(256, relu)
      -> Dense(784, sigmoid)
      -> Reshape (28, 28, 1)
"""

from __future__ import annotations

from tensorflow import keras

from autoencoders.data import IMAGE_SHAPE, PIXELS
from autoencoders.optim import adam

NAME = "dense"
DEFAULT_LATENT = 64
SAMPLES = False
SPARSITY = False


def build(latent_dim: int = DEFAULT_LATENT) -> tuple[keras.Model, keras.Model, keras.Model]:
    if latent_dim < 1:
        raise ValueError("latent_dim must be at least 1")

    encoder_input = keras.Input(shape=IMAGE_SHAPE, name="img")
    flat = keras.layers.Flatten(name="flatten")(encoder_input)
    hidden = keras.layers.Dense(256, activation="relu", name="encoder_hidden")(flat)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="encoder_output")(hidden)
    encoder = keras.Model(encoder_input, latent, name="encoder")

    decoder_input = keras.Input(shape=(latent_dim,), name="encoded")
    hidden = keras.layers.Dense(256, activation="relu", name="decoder_hidden")(decoder_input)
    pixels = keras.layers.Dense(PIXELS, activation="sigmoid", name="decoder_dense")(hidden)
    decoder_output = keras.layers.Reshape(IMAGE_SHAPE, name="decoder_output")(pixels)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    autoencoder = keras.Model(encoder_input, decoder(encoder(encoder_input)), name=NAME)
    return encoder, decoder, autoencoder


def compile(autoencoder: keras.Model) -> keras.Model:
    autoencoder.compile(optimizer=adam(), loss="mse")
    return autoencoder


def prepare_xy(images):
    return images, images
