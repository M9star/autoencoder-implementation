"""Variational autoencoder.

The bottleneck is a distribution. After training, new digits come from
sampling ``N(0, 1)`` and running the decoder.
"""

from __future__ import annotations

import keras
import tensorflow as tf

from autoencoders.data import IMAGE_SHAPE
from autoencoders.optim import adam

NAME = "variational"
DEFAULT_LATENT = 16
SAMPLES = True
SPARSITY = False


@keras.saving.register_keras_serializable(package="autoencoders")
class Sampling(keras.layers.Layer):
    """z = mean + std * epsilon, with epsilon from N(0, 1)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@keras.saving.register_keras_serializable(package="autoencoders")
class VAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_tracker = keras.metrics.Mean(name="recon")
        self.kl_tracker = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.kl_tracker]

    def call(self, inputs):
        _z_mean, _z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def vae_losses(self, x, training: bool):
        z_mean, z_log_var, z = self.encoder(x, training=training)
        reconstruction = self.decoder(z, training=training)
        recon = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2))
        )
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        return recon + kl, recon, kl

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        with tf.GradientTape() as tape:
            total, recon, kl = self.vae_losses(x, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(total, self.trainable_variables), self.trainable_variables)
        )
        self.loss_tracker.update_state(total)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        total, recon, kl = self.vae_losses(x, training=False)
        self.loss_tracker.update_state(total)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        return {"encoder": keras.saving.serialize_keras_object(self.encoder),
                "decoder": keras.saving.serialize_keras_object(self.decoder)}

    @classmethod
    def from_config(cls, config):
        encoder = keras.saving.deserialize_keras_object(config["encoder"])
        decoder = keras.saving.deserialize_keras_object(config["decoder"])
        return cls(encoder, decoder)


def build(latent_dim: int = DEFAULT_LATENT) -> tuple[keras.Model, keras.Model, keras.Model]:
    if latent_dim < 1:
        raise ValueError("latent_dim must be at least 1")

    encoder_input = keras.Input(shape=IMAGE_SHAPE, name="img")
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_input)
    x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])
    encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

    decoder_input = keras.Input(shape=(latent_dim,), name="encoded")
    x = keras.layers.Dense(7 * 7 * 64, activation="relu")(decoder_input)
    x = keras.layers.Reshape((7, 7, 64))(x)
    x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    decoder_output = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    autoencoder = VAE(encoder, decoder, name=NAME)
    autoencoder.build((None, *IMAGE_SHAPE))
    return encoder, decoder, autoencoder


def compile(autoencoder: keras.Model) -> keras.Model:
    autoencoder.compile(optimizer=adam())
    return autoencoder


def prepare_xy(images):
    return images, images
