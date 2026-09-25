"""Shared optimizer for every autoencoder family."""

from __future__ import annotations

from tensorflow import keras


def adam() -> keras.optimizers.Optimizer:
    """Adam at 0.001 with a per-step decay of 1e-6."""
    learning_rate = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.001,
        decay_steps=1,
        decay_rate=1e-6,
    )
    return keras.optimizers.Adam(learning_rate=learning_rate)
