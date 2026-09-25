"""Apple GPU setup for tensorflow-metal."""

from __future__ import annotations

import tensorflow as tf


def configure_gpu() -> None:
    """Use the Apple GPU through tensorflow-metal.

    The Metal graph optimizer drops ReLU, so the network trains as a linear
    model and the loss diverges. Leave the rest of the GPU runtime on.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("No GPU found. Install tensorflow-metal and use TensorFlow 2.18.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    print(f"training device: {gpus[0].name}")
