"""MNIST autoencoder family.

Each family is a module with ``build``, ``compile``, and ``prepare_xy``.
"""

from __future__ import annotations

from autoencoders import convolutional, denoising, dense, sparse, variational

FAMILIES = {
    dense.NAME: dense,
    convolutional.NAME: convolutional,
    denoising.NAME: denoising,
    variational.NAME: variational,
    sparse.NAME: sparse,
}


def get_family(name: str):
    try:
        return FAMILIES[name]
    except KeyError as exc:
        known = ", ".join(FAMILIES)
        raise SystemExit(f"Unknown model {name!r}. Choose one of: {known}") from exc
