"""Shared reconstruction figures, plus extras for VAE and sparse models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from autoencoders.images import add_noise, naive_resize, remove_values


def _show(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(np.squeeze(image), cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def _latent_vector(encoder, images: np.ndarray) -> np.ndarray:
    encoded = encoder.predict(images, verbose=0)
    if isinstance(encoded, (list, tuple)):
        encoded = encoded[0]
    return np.asarray(encoded)


def save_loss_curve(output_dir: Path, loss: list[float], val_loss: list[float]) -> None:
    epochs = range(1, len(loss) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(epochs), loss, label="train")
    ax.plot(list(epochs), val_loss, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=120)
    plt.close(fig)


def save_figures(
    output_dir: Path,
    encoder,
    decoder,
    autoencoder,
    x_test: np.ndarray,
    latent_dim: int,
    *,
    samples: bool = False,
    sparsity: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = x_test[0]
    side = int(np.sqrt(latent_dim))
    encoded = _latent_vector(encoder, sample[None, ...])
    reconstructed = autoencoder.predict(sample[None, ...], verbose=0)

    print(f"encoded shape: {encoded[0].shape}")
    print(f"encoded values: {np.array2string(encoded[0], precision=3, suppress_small=True)}")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    _show(axes[0], sample, "original")
    if side * side == latent_dim:
        _show(axes[1], encoded[0].reshape(side, side), f"encoded {side}x{side}")
    else:
        axes[1].plot(encoded[0])
        axes[1].set_title(f"encoded ({latent_dim})")
        axes[1].set_xticks([])
    _show(axes[2], reconstructed[0], "reconstructed")
    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction.png", dpi=120)
    plt.close(fig)

    count = 8
    originals = x_test[:count]
    decoded = autoencoder.predict(originals, verbose=0)
    fig, axes = plt.subplots(2, count, figsize=(count * 1.4, 3.2))
    for i in range(count):
        _show(axes[0, i], originals[i], "original" if i == 0 else "")
        _show(axes[1, i], decoded[i], "decoded" if i == 0 else "")
    fig.tight_layout()
    fig.savefig(output_dir / "test_grid.png", dpi=120)
    plt.close(fig)

    if side * side == latent_dim:
        small, stretched = naive_resize(sample, side)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        _show(axes[0], small, f"resized to {side}x{side}")
        _show(axes[1], stretched, "stretched back")
        _show(axes[2], reconstructed[0], "autoencoder")
        fig.tight_layout()
        fig.savefig(output_dir / "resize_vs_autoencoder.png", dpi=120)
        plt.close(fig)

    noisy = add_noise(sample, random_chance=5)
    denoised = autoencoder.predict(noisy[None, ...], verbose=0)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    _show(axes[0], sample, "original")
    _show(axes[1], noisy, "5% noise")
    _show(axes[2], denoised[0], "denoised")
    fig.tight_layout()
    fig.savefig(output_dir / "denoise.png", dpi=120)
    plt.close(fig)

    gapped = remove_values(sample, random_chance=35)
    filled = autoencoder.predict(gapped[None, ...], verbose=0)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    _show(axes[0], sample, "original")
    _show(axes[1], gapped, "35% pixels removed")
    _show(axes[2], filled[0], "filled in")
    fig.tight_layout()
    fig.savefig(output_dir / "fill_gaps.png", dpi=120)
    plt.close(fig)

    if samples:
        z = tf.random.normal((16, latent_dim))
        generated = decoder.predict(z, verbose=0)
        fig, axes = plt.subplots(2, 8, figsize=(11.2, 3.2))
        for i, ax in enumerate(axes.ravel()):
            _show(ax, generated[i], "sample" if i == 0 else "")
        fig.tight_layout()
        fig.savefig(output_dir / "samples.png", dpi=120)
        plt.close(fig)

    if sparsity:
        codes = _latent_vector(encoder, x_test[:2000])
        inactive = float(np.mean(codes < 1e-3))
        print(f"latent units near zero: {inactive:.1%}")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(codes.ravel(), bins=60)
        ax.set_title(f"latent activations  ({inactive:.1%} near zero)")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output_dir / "sparsity.png", dpi=120)
        plt.close(fig)
