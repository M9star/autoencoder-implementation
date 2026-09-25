"""Train one MNIST autoencoder family and write figures.

Usage:
    python train.py --model dense
    python train.py --model convolutional --epochs 20
    python train.py --model denoising
    python train.py --model variational
    python train.py --model sparse
    python train.py --model all --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import tensorflow as tf

from autoencoders import FAMILIES, get_family
from autoencoders.data import load_mnist
from autoencoders.figures import save_figures, save_loss_curve
from autoencoders.gpu import configure_gpu

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MNIST autoencoder family.")
    parser.add_argument(
        "--model",
        default="dense",
        choices=[*FAMILIES, "all"],
        help="Which family to train.",
    )
    parser.add_argument("--latent-dim", type=int, default=None, help="Bottleneck size. Each family has its own default.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def family_dirs(name: str) -> tuple[Path, Path]:
    return MODEL_DIR / name, OUTPUT_DIR / name


def clear_family(name: str) -> tuple[Path, Path]:
    model_dir, output_dir = family_dirs(name)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in model_dir.glob("*.keras"):
        path.unlink()
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
    return model_dir, output_dir


def train_family(family, args: argparse.Namespace) -> None:
    latent_dim = args.latent_dim if args.latent_dim is not None else family.DEFAULT_LATENT
    print(f"family: {family.NAME}")
    print(f"bottleneck: {latent_dim}  ({latent_dim / (28 * 28):.2%} of the input)")

    encoder, decoder, autoencoder = family.build(latent_dim)
    family.compile(autoencoder)
    autoencoder.summary()

    x_train, x_test = load_mnist()
    model_dir, output_dir = clear_family(family.NAME)

    loss: list[float] = []
    val_loss: list[float] = []
    best_val = float("inf")
    best_weights = None
    best_epoch = 0
    for epoch in range(args.epochs):
        x_in, y_in = family.prepare_xy(x_train)
        history = autoencoder.fit(
            x_in,
            y_in,
            epochs=1,
            batch_size=args.batch_size,
            validation_split=0.10,
        )
        loss.append(history.history["loss"][0])
        val_loss.append(history.history["val_loss"][0])
        if val_loss[-1] < best_val:
            best_val = val_loss[-1]
            best_epoch = epoch + 1
            best_weights = autoencoder.get_weights()

    if best_weights is not None:
        autoencoder.set_weights(best_weights)
    print(f"best validation loss {best_val:.6f} at epoch {best_epoch}")

    path = model_dir / f"{family.NAME}-{latent_dim}-{args.epochs}.keras"
    autoencoder.save(path)
    print(f"saved {path}")

    x_eval, y_eval = family.prepare_xy(x_test)
    test_loss = autoencoder.evaluate(x_eval, y_eval, verbose=0)
    if isinstance(test_loss, list):
        test_loss = test_loss[0]
    print(f"test loss: {test_loss:.6f}")

    save_loss_curve(output_dir, loss, val_loss)
    save_figures(
        output_dir,
        encoder,
        decoder,
        autoencoder,
        x_test,
        latent_dim,
        samples=family.SAMPLES,
        sparsity=family.SPARSITY,
    )
    print(f"figures written to {output_dir}")


def main() -> None:
    args = parse_args()
    configure_gpu()
    tf.keras.utils.set_random_seed(args.seed)

    names = list(FAMILIES) if args.model == "all" else [args.model]
    for name in names:
        train_family(get_family(name), args)


if __name__ == "__main__":
    main()
