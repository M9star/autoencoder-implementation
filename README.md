# MNIST autoencoder family

Each model compresses a 28×28 handwritten digit and rebuilds it. Training is unsupervised: the model is never told which digit it is looking at.

| Family | Module | What is different |
|---|---|---|
| `dense` | `autoencoders/dense.py` | Fully connected bottleneck. Copies a clean digit. |
| `convolutional` | `autoencoders/convolutional.py` | Convolution keeps the 2D layout of the image. |
| `denoising` | `autoencoders/denoising.py` | Same convolutional body. Input is noisy; target is clean. |
| `variational` | `autoencoders/variational.py` | Bottleneck is a distribution. Decoder can sample new digits. |
| `sparse` | `autoencoders/sparse.py` | Wide bottleneck with an L1 penalty so most units stay off. |

Shared pieces live next to those modules: `data.py` loads `data/mnist.npz`, `gpu.py` turns on tensorflow-metal, `figures.py` writes the plots. The dataset, trained weights, and figures stay on this machine and are not in git.

## Setup

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --model dense
python train.py --model convolutional --epochs 20
python train.py --model denoising
python train.py --model variational
python train.py --model sparse
python train.py --model all --epochs 5
```

Defaults: 3 epochs, batch size 128, 10% of the training images held out for validation. Bottleneck size is 64 for dense, convolutional, and denoising; 16 for variational; 256 for sparse. Training runs on the Mac GPU through tensorflow-metal.

Each family writes only to its own folders. A later run of that family replaces its own checkpoint and figures, and leaves the others alone.

```
models/<family>/<family>-<latent>-<epochs>.keras
outputs/<family>/
```

| File | Contents |
|---|---|
| `loss.png` | train and validation loss by epoch |
| `reconstruction.png` | original digit, encoded view, reconstruction |
| `test_grid.png` | several test digits and their reconstructions |
| `resize_vs_autoencoder.png` | nearest-neighbor shrink-and-stretch vs the autoencoder (square latent only) |
| `denoise.png` | a digit with 5% random pixels, then reconstructed |
| `fill_gaps.png` | a digit with 35% of pixels set to 0, then reconstructed |
| `samples.png` | new digits from the variational decoder |
| `sparsity.png` | latent-activation histogram for the sparse model |
