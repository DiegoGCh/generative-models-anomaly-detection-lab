# Lab 2: Anomaly Detection with MVTec AD

Course: Generative Models
Dataset: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## What this does

We train a convolutional VAE on normal images only. The model learns to reconstruct what "normal" looks like. When it sees a defective image, reconstruction fails at the defect region because the model never saw that pattern during training. The pixel-wise difference between input and reconstruction is the anomaly map. Threshold that map and you get a binary defect mask.

```
input -> VAE -> reconstruction -> pixel error -> gaussian blur -> threshold -> mask
```

## Why VAE and not a plain autoencoder

Two reasons. First, the course framework is probabilistic, so we use the VAE loss (reconstruction + KL). Second, the VAE gives us a log-likelihood which a plain AE does not, and that opens a second detection signal we can use if needed.

We use beta-VAE with a low beta (0.5). The original beta-VAE paper (Higgins et al., 2017) increases beta to get disentangled representations. We go the opposite direction: low beta means less KL pressure, which means the encoder keeps more information about the input, which means sharper reconstructions, which means fewer false positives.

## Why LATENT_DIM matters more than you think

The latent dimension is not a quality knob. It is the detection mechanism.

A large latent space gives the model enough capacity to reconstruct defects too, which kills the anomaly signal. A small latent space forces the model to compress: only the most common patterns (normal ones) fit. Defects, being rare and unseen during training, do not compress well and get reconstructed poorly. That poor reconstruction is what the detector fires on.

We started at 128 and ran a second pass at 64 for classes where the maps were activating but not precisely enough.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate       # Windows
pip install -r requirements.txt
```

Put the MVTec AD dataset in `Data/`. The folder structure should look like:

```
Data/
  bottle/
    train/good/
    test/good/
    test/broken_large/
    ...
    ground_truth/
  cable/
  ...
```

## Running

```bash
# Smoke test: bottle only, 5 epochs, 128px
python main.py

# Full run: all 15 classes, 50 epochs, 256px
python main.py --full

# Single class
python main.py --full --category carpet

# Re-run borderline classes with LATENT_DIM=64
python main.py --rerun

# Re-evaluate existing checkpoints with improved anomaly map (no retraining)
python main.py --eval-only --full

# Re-evaluate single class
python main.py --eval-only --category hazelnut
```

## What gets saved

```
checkpoints/           # model weights per class (LATENT_DIM=128)
checkpoints_rerun/     # model weights per class (LATENT_DIM=64)
results/               # figures and F1 curves, baseline evaluation
results_rerun/         # same but for the LATENT_DIM=64 rerun
results_improved/      # figures and F1 curves, multi-scale SSIM evaluation
results/results.json   # F1 table
```

Figures per class: one good image (map should be flat), two defective images (map should fire on the defect).

## Loss function

```
L = lambda1 * MSE(x, x_hat) + lambda2 * (1 - SSIM(x, x_hat)) + beta * KL
```

Both reconstruction and KL terms are normalized by their number of elements before combining. Without that, beta does not mean what you think it means: the reconstruction term sums over ~196k pixels and the KL sums over 128 dimensions, so raw they are on completely different scales.

## Anomaly map

```
ssim_err = mean over w in {3, 7, 11} of (1 - SSIM_w(x, x_hat))   # per-pixel
score    = lambda1 * L2(x, x_hat) + lambda2 * ssim_err
score    = gaussian_blur(score, sigma=4)
score    = normalize globally across all test images for that class
```

**Multi-scale per-pixel SSIM.** The pytorch-msssim library with `size_average=False` returns one SSIM value per image, not per pixel. For spatial localization, a sliding Gaussian window at three scales is used instead. Small windows (3) detect fine texture differences; large windows (11) detect structural differences. Averaging across scales makes the map robust to defects of different sizes.

**Global normalization.** Per-image normalization means every image, including clean ones, gets normalized to [0, 1] and fires at low thresholds. With global normalization, a clean image that reconstructs well stays at low values across the board.

Gaussian sigma=4 is standard for MVTec (Bergmann et al., 2019). It removes isolated pixel noise without blurring actual defect regions.

## Threshold selection

We sweep 50 threshold values on the test set and report F1 at the best one per class. This is the standard evaluation protocol for MVTec: you are evaluating the quality of the anomaly map, not the threshold selection. A good map will have a clear peak in the F1 vs threshold curve. A flat curve means the map is not discriminating.

## Results summary

Two evaluation runs on the same trained model (LATENT_DIM=128, 50 epochs):

| Class | F1 baseline | F1 improved | Notes |
|---|---|---|---|
| hazelnut | 0.169 | **0.378** | map fires on defect |
| metal_nut | 0.172 | **0.243** | map fires on defect |
| bottle | 0.104 | **0.196** | ring pattern, marginal |
| leather | 0.077 | **0.195** | map fires on defect correctly |
| cable | 0.121 | **0.202** | partial detection |
| wood | 0.077 | **0.219** | detects large defects |
| zipper | 0.040 | **0.119** | partial |
| pill | 0.084 | **0.116** | fires on text imprint, not actual defect |
| tile | 0.136 | **0.122** | slight regression (small window SSIM penalizes regular texture) |
| transistor | 0.071 | **0.074** | partial |
| capsule | 0.058 | **0.086** | map fires on defect, small defects hurt F1 |
| screw | 0.008 | **0.053** | structural failure |
| grid | 0.015 | **0.032** | structural failure |
| carpet | 0.031 | **0.031** | structural failure |
| toothbrush | 0.029 | **0.029** | structural failure |
| **avg** | **0.080** | **0.140** | |

The improved evaluation uses multi-scale per-pixel SSIM (windows 3, 7, 11) instead of a single scalar SSIM. The original pytorch-msssim call with `size_average=False` returns one value per image, not per pixel, so the SSIM term in the baseline contributed nothing to spatial localization. With per-pixel SSIM, the anomaly map captures both fine-grained texture differences (small window) and structural differences (large window), improving average F1 by 75% without retraining.

Classes marked as structural failure share a pattern: the VAE outputs a flat gray blob for reconstruction. The model learned the background but not the object. This happens with objects that vary in orientation across images (screw, toothbrush) or with periodic textures where reconstruction error is uniformly high everywhere (grid, carpet). Reducing LATENT_DIM does not fix this. It is a known limitation of reconstruction-based methods for these specific classes.

## Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| IMG_SIZE | 256 | better mask precision |
| LATENT_DIM | 128 / 64 (rerun) | calibrated, not maximized |
| beta | 0.5 | low for sharper reconstructions |
| epochs | 50 | small dataset, converges fast |
| batch_size | 16 | fits in 8GB VRAM |
| optimizer | Adam lr=1e-4 | 1e-3 tends to diverge on VAE |
| lambda1, lambda2 | 0.5, 0.5 | equal weight MSE and SSIM |
| gaussian sigma | 4 | standard for MVTec |
| augmentation | hflip, vflip, rot+-15 | avoids memorization of ~200-400 images per class |
