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

### Checkpoints (pre-trained weights)

Trained weights for all 15 classes are available here:
[Google Drive - checkpoints/](https://drive.google.com/drive/folders/1jpYVJED45yUzopOt_WvAs22kEDuP0YYK?usp=drive_link)

Download the `checkpoints/` folder and place it at the root of the repo:

```
Lab2/
  checkpoints/       <-- put it here
    bottle.pth
    cable.pth
    capsule.pth
    ...
  src/
  main.py
  ...
```

Then run the improved evaluation directly, no training needed:

```bash
# Evaluate all 15 classes with multi-scale per-pixel SSIM
python main.py --eval-only --full

# Evaluate a single class
python main.py --eval-only --category hazelnut
```

Results and figures are saved to `results_improved/`.

### Dataset

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

# V2: train with perceptual loss + KL annealing (all 15 classes)
python main.py --v2 --full

# V2: single class
python main.py --v2 --category bottle

# V2: re-evaluate existing V2 checkpoints (no retraining)
python main.py --eval-only --v2 --full
```

## What gets saved

```
checkpoints/           # model weights per class (LATENT_DIM=128, baseline)
checkpoints_rerun/     # model weights per class (LATENT_DIM=64)
checkpoints_v2/        # model weights per class (perceptual loss + KL annealing)
results/               # figures and F1 curves, baseline evaluation
results_rerun/         # same but for the LATENT_DIM=64 rerun
results_improved/      # figures and F1 curves, multi-scale SSIM evaluation
results_v2/            # figures and F1 curves, V2 evaluation
results/results.json   # F1 table
```

Figures per class: one good image (map should be flat), five defective images (map should fire on the defect).

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

## Why one model per class

One model per class is the correct and expected setup for MVTec. Each class has a completely different visual distribution. A single model trained on all classes would learn the average of everything and reconstruct nothing well. All reference papers (PatchCore, SPADE, CFlow, Bergmann et al. 2019) use separate models per class.

## Threshold selection

We sweep 50 threshold values on the test set and report F1 at the best one per class. This is the standard evaluation protocol for MVTec: you are evaluating the quality of the anomaly map, not the threshold selection.

A good map has a clear peak in the F1 vs threshold curve — there exists a threshold where defect pixels score high and normal pixels score low. A flat curve means the map is not discriminating: defect and normal scores overlap regardless of where you cut.

```
Good map (hazelnut):      Bad map (carpet):
F1                        F1
 |      *                  | * * * * * * *
 |    *   *                |
 +--------> threshold      +--------> threshold
   Clear peak                Flat — no threshold works
```

The reported F1 is always at the peak. The threshold itself is not the point.

## How to read the figures

Each figure has four panels: `Input | Reconstruction | Anomaly Map | Predicted Mask`

**Anomaly Map** uses a "hot" colormap: black = low error (normal), red = medium error, yellow/white = high error (likely defect).

**Predicted Mask** has two overlaid layers:
- White = model prediction (score >= threshold)
- Solid green = ground truth from the dataset (not generated by the model)
- White + green overlap = correct detection (TP)
- White only = false positive (FP)
- Green only = missed defect (FN)
- Gray background = nothing (no prediction, no GT)

Good images have no green overlay (no GT mask). White spots on good images are false positives — normal trade-off controlled by the threshold.

## What can still be improved (without changing the architecture)

| Improvement | Benefits | Requires retraining |
|---|---|---|
| Reduce rotation to +-5 deg or remove | bottle, tile, metal_nut (fixed orientation) | Yes |
| Remove RandomVerticalFlip | pill, capsule, screw | Yes |
| Color jitter (brightness/contrast +-10%) | all classes | Yes |

**Why rotation hurts some classes:** the VAE learns to reconstruct the average of all training orientations. For objects with fixed orientation (bottle always upright, tile always flat) this average is still a recognizable object. For objects with variable orientation (hazelnut, screw) the average across rotations becomes a featureless blob. The anomaly map then fires everywhere on the object surface, not just on the defect.

## Results summary

Three runs, same architecture (LATENT_DIM=128, 50 epochs, 256px):

| Class | Baseline | Improved | V2 | Notes |
|---|---|---|---|---|
| hazelnut | 0.169 | 0.378 | **0.380** | map fires on defect |
| metal_nut | 0.172 | 0.243 | **0.257** | map fires on defect |
| bottle | 0.104 | 0.196 | **0.259** | biggest V2 gain — perceptual loss fixes blob |
| leather | 0.077 | 0.195 | **0.206** | map fires correctly |
| cable | 0.121 | 0.202 | **0.200** | partial detection |
| wood | 0.077 | 0.219 | **0.219** | detects large defects |
| pill | 0.084 | 0.116 | **0.148** | fires on text imprint area |
| tile | 0.136 | 0.122 | **0.122** | slight regression (small SSIM window penalizes regular texture) |
| zipper | 0.040 | 0.119 | **0.121** | partial |
| toothbrush | 0.029 | 0.029 | **0.067** | V2 helps despite structural failure |
| transistor | 0.071 | 0.074 | **0.075** | partial |
| capsule | 0.058 | 0.086 | **0.084** | small defects hurt F1 |
| screw | 0.008 | 0.053 | **0.050** | structural failure |
| carpet | 0.031 | 0.031 | **0.031** | structural failure |
| grid | 0.015 | 0.032 | **0.030** | structural failure |
| **avg** | **0.080** | **0.140** | **0.150** | |

**Baseline:** L2 anomaly map only, single-scale SSIM scalar (not spatial).

**Improved:** same trained model, multi-scale per-pixel SSIM (windows 3, 7, 11). The original pytorch-msssim call returns one scalar per image, not a spatial map — the SSIM term contributed nothing to localization. With sliding Gaussian window SSIM, avg F1 improved 75% without retraining.

**V2:** retrained with perceptual loss (VGG16 features, lambda3=0.01) and KL annealing (beta 0→0.5 over 10 epochs). Biggest gains on classes where the baseline produced gray blobs (bottle +32%, toothbrush +131%). Structural failures (carpet, grid, screw) are unchanged — the issue is fundamental to reconstruction-based detection, not the loss function.

Classes marked as structural failure share a pattern: the VAE reconstructs a flat blob instead of the object. This happens when objects vary in orientation across training images (screw, toothbrush — the VAE averages all angles into a blob) or when periodic textures make reconstruction error uniformly high everywhere (grid, carpet — the VAE cannot represent the phase of a repeating pattern). Reducing LATENT_DIM does not fix this. It is a known limitation of reconstruction-based anomaly detection.

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
