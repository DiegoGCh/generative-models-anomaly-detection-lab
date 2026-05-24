"""
Evaluación: mapa de anomalía, umbralización, F1, morfología, figuras.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
from pathlib import Path


def _gaussian_filter(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Suavizado gaussiano via cv2 (evita scipy DLL issues en Windows)."""
    ksize = int(6 * sigma + 1) | 1   # impar más cercano a 6*sigma
    return cv2.GaussianBlur(arr.astype(np.float32), (ksize, ksize), sigma)


# ─────────────────────────────────────────────
# 1. Mapa de anomalía por imagen
# ─────────────────────────────────────────────

def anomaly_map(model, img: torch.Tensor, device,
                lambda1: float = 0.5,
                lambda2: float = 0.5,
                sigma: float = 4.0) -> np.ndarray:
    """
    Devuelve el mapa de anomalía en scores crudos (sin normalizar).
    La normalización se hace globalmente en evaluate_category para que
    imágenes sanas con error bajo queden en valores bajos vs defectuosas.
    img: tensor [1, C, H, W] en [0,1]
    """
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        recon = model.reconstruct(img)

        # L2 por píxel: [1, H, W]
        l2 = torch.mean((img - recon) ** 2, dim=1, keepdim=True)

        # SSIM disimilitud por píxel: 1 - SSIM map
        ssim_map = ssim(recon, img, data_range=1.0, size_average=False)  # [1, 1, H, W]
        ssim_err = 1.0 - ssim_map

        # Combinar
        score_map = lambda1 * l2 + lambda2 * ssim_err   # [1, 1, H, W]
        score_map = score_map.squeeze().cpu().numpy()    # [H, W]

    # Suavizado gaussiano σ=4 (estándar papers MVTec)
    score_map = _gaussian_filter(score_map, sigma=sigma)

    # NO normalizar aquí — se normaliza globalmente en evaluate_category
    return score_map


# ─────────────────────────────────────────────
# 2. Morfología post-umbral
# ─────────────────────────────────────────────

def apply_morphology(binary_mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apertura (quita manchitas) + Cierre (rellena huecos).
    binary_mask: np.uint8 [H, W] con valores 0/255
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
    closed = cv2.morphologyEx(opened,      cv2.MORPH_CLOSE, kernel)
    return closed


# ─────────────────────────────────────────────
# 3. F1 a nivel de píxel
# ─────────────────────────────────────────────

def pixel_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    pred_mask, gt_mask: arrays binarios [H, W] con valores 0/1.
    """
    pred = pred_mask.astype(bool).flatten()
    gt   = gt_mask.astype(bool).flatten()

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


# ─────────────────────────────────────────────
# 4. Evaluación completa por clase
# ─────────────────────────────────────────────

def evaluate_category(model, test_loader, device, category: str,
                       lambda1: float = 0.5,
                       lambda2: float = 0.5,
                       sigma: float = 4.0,
                       n_thresholds: int = 50,
                       morph_kernel: int = 5) -> dict:
    """
    Barrido de umbral en test → reporta F1 al mejor umbral.
    Devuelve dict con best_f1, best_threshold, f1_curve, thresholds.
    """
    model.eval()
    all_maps  = []
    all_masks = []

    for img, mask, label in test_loader:
        amap = anomaly_map(model, img, device, lambda1, lambda2, sigma)
        all_maps.append(amap)
        all_masks.append(mask.squeeze().numpy())

    # Normalización GLOBAL por clase:
    # imágenes sanas (error bajo) quedan en valores bajos,
    # defectuosas (error alto) quedan en valores altos.
    # Si se normaliza por imagen, hasta las sanas llegan a 1.0.
    global_min = min(m.min() for m in all_maps)
    global_max = max(m.max() for m in all_maps)
    if global_max > global_min:
        all_maps = [(m - global_min) / (global_max - global_min) for m in all_maps]

    thresholds = np.linspace(0, 1, n_thresholds)
    f1_curve   = []

    for t in thresholds:
        f1s = []
        for amap, gt in zip(all_maps, all_masks):
            pred_bin = (amap >= t).astype(np.uint8) * 255
            pred_bin = apply_morphology(pred_bin, morph_kernel)
            pred_bin = (pred_bin > 0).astype(np.uint8)
            gt_bin   = (gt > 0.5).astype(np.uint8)
            f1s.append(pixel_f1(pred_bin, gt_bin))
        f1_curve.append(np.mean(f1s))

    best_idx   = int(np.argmax(f1_curve))
    best_f1    = f1_curve[best_idx]
    best_thresh = thresholds[best_idx]

    print(f"  [{category}] best F1={best_f1:.4f} @ threshold={best_thresh:.3f}")

    return {
        "category":       category,
        "best_f1":        best_f1,
        "best_threshold": best_thresh,
        "f1_curve":       f1_curve,
        "thresholds":     thresholds.tolist(),
        "all_maps":       all_maps,
        "all_masks":      all_masks,
    }


# ─────────────────────────────────────────────
# 5. Figuras: input → recon → anomaly map → mask
# ─────────────────────────────────────────────

def save_figures(model, test_loader, device, category: str,
                 best_threshold: float,
                 out_dir: str,
                 lambda1: float = 0.5,
                 lambda2: float = 0.5,
                 sigma: float = 4.0,
                 morph_kernel: int = 5,
                 n_good: int = 1,
                 n_defect: int = 2):
    """
    Guarda figuras: 1 good + n_defect defectuosas.
    Columnas: input | reconstruction | anomaly map | thresholded mask
    """
    out_path = Path(out_dir) / category
    out_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    good_saved    = 0
    defect_saved  = 0

    for img, mask, label in test_loader:
        is_good = (label.item() == 0)
        if is_good and good_saved >= n_good:
            continue
        if not is_good and defect_saved >= n_defect:
            continue

        amap = anomaly_map(model, img, device, lambda1, lambda2, sigma)

        # Máscara binarizada
        pred_bin = (amap >= best_threshold).astype(np.uint8) * 255
        pred_bin = apply_morphology(pred_bin, morph_kernel)

        # Convertir tensores a numpy
        img_np  = img.squeeze().permute(1, 2, 0).numpy()       # [H,W,3]
        with torch.no_grad():
            recon_t = model.reconstruct(img.to(device))
        recon_np = recon_t.squeeze().permute(1, 2, 0).cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        tag = "good" if is_good else "defect"

        axes[0].imshow(img_np);             axes[0].set_title("Input");           axes[0].axis("off")
        axes[1].imshow(recon_np);           axes[1].set_title("Reconstruction");  axes[1].axis("off")
        axes[2].imshow(amap, cmap="hot");   axes[2].set_title("Anomaly Map");     axes[2].axis("off")
        axes[3].imshow(pred_bin, cmap="gray"); axes[3].set_title("Thresholded"); axes[3].axis("off")

        if not is_good:
            gt_np = mask.squeeze().numpy()
            axes[3].imshow(gt_np, cmap="Greens", alpha=0.4)  # GT overlay

        fname = f"{tag}_{good_saved if is_good else defect_saved}.png"
        fig.suptitle(f"{category} — {tag}", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path / fname, dpi=100)
        plt.close(fig)

        if is_good:
            good_saved += 1
        else:
            defect_saved += 1

        if good_saved >= n_good and defect_saved >= n_defect:
            break

    print(f"  [{category}] Figuras guardadas en {out_path}")


def save_f1_curve(results: dict, out_dir: str):
    """Curva F1 vs umbral para una clase."""
    out_path = Path(out_dir) / results["category"]
    out_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(results["thresholds"], results["f1_curve"], marker=".")
    plt.axvline(results["best_threshold"], color="red", linestyle="--",
                label=f"best={results['best_f1']:.3f} @ t={results['best_threshold']:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Pixel F1")
    plt.title(f"F1 vs Threshold — {results['category']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "f1_curve.png", dpi=100)
    plt.close()
