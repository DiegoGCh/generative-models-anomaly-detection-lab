"""
Evaluación: mapa de anomalía, umbralización, F1, morfología, figuras.

Mejoras sobre el baseline:
  - Multi-scale per-pixel SSIM: ventanas 3, 7, 11 usando Gaussian sliding window.
    El SSIM original de pytorch-msssim (size_average=False) devuelve un escalar
    por imagen, no un mapa por píxel. Aquí se implementa el mapa local real.
    Ventanas pequeñas detectan errores de textura fina; ventanas grandes capturan
    diferencias estructurales. El promedio de las tres escalas es más robusto.

  - Feature-space map: compara features intermedias del encoder para
    input vs. reconstrucción. El encoder pasa ambas imágenes y se calcula
    la diferencia L2 en espacio de features por capa, upsampled a resolución
    original y promediada. Captura diferencias que el error de píxel no detecta
    (ej. textura de superficie, estructura de bordes).

  - Ambos mapas se normalizan globalmente por separado antes de combinar.
    feat_alpha controla el peso del mapa de features (default 0.3).
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pytorch_msssim import ssim as _pytorch_ssim
import matplotlib.pyplot as plt
from pathlib import Path


def _gaussian_filter(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Suavizado gaussiano via cv2 (evita scipy DLL issues en Windows)."""
    ksize = int(6 * sigma + 1) | 1   # impar más cercano a 6*sigma
    return cv2.GaussianBlur(arr.astype(np.float32), (ksize, ksize), sigma)


def _local_ssim_map(x: torch.Tensor, y: torch.Tensor,
                    win_size: int = 11, sigma: float = 1.5,
                    data_range: float = 1.0) -> torch.Tensor:
    """
    Mapa SSIM por píxel usando ventana Gaussiana deslizante.

    pytorch-msssim con size_average=False devuelve un escalar por imagen,
    no un mapa espacial. Esta función implementa el mapa real [B, 1, H, W]
    usando convolución separable Gaussiana con padding 'same'.

    La estabilidad numérica se garantiza con las constantes C1, C2 (SSIM paper)
    y clamp de varianzas (pueden ser ligeramente negativas por precisión float).
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    _, C, _, _ = x.shape
    pad = win_size // 2

    # Kernel Gaussiano 1D separable
    coords = torch.arange(win_size, dtype=x.dtype, device=x.device) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    win_row = g.view(1, 1, 1, win_size).expand(C, 1, 1, win_size).contiguous()
    win_col = g.view(1, 1, win_size, 1).expand(C, 1, win_size, 1).contiguous()

    def smooth(t):
        t = F.conv2d(t, win_row, padding=(0, pad), groups=C)
        t = F.conv2d(t, win_col, padding=(pad, 0), groups=C)
        return t

    mu_x  = smooth(x)
    mu_y  = smooth(y)
    mu_xx = smooth(x * x)
    mu_yy = smooth(y * y)
    mu_xy = smooth(x * y)

    sigma_x  = (mu_xx - mu_x.pow(2)).clamp(min=0.0)
    sigma_y  = (mu_yy - mu_y.pow(2)).clamp(min=0.0)
    sigma_xy = mu_xy - mu_x * mu_y

    num   = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denom = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)

    ssim_map = num / denom                         # [B, C, H, W]
    return ssim_map.mean(dim=1, keepdim=True)      # [B, 1, H, W]


# ─────────────────────────────────────────────
# 1. Mapa de anomalía pixel-level (multi-scale SSIM)
# ─────────────────────────────────────────────

def anomaly_map(model, img: torch.Tensor, device,
                lambda1: float = 0.5,
                lambda2: float = 0.5,
                sigma: float = 4.0) -> np.ndarray:
    """
    Mapa de anomalía pixel-level con multi-scale SSIM per-pixel.

    score = lambda1 * L2(x, x_hat) + lambda2 * mean_w(1 - SSIM_w(x, x_hat))
    donde w in {3, 7, 11} (ventanas Gaussianas).

    Devuelve scores crudos [H, W] sin normalizar.
    La normalización global se hace en evaluate_category.
    """
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        recon = model.reconstruct(img)

        # L2 por píxel: [1, 1, H, W]
        l2 = torch.mean((img - recon) ** 2, dim=1, keepdim=True)

        # Multi-scale SSIM per-pixel: promedio sobre ventanas 3, 7, 11
        ssim_err = torch.zeros_like(l2)
        for win_size in [3, 7, 11]:
            s = _local_ssim_map(recon, img, win_size=win_size, sigma=1.5, data_range=1.0)
            ssim_err = ssim_err + (1.0 - s)
        ssim_err = ssim_err / 3.0

        score_map = (lambda1 * l2 + lambda2 * ssim_err).squeeze().cpu().numpy()

    score_map = _gaussian_filter(score_map, sigma=sigma)
    return score_map


# ─────────────────────────────────────────────
# 2. Mapa de anomalía en espacio de features
# ─────────────────────────────────────────────

def feature_diff_map(model, img: torch.Tensor, device,
                     sigma: float = 4.0) -> np.ndarray:
    """
    Feature-space anomaly map.

    El encoder procesa tanto el input original como la reconstrucción.
    Para cada una de las 5 capas del encoder se calcula el error L2 entre
    las activaciones del input vs. las de la reconstrucción. Cada mapa de error
    se upsamplea a resolución original y se promedia sobre las capas.

    Intuición: si la reconstrucción perdió información de un defecto, el encoder
    producirá activaciones distintas para input vs. reconstrucción en esa región.
    Capas tempranas (alta resolución) localizan defectos finos; capas tardías
    capturan diferencias estructurales más globales.

    Devuelve scores crudos [H, W] sin normalizar.
    """
    model.eval()
    feat_in  = {}
    feat_rec = {}

    def make_hook(store, key):
        def hook(m, inp, out):
            store[key] = out.detach()
        return hook

    with torch.no_grad():
        img = img.to(device)

        # Capturar features del input
        hooks = [
            block.register_forward_hook(make_hook(feat_in, i))
            for i, block in enumerate(model.encoder.conv)
        ]
        mu, _ = model.encoder(img)
        for h in hooks:
            h.remove()

        # Reconstrucción determinística desde mu
        recon = model.decoder(mu)

        # Capturar features de la reconstrucción
        hooks = [
            block.register_forward_hook(make_hook(feat_rec, i))
            for i, block in enumerate(model.encoder.conv)
        ]
        model.encoder(recon)
        for h in hooks:
            h.remove()

        # L2 diff por capa, upsampled a resolución original, promediado
        H, W = img.shape[2], img.shape[3]
        feat_map = torch.zeros(1, 1, H, W, device=device)
        n_layers = len(model.encoder.conv)

        for i in range(n_layers):
            diff = (feat_in[i] - feat_rec[i]).pow(2).mean(dim=1, keepdim=True)
            diff = F.interpolate(diff, size=(H, W), mode='bilinear', align_corners=False)
            feat_map = feat_map + diff

        feat_map = (feat_map / n_layers).squeeze().cpu().numpy()

    feat_map = _gaussian_filter(feat_map, sigma=sigma)
    return feat_map


# ─────────────────────────────────────────────
# 3. Morfología post-umbral
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
# 4. F1 a nivel de píxel
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
# 5. Utilidad: normalización global
# ─────────────────────────────────────────────

def _global_norm(maps: list) -> tuple:
    """
    Normaliza lista de arrays numpy a [0, 1] usando min/max globales.
    Devuelve (normed_maps, gmin, gmax).
    """
    gmin = float(min(m.min() for m in maps))
    gmax = float(max(m.max() for m in maps))
    if gmax > gmin:
        normed = [(m - gmin) / (gmax - gmin) for m in maps]
    else:
        normed = [np.zeros_like(m) for m in maps]
    return normed, gmin, gmax


# ─────────────────────────────────────────────
# 6. Evaluación completa por clase
# ─────────────────────────────────────────────

def evaluate_category(model, test_loader, device, category: str,
                       lambda1: float = 0.5,
                       lambda2: float = 0.5,
                       sigma: float = 4.0,
                       n_thresholds: int = 50,
                       morph_kernel: int = 5,
                       feat_alpha: float = 0.0) -> dict:
    """
    Barrido de umbral en test → reporta F1 al mejor umbral.

    Pipeline:
      1. Calcula pixel map (multi-scale per-pixel SSIM + L2) por imagen.
      2. Normaliza globalmente.
      3. Barre 50 umbrales, reporta mejor F1.

    feat_alpha: peso del feature-space map en el combinado.
    Sweep sobre clases representativas mostró que feat_alpha=0.0 (pixel only)
    da el mejor promedio. El feature map ayuda a wood/capsule pero penaliza
    hazelnut/leather — efecto neto negativo en el promedio. Se mantiene el
    parámetro para experimentación futura.
    """
    model.eval()
    all_pixel_maps = []
    all_feat_maps  = []
    all_masks      = []

    for img, mask, label in test_loader:
        all_pixel_maps.append(anomaly_map(model, img, device, lambda1, lambda2, sigma))
        if feat_alpha > 0.0:
            all_feat_maps.append(feature_diff_map(model, img, device, sigma))
        all_masks.append(mask.squeeze().numpy())

    # Normalización global del pixel map
    pixel_norm, pmin, pmax = _global_norm(all_pixel_maps)

    if feat_alpha > 0.0:
        feat_norm, fmin, fmax = _global_norm(all_feat_maps)
        all_maps = [
            (1.0 - feat_alpha) * p + feat_alpha * f
            for p, f in zip(pixel_norm, feat_norm)
        ]
    else:
        feat_norm, fmin, fmax = [], 0.0, 1.0
        all_maps = pixel_norm

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

    best_idx    = int(np.argmax(f1_curve))
    best_f1     = f1_curve[best_idx]
    best_thresh = thresholds[best_idx]

    print(f"  [{category}] best F1={best_f1:.4f} @ threshold={best_thresh:.3f}")

    return {
        "category":       category,
        "best_f1":        best_f1,
        "best_threshold": best_thresh,
        "f1_curve":       f1_curve,
        "thresholds":     thresholds.tolist(),
        "all_maps":       all_maps,      # mapa combinado, ya normalizado a [0, 1]
        "all_masks":      all_masks,
        "global_min":     0.0,           # combinado ya está en [0, 1]
        "global_max":     1.0,
        "pixel_min":      pmin,
        "pixel_max":      pmax,
        "feat_min":       fmin,
        "feat_max":       fmax,
        "feat_alpha":     feat_alpha,
    }


# ─────────────────────────────────────────────
# 7. Figuras: input → recon → anomaly map → mask
# ─────────────────────────────────────────────

def save_figures(model, test_loader, device, category: str,
                 best_threshold: float,
                 out_dir: str,
                 global_min: float = 0.0,
                 global_max: float = 1.0,
                 pixel_min: float = 0.0,
                 pixel_max: float = 1.0,
                 feat_min: float = 0.0,
                 feat_max: float = 1.0,
                 feat_alpha: float = 0.3,
                 lambda1: float = 0.5,
                 lambda2: float = 0.5,
                 sigma: float = 4.0,
                 morph_kernel: int = 5,
                 n_good: int = 1,
                 n_defect: int = 5):
    """
    Guarda figuras: 1 good + n_defect defectuosas.
    Columnas: input | reconstruction | anomaly map | thresholded mask

    Replica exactamente la normalización de evaluate_category:
    pixel y feature maps se normalizan con los mismos stats globales,
    luego se combinan con feat_alpha.
    """
    out_path = Path(out_dir) / category
    out_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    good_saved   = 0
    defect_saved = 0

    for img, mask, label in test_loader:
        is_good = (label.item() == 0)
        if is_good and good_saved >= n_good:
            continue
        if not is_good and defect_saved >= n_defect:
            continue

        # Pixel map normalizado con stats globales
        pmap = anomaly_map(model, img, device, lambda1, lambda2, sigma)
        if pixel_max > pixel_min:
            pmap = np.clip((pmap - pixel_min) / (pixel_max - pixel_min), 0.0, 1.0)

        if feat_alpha > 0.0:
            fmap = feature_diff_map(model, img, device, sigma)
            if feat_max > feat_min:
                fmap = np.clip((fmap - feat_min) / (feat_max - feat_min), 0.0, 1.0)
            amap = (1.0 - feat_alpha) * pmap + feat_alpha * fmap
        else:
            amap = pmap

        # Máscara binarizada con threshold calibrado en el mapa combinado
        pred_bin = (amap >= best_threshold).astype(np.uint8) * 255
        pred_bin = apply_morphology(pred_bin, morph_kernel)

        # Tensores → numpy para visualización
        img_np = img.squeeze().permute(1, 2, 0).numpy()
        with torch.no_grad():
            recon_t = model.reconstruct(img.to(device))
        recon_np = recon_t.squeeze().permute(1, 2, 0).cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        tag = "good" if is_good else "defect"

        axes[0].imshow(img_np);              axes[0].set_title("Input");          axes[0].axis("off")
        axes[1].imshow(recon_np);            axes[1].set_title("Reconstruction"); axes[1].axis("off")
        axes[2].imshow(amap, cmap="hot");    axes[2].set_title("Anomaly Map");    axes[2].axis("off")
        axes[3].imshow(pred_bin, cmap="gray"); axes[3].set_title("Predicted Mask"); axes[3].axis("off")

        if not is_good:
            gt_np = mask.squeeze().numpy()

            # Overlay RGBA: solo pixeles GT=1 reciben color verde solido.
            # GT=0 queda completamente transparente → no agrega tinte al fondo.
            gt_rgba = np.zeros((*gt_np.shape, 4), dtype=np.float32)
            gt_rgba[gt_np > 0.5] = [0.0, 0.78, 0.2, 0.6]  # verde semitransparente → overlap visible
            axes[3].imshow(gt_rgba)

            # Leyenda explicita: blanco = modelo, verde = ground truth
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="white",           edgecolor="gray", label="Model prediction"),
                Patch(facecolor=(0.0, 0.85, 0.2),  edgecolor="gray", label="Ground truth (GT)"),
            ]
            axes[3].legend(
                handles=legend_elements,
                loc="lower left",
                fontsize=6,
                framealpha=0.8,
                handlelength=1.0,
                handleheight=0.8,
            )

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
