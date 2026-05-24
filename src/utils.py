"""
Loss functions y helpers.
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 0.5,
             lambda1: float = 0.5,
             lambda2: float = 0.5) -> dict:
    """
    L = L_recon + beta * L_KL

    L_recon = lambda1 * MSE(x, recon) + lambda2 * (1 - SSIM(x, recon))

    Ambos términos normalizados por su nº de elementos antes de aplicar beta,
    para que beta signifique lo que uno cree.
    """
    B, C, H, W = x.shape

    # --- Reconstrucción ---
    mse  = F.mse_loss(recon, x, reduction="sum") / (B * C * H * W)
    ssim_val = ssim(recon, x, data_range=1.0, size_average=True)
    recon_loss = lambda1 * mse + lambda2 * (1.0 - ssim_val)

    # --- KL ---
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)  normalizado por latent_dim
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / (B * mu.shape[1])   # normalizar por batch y latent_dim

    total = recon_loss + beta * kl

    return {
        "loss":       total,
        "recon_loss": recon_loss,
        "kl":         kl,
        "mse":        mse,
        "ssim":       ssim_val,
    }
