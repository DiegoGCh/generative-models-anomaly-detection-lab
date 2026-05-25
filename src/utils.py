"""
Loss functions y helpers.

Cambios vs. baseline:
  - PerceptualLoss: compara features intermedias de VGG16 preentrenado
    (relu1_2, relu2_2, relu3_3) entre la imagen original y la reconstruccion.
    Fuerza al decoder a producir texturas y colores realistas en lugar de
    promediar en pixeles (blob gris).

  - vae_loss acepta lambda3 y un objeto PerceptualLoss opcional.
    Si lambda3=0.0 o perceptual_loss=None, el comportamiento es identico
    al baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


# ─────────────────────────────────────────────
# Perceptual Loss (VGG16)
# ─────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Loss perceptual usando features intermedias de VGG16 congelado.

    Extrae activaciones en relu1_2, relu2_2, relu3_3 y calcula el MSE
    entre las features del input original y las de la reconstruccion.

    Por que VGG16:
      - Entrenada en ImageNet para clasificacion → sus features capturan
        texturas, bordes y estructuras semanticas de forma compacta.
      - relu1_2 → detalles de borde y color (256x256)
      - relu2_2 → patrones de textura medianos (128x128)
      - relu3_3 → estructuras de forma mas globales (64x64)

    Por que no backpropagar por VGG:
      - VGG esta congelada (requires_grad=False).
      - Los gradientes fluyen de la perdida hasta `recon` (la prediccion
        del decoder) pero no modifican los pesos de VGG.
      - Para el input original x (target fijo) se usa torch.no_grad()
        en el paso de VGG → ahorra memoria de activaciones.

    Normalizacion ImageNet: VGG fue entrenada con mean/std de ImageNet.
    Nuestras imagenes estan en [0,1]. Normalizar antes de pasar por VGG
    es obligatorio para que las features sean comparables a las del
    preentrenamiento.
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models

        # VGG16 preentrenado, solo la parte convolucional
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        except AttributeError:
            # torchvision < 0.13 — API antigua
            vgg = models.vgg16(pretrained=True).features

        for p in vgg.parameters():
            p.requires_grad_(False)

        # Slices secuenciales: cada uno toma la salida del anterior
        # features[:4]   → relu1_2  output [B, 64, H,   W  ]
        # features[4:9]  → relu2_2  output [B, 128, H/2, W/2] (despues de MaxPool)
        # features[9:16] → relu3_3  output [B, 256, H/4, W/4] (despues de MaxPool)
        self.slice1 = nn.Sequential(*list(vgg.children())[:4]).eval()
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9]).eval()
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]).eval()

        # Normalizacion ImageNet
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """
        x:     imagen original  [B, 3, H, W] en [0, 1]  — target, sin grad VGG
        recon: reconstruccion   [B, 3, H, W] en [0, 1]  — prediccion, con grad
        """
        x_n     = (x     - self.mean) / self.std
        recon_n = (recon - self.mean) / self.std

        loss = torch.tensor(0.0, device=x.device)

        # Features del target: no necesitamos gradientes aqui
        with torch.no_grad():
            fx1 = self.slice1(x_n)
            fx2 = self.slice2(fx1)
            fx3 = self.slice3(fx2)

        # Features de la reconstruccion: gradientes fluyen hasta recon
        fy1 = self.slice1(recon_n)
        loss = loss + F.mse_loss(fy1, fx1)

        fy2 = self.slice2(fy1)
        loss = loss + F.mse_loss(fy2, fx2)

        fy3 = self.slice3(fy2)
        loss = loss + F.mse_loss(fy3, fx3)

        return loss / 3.0


# ─────────────────────────────────────────────
# VAE Loss
# ─────────────────────────────────────────────

def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 0.5,
             lambda1: float = 0.5,
             lambda2: float = 0.5,
             lambda3: float = 0.0,
             perceptual_loss: PerceptualLoss = None) -> dict:
    """
    L = L_recon + beta * L_KL

    L_recon = lambda1 * MSE(x, recon)
            + lambda2 * (1 - SSIM(x, recon))
            + lambda3 * PerceptualLoss(x, recon)   [si lambda3 > 0]

    Todos los terminos normalizados por su numero de elementos antes de
    aplicar beta y los lambdas, para que los pesos signifiquen lo mismo
    independientemente de la resolucion o la dimension latente.

    Cuando lambda3=0.0 o perceptual_loss=None el comportamiento es
    identico al baseline.
    """
    B, C, H, W = x.shape

    # Reconstruccion pixel-level
    mse      = F.mse_loss(recon, x, reduction="sum") / (B * C * H * W)
    ssim_val = ssim(recon, x, data_range=1.0, size_average=True)
    recon_loss = lambda1 * mse + lambda2 * (1.0 - ssim_val)

    # Perceptual loss (VGG features)
    if lambda3 > 0.0 and perceptual_loss is not None:
        perc = perceptual_loss(x, recon)
        recon_loss = recon_loss + lambda3 * perc
    else:
        perc = torch.tensor(0.0, device=x.device)

    # KL — forma cerrada, normalizado por batch y latent_dim
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / (B * mu.shape[1])

    total = recon_loss + beta * kl

    return {
        "loss":        total,
        "recon_loss":  recon_loss,
        "kl":          kl,
        "mse":         mse,
        "ssim":        ssim_val,
        "perceptual":  perc,
    }
