"""
Loop de entrenamiento del VAE para una clase MVTec.

Cambios vs. baseline:
  - KL Annealing: beta sube linealmente de 0 a su valor objetivo durante
    los primeros beta_warmup epochs. El decoder aprende a reconstruir
    primero (sin presion KL) y el espacio latente se regulariza despues.

  - Soporte para perceptual_loss: se pasa como argumento opcional.
    Si es None o lambda3=0, comportamiento identico al baseline.
"""

import torch
from tqdm import tqdm
from src.utils import vae_loss


def train_one_epoch(model, loader, optimizer, device,
                    beta, lambda1, lambda2, lambda3=0.0,
                    perceptual_loss=None):
    model.train()
    totals = {"loss": 0, "recon_loss": 0, "kl": 0, "perceptual": 0}

    for x in tqdm(loader, leave=False):
        x = x.to(device)
        optimizer.zero_grad()

        recon, mu, logvar = model(x)
        metrics = vae_loss(
            recon, x, mu, logvar,
            beta=beta,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            perceptual_loss=perceptual_loss,
        )

        metrics["loss"].backward()
        optimizer.step()

        for k in totals:
            totals[k] += metrics[k].item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def train(model, train_loader, device,
          epochs: int = 50,
          lr: float = 1e-4,
          beta: float = 0.5,
          lambda1: float = 0.5,
          lambda2: float = 0.5,
          lambda3: float = 0.0,
          beta_warmup: int = 10,
          perceptual_loss=None):
    """
    beta_warmup: numero de epochs para rampar beta de 0 a su valor final.
                 0 = sin annealing (comportamiento original).

    perceptual_loss: instancia de PerceptualLoss o None.
                     Se crea en main.py si lambda3 > 0.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):

        # KL annealing: beta sube linealmente de 0 a beta_target
        # en los primeros beta_warmup epochs.
        # El decoder aprende reconstruccion limpia antes de que el KL
        # presione hacia la prior. Reduce riesgo de posterior collapse
        # y mejora calidad de reconstruccion inicial.
        if beta_warmup > 0:
            current_beta = beta * min(1.0, epoch / beta_warmup)
        else:
            current_beta = beta

        metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            current_beta, lambda1, lambda2, lambda3,
            perceptual_loss,
        )
        history.append(metrics)

        if epoch % 5 == 0 or epoch == 1:
            perc_str = f" | perc={metrics['perceptual']:.4f}" if lambda3 > 0 else ""
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"loss={metrics['loss']:.4f} | "
                  f"recon={metrics['recon_loss']:.4f} | "
                  f"kl={metrics['kl']:.4f}"
                  f"{perc_str} | beta={current_beta:.3f}")

    return history
