"""
Loop de entrenamiento del VAE para una clase MVTec.
"""

import torch
from tqdm import tqdm
from src.utils import vae_loss


def train_one_epoch(model, loader, optimizer, device, beta, lambda1, lambda2):
    model.train()
    totals = {"loss": 0, "recon_loss": 0, "kl": 0}

    for x in tqdm(loader, leave=False):
        x = x.to(device)
        optimizer.zero_grad()

        recon, mu, logvar = model(x)
        metrics = vae_loss(recon, x, mu, logvar, beta=beta, lambda1=lambda1, lambda2=lambda2)

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
          lambda2: float = 0.5):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(model, train_loader, optimizer, device, beta, lambda1, lambda2)
        history.append(metrics)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"loss={metrics['loss']:.4f} | "
                  f"recon={metrics['recon_loss']:.4f} | "
                  f"kl={metrics['kl']:.4f}")

    return history
