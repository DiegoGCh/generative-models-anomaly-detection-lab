"""
VAE Convolucional.
Encoder: 256→128→64→32→16→8, canales 32→64→128→256→512
Decoder: espejo con ConvTranspose2d
Salida con sigmoid → [0, 1]
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, img_size: int = 256):
        super().__init__()
        # 5 bloques conv stride-2: img_size → img_size/32
        self.conv = nn.Sequential(
            self._block(3,   32),   # /2
            self._block(32,  64),   # /2
            self._block(64,  128),  # /2
            self._block(128, 256),  # /2
            self._block(256, 512),  # /2
        )
        # Tamaño espacial después de 5 stride-2: img_size // 32
        spatial = img_size // 32
        flat = 512 * spatial * spatial
        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, img_size: int = 256):
        super().__init__()
        spatial = img_size // 32
        self._spatial = spatial
        flat = 512 * spatial * spatial
        self.fc = nn.Linear(latent_dim, flat)

        # Espejo del encoder: 8→256
        self.deconv = nn.Sequential(
            self._block(512, 256),  # 8→16
            self._block(256, 128),  # 16→32
            self._block(128, 64),   # 32→64
            self._block(64,  32),   # 64→128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128→256
            nn.Sigmoid(),           # [0, 1]
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, self._spatial, self._spatial)
        return self.deconv(h)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, img_size: int = 256):
        super().__init__()
        self.encoder = Encoder(latent_dim, img_size)
        self.decoder = Decoder(latent_dim, img_size)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """z = mu + sigma * eps,  eps ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def reconstruct(self, x):
        """Inferencia determinística (usa mu, no muestrea)."""
        mu, _ = self.encoder(x)
        return self.decoder(mu)
