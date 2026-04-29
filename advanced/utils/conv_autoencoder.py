"""Conv1D autoencoder model used for raw time-series windows.

This file contains only the model definition from the current `autoencoding.py`
style pipeline. It does not load data, train the model, run clustering, or make
figures.

Expected input shape for `forward` and `encode`:

    (batch_size, n_channels, window_size)

For the current data pipeline, that means:

    (batch_size, 3, 60)

where the channels are HR, EDA, and TEMP.
"""

from __future__ import annotations

import torch
import torch.nn as nn


LATENT_DIM = 32


def conv_encoder_flat_size(window: int) -> tuple[int, int]:
    """Compute flattened encoder size after the three stride-2 convolutions.

    This mirrors the helper in the original script.
    """
    t = window
    t = (t + 2 * 2 - 5) // 2 + 1  # conv1: kernel=5, stride=2, padding=2
    t = (t + 2 * 2 - 5) // 2 + 1  # conv2: kernel=5, stride=2, padding=2
    t = (t + 2 * 1 - 3) // 2 + 1  # conv3: kernel=3, stride=2, padding=1
    return 128 * t, t


class Conv1DAutoencoder(nn.Module):
    """Convolutional autoencoder for 1D biosignal windows.

    The architecture matches the current `autoencoding.py` model:

    Encoder:
        Conv1d(n_channels -> 32) -> ReLU
        Conv1d(32 -> 64) -> ReLU
        Conv1d(64 -> 128) -> ReLU
        Flatten -> Linear(latent_dim)

    Decoder:
        Linear -> Unflatten
        ConvTranspose1d(128 -> 64) -> ReLU
        ConvTranspose1d(64 -> 32) -> ReLU
        ConvTranspose1d(32 -> n_channels)
    """

    def __init__(self, n_channels: int, latent_dim: int = LATENT_DIM, window: int = 60):
        super().__init__()
        flat_size, t_out = conv_encoder_flat_size(window)
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.window = window
        self.flat_size = flat_size
        self.t_out = t_out

        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat_size),
            nn.Unflatten(1, (128, t_out)),
            nn.ConvTranspose1d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                64,
                32,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                32,
                n_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstructed input and latent vector."""
        z = self.encoder(x)
        recon = self.decoder(z)
        # ConvTranspose1d can add a few extra time steps. Match input length.
        recon = recon[:, :, : x.shape[2]]
        return recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the latent representation."""
        return self.encoder(x)


def build_model(
    n_channels: int = 3,
    latent_dim: int = LATENT_DIM,
    window: int = 60,
    device: torch.device | str | None = None,
) -> Conv1DAutoencoder:
    """Small convenience factory for the autoencoder."""
    model = Conv1DAutoencoder(
        n_channels=n_channels,
        latent_dim=latent_dim,
        window=window,
    )
    if device is not None:
        model = model.to(device)
    return model
