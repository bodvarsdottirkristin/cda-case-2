"""
LSTM autoencoder for physiological time-series windows.

Input shape:
    batch_size x time_steps x n_channels

For our data:
    batch_size x 60 x 3
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, n_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence into latent vector.

        x:
            batch_size x time_steps x n_channels
        """
        _, (h_n, _) = self.encoder_lstm(x)

        # Last layer hidden state
        h_last = h_n[-1]

        z = self.to_latent(h_last)
        return z

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent vector back into sequence.
        """
        hidden = self.from_latent(z)

        # Repeat hidden representation across time.
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.output_layer(decoded)

        return reconstruction

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
            reconstruction, latent
        """
        seq_len = x.shape[1]
        z = self.encode(x)
        reconstruction = self.decode(z, seq_len)
        return reconstruction, z


def build_model(
    n_channels: int,
    latent_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    device: torch.device | str = "cpu",
) -> LSTMAutoencoder:
    model = LSTMAutoencoder(
        n_channels=n_channels,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    return model.to(device)