from __future__ import annotations

import torch
import torch.nn as nn


class SignalValidatorGRU(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [batch, sequence, features], got {tuple(x.shape)}")
        output, _ = self.gru(x)
        return self.sigmoid(self.fc(output[:, -1, :]))
