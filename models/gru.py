from __future__ import annotations

import torch
import torch.nn as nn

from models.candidate_context import CONTEXT_COLUMNS, context_vector


class SignalValidatorGRU(nn.Module):
    """GRU classifier with explicit candidate direction/event context.

    The base market sequence remains unchanged. Candidate-level context is
    concatenated to each timestep so LONG and SHORT candidates on the same
    candle cannot be represented as identical inputs.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        context_dim: int = len(CONTEXT_COLUMNS),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or context_dim <= 0:
            raise ValueError("input_dim and context_dim must be positive")
        self.base_input_dim = input_dim
        self.context_dim = context_dim
        self.gru = nn.GRU(
            input_size=input_dim + context_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        direction: torch.Tensor | None = None,
        event_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected [batch, sequence, features], got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.base_input_dim:
            raise ValueError(
                f"Expected {self.base_input_dim} base features, got {x.shape[-1]}"
            )

        batch, steps, _ = x.shape
        if event_context is None:
            if direction is None:
                raise ValueError(
                    "Candidate direction/event context is required for inference"
                )
            if direction.ndim == 1:
                direction = direction[:, None]
            if direction.shape != (batch, 1):
                raise ValueError(
                    f"Direction must have shape [{batch}, 1], got {tuple(direction.shape)}"
                )
            event_context = torch.zeros(
                (batch, self.context_dim),
                dtype=x.dtype,
                device=x.device,
            )
            event_context[:, 0:1] = direction
        else:
            if event_context.ndim != 2 or event_context.shape != (batch, self.context_dim):
                raise ValueError(
                    "event_context must have shape "
                    f"[{batch}, {self.context_dim}]"
                )

        context = event_context[:, None, :].expand(-1, steps, -1)
        model_input = torch.cat([x, context], dim=-1)
        output, _ = self.gru(model_input)
        return self.sigmoid(self.fc(output[:, -1, :]))

    @staticmethod
    def encode_context(
        directions: list[str],
        event_types: list[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if len(directions) != len(event_types):
            raise ValueError("directions and event_types must have equal length")
        vectors = [
            context_vector(direction, event_type)
            for direction, event_type in zip(directions, event_types)
        ]
        return torch.tensor(vectors, dtype=torch.float32, device=device)
