import torch
import torch.nn as nn

class SignalValidatorGRU(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        """
        GRU-based neural network architecture for validating trading signals.
        Outputs a probability score between 0 and 1 indicating signal success likelihood.
        """
        super(SignalValidatorGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GRU and fully connected classification head.
        """
        # Handle 2D inputs by adding a sequence dimension if necessary
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        out, _ = self.gru(x)
        
        # Extract output from the final time step
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out
