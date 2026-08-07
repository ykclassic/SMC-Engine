import torch
import torch.nn as nn

class SignalValidatorGRU(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 1, output_dim: int = 1):
        """
        GRU-based neural network architecture matching the pre-trained weights schema.
        """
        super(SignalValidatorGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Single linear layer and activation matching checkpoint keys: fc.weight, fc.bias
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GRU and linear head.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        out, _ = self.gru(x)
        
        # Extract output from the final time step
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out
