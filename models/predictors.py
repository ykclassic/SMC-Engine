import torch
import torch.nn as nn

class SignalValidatorGRU(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, num_layers=2):
        super(SignalValidatorGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :]) # Take the last time step
        return self.sigmoid(out)
