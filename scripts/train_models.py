import os
import sys
import torch
import torch.optim as optim

# --- FIX: Add project root to Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.predictors import SignalValidatorGRU
from models.feature_engineering import FeatureEngineer


def train_on_history(df):
    fe = FeatureEngineer()
    data = fe.prepare_smc_features(df)

    # Target: 1 if price went up 1% in next 10 candles, else 0
    targets = (df['close'].shift(-10) > df['close'] * 1.01).astype(int).values

    model = SignalValidatorGRU()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    # Convert to tensors
    X = torch.FloatTensor(data).unsqueeze(0)  # (1, seq_len, features)
    y = torch.FloatTensor(targets).unsqueeze(0).unsqueeze(-1)

    # Basic training loop
    model.train()
    for epoch in range(5):  # minimal training for CI
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y[:, -1, :])  # match last timestep
        loss.backward()
        optimizer.step()

    # Ensure directory exists
    os.makedirs('models/weights', exist_ok=True)

    torch.save(model.state_dict(), 'models/weights/latest_gru.pth')
    print("✅ Model trained and saved to models/weights/latest_gru.pth")


if __name__ == "__main__":
    # NOTE: In CI, you currently don't pass real data
    # So this will fail unless you load data here

    import pandas as pd

    print("⚠️ No dataset loader implemented. Using dummy data for CI...")

    # Dummy dataset (prevents pipeline crash)
    df = pd.DataFrame({
        'close': [100 + i for i in range(100)]
    })

    train_on_history(df)
