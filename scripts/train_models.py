import os
import sys
import torch
import torch.optim as optim

# --- FIX: Add project root to Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.predictors import SignalValidatorGRU
from models.feature_engineering import FeatureEngineer


def train_on_history(df)import os
import sys
import torch
import torch.optim as optim
import pandas as pd

# --- FIX: Add project root to Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.predictors import SignalValidatorGRU
from models.feature_engineering import FeatureEngineer
from core.engine import SMCEngine
from utils.config_loader import load_all_configs
from data.exchange_api import ExchangeInterface


def generate_smc_dataframe(config):
    """
    Fetch real market data and generate SMC-enriched dataframe.
    """
    exchange = ExchangeInterface(config)
    engine = SMCEngine(config)

    symbol = "BTC/USDT:USDT"

    macro_df = exchange.fetch_ohlcv(symbol, config['timeframes']['macro'], limit=300)
    micro_df = exchange.fetch_ohlcv(symbol, config['timeframes']['micro'], limit=300)

    if macro_df.empty or micro_df.empty:
        raise ValueError("❌ Failed to fetch data for training.")

    # Run SMC pipeline (this creates BOS, FVG, OB columns)
    _, macro_processed, micro_processed = engine.process_market(macro_df, micro_df)

    return micro_processed


def train_on_history(df):
    fe = FeatureEngineer()

    # --- SAFETY FIX: Ensure required columns exist ---
    for col in ['bos', 'fvg', 'order_block']:
        if col not in df.columns:
            df[col] = None

    data = fe.prepare_smc_features(df)

    # Target: price increases 1% within next 10 candles
    targets = (df['close'].shift(-10) > df['close'] * 1.01).astype(int).fillna(0).values

    model = SignalValidatorGRU()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    # Convert to tensors
    X = torch.FloatTensor(data).unsqueeze(0)  # (1, seq_len, features)
    y = torch.FloatTensor(targets).unsqueeze(0).unsqueeze(-1)

    model.train()

    for epoch in range(5):  # keep CI fast
        optimizer.zero_grad()
        outputs = model(X)

        # Match last timestep
        loss = criterion(outputs, y[:, -1, :])

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    # Save model
    os.makedirs('models/weights', exist_ok=True)
    torch.save(model.state_dict(), 'models/weights/latest_gru.pth')

    print("✅ Model trained and saved successfully.")


if __name__ == "__main__":
    print("🚀 Starting training pipeline...")

    config = load_all_configs()

    try:
        df = generate_smc_dataframe(config)
    except Exception as e:
        print(f"⚠️ Data fetch failed: {e}")
        print("⚠️ Falling back to synthetic data (NOT ideal for production).")

        # fallback dataset
        df = pd.DataFrame({
            'close': [100 + i for i in range(200)]
        })

    train_on_history(df):
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
