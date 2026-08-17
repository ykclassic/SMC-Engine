# SMC Engine

Production-oriented SMC/ICT signal engine for Bitget USDT-margined perpetuals. The engine is deliberately **fail-closed**: it will not broadcast a signal unless market data, deterministic MTF confluence, risk levels, and a validated AI model all pass their gates.

## Pipeline

```text
Bitget OHLCV
  -> closed-candle validation
  -> Daily / H4 / H1 / M15 structure
  -> BOS / CHoCH / liquidity / FVG / order blocks
  -> deterministic MTF confluence
  -> calibrated GRU validation
  -> ATR risk levels (1.5 ATR SL, configurable RR)
  -> canonical TradingSignal contract
  -> SQLite audit journal
  -> Discord delivery with retry/rate-limit handling
```

## Important production rule

The repository no longer falls back to synthetic data. The previous synthetic-trained GRU artifact was removed because it could not provide trustworthy live validation. A live deployment requires these three generated artifacts:

- `models/weights/latest_gru.pth`
- `models/weights/feature_scaler.joblib`
- `models/weights/model_metadata.json`

Run **AI - Model Retraining Pipeline** manually after merging the production changes, or wait for its scheduled run. The pipeline trains only from real exchange data and promotes a model only after chronological out-of-sample validation.

## Configuration

All symbols, timeframes, thresholds, model paths, risk parameters, notification retry settings, and journal paths are centralized in `config/settings.yaml`.

Bitget symbols use canonical CCXT futures identifiers such as `BTC/USDT:USDT`.

## Local setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-live.txt
```

Set the required environment variables:

```text
EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=
EXCHANGE_PASSPHRASE=
DISCORD_WEBHOOK_URL=
```

Then run:

```bash
python main.py
```

Every scan is persisted to `data/signal_journal.sqlite3`, including rejected setups and their reason. A no-signal cycle is a valid outcome; data/model/delivery failures return a non-zero process status.

## Model training

```bash
pip install -r requirements-train.txt
python scripts/train_models.py
python scripts/validate_model.py
```

Training uses a time-ordered train/validation/test split and a trading-aligned target: whether TP is reached before SL within the configured horizon. The scaler is fitted only on the training partition and persisted for identical live inference.

## Backtesting

```bash
python scripts/run_backtest.py
```

The replay uses only candles available at each simulated M15 timestamp; future candles are not supplied to the SMC engine.

## Testing

```bash
pytest tests/ -q
```

## GitHub Actions

- `live_signal_monitor.yml` — 15-minute live scan.
- `ai_retrain.yml` — weekly real-data training plus manual dispatch.
- `lint.yml` — Python 3.11 syntax checks and tests.

The live workflow intentionally fails when validated model artifacts are absent. This prevents an unvalidated model from silently returning misleading confidence values.
