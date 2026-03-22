# Nexus-SMC-SignalEngine 🚀

The **Nexus-SMC-SignalEngine** is a hybrid algorithmic trading suite developed by **TechSolute**. It combines the deterministic precision of **Smart Money Concepts (SMC)** with the predictive power of **Recurrent Neural Networks (GRU)** and **Sentiment Analysis**.

## 🏗 System Architecture

### 1. The Deterministic Layer (`core/`)
Uses high-frequency price action to identify:
* **Market Structure:** BOS (Break of Structure) and ChoCh (Change of Character).
* **Liquidity:** EQH/EQL sweeps and stop-hunts.
* **Supply & Demand:** Unmitigated Order Blocks (OB) and Fair Value Gaps (FVG).

### 2. The Probabilistic Layer (`models/`)
Acts as a "Confirmation Gate" for every signal:
* **GRU Predictor:** Analyzes the last 50 candles to predict the probability of success.
* **FinBERT Sentiment:** Scans financial news to ensure the signal aligns with market mood.

### 3. The Delivery Layer (`alerts/`)
Formatted Discord embeds featuring dynamic Entry, SL, TP, and Risk/Reward calculations.

## 🚀 Deployment & CI/CD
* **Continuous Integration:** GitHub Actions check code quality and run unit tests on every push.
* **Continuous Training:** The AI model is automatically retrained every Saturday on fresh market data.
* **Dockerized:** Deployable to any cloud environment (Render, Railway, VPS).

## 🛠 Setup
1. Define your API keys in `.env`.
2. Customize `config/settings.yaml`.
3. Run `python main.py` for live monitoring or `scripts/train_models.py` to initialize the AI.

---
*Developed for the TechSolute Nexus Intelligence Suite.*
