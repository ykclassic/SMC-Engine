# Nexus-SMC-SignalEngine 🚀

An institutional-grade cryptocurrency signal generator built on **Smart Money Concepts (SMC)**. This engine utilizes multi-timeframe analysis to identify high-probability trade setups based on institutional order flow.

## 🧠 Core Methodology
The system follows a strict 5-step algorithmic pipeline:
1. **Market Structure:** Identifies HH/HL/LL/LH and confirms trend via **Body-Close BOS**.
2. **Liquidity:** Locates Equal Highs/Lows and detects **Stop Hunts (Sweeps)**.
3. **Imbalance:** Detects **Fair Value Gaps (FVG)** to identify market inefficiency.
4. **Supply/Demand:** Pins specific **Order Blocks (OB)** that caused structural breaks.
5. **Top-Down Confluence:** Validates 15m entries only when they align with 4H Macro Bias.

## 📁 Project Structure
The project follows a modular, scalable architecture:
- `core/`: Deterministic SMC logic (Structure, Liquidity, Zones).
- `data/`: Exchange-agnostic API connectors (CCXT).
- `alerts/`: Discord integration and signal formatting.
- `models/`: (Expansion Ready) AI/ML filters (LSTM/Sentiment).
- `config/`: Centralized YAML settings.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker (Optional for containerized deployment)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
