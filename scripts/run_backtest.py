from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import SMCEngine
from data.exchange_api import ExchangeInterface
from utils.config_loader import load_all_configs


def run_backtest_session() -> int:
    config = load_all_configs(require_secrets=False)
    exchange = ExchangeInterface(config)
    symbol = config["trading"]["symbols"][0]
    tf = config["market_data"]["timeframes"]
    limits = config["market_data"]["limits"]

    daily = exchange.fetch_ohlcv(symbol, tf["daily"], max(500, limits["daily"]))
    h4 = exchange.fetch_ohlcv(symbol, tf["h4"], max(500, limits["h4"]))
    h1 = exchange.fetch_ohlcv(symbol, tf["h1"], max(500, limits["h1"]))
    m15 = exchange.fetch_ohlcv(symbol, tf["m15"], max(1000, limits["m15"]))
    engine = SMCEngine(config)
    results = []

    for i in range(100, len(m15)):
        cutoff = m15.iloc[i]["timestamp"]
        daily_slice = daily[daily["timestamp"] <= cutoff]
        h4_slice = h4[h4["timestamp"] <= cutoff]
        h1_slice = h1[h1["timestamp"] <= cutoff]
        m15_slice = m15.iloc[:i]
        if min(map(len, (daily_slice, h4_slice, h1_slice, m15_slice))) < 50:
            continue
        signal, diagnostic, *_ = engine.process_market(daily_slice, h4_slice, h1_slice, m15_slice)
        if signal:
            results.append({
                "timestamp": signal.timestamp,
                "symbol": symbol,
                "side": signal.side,
                "entry": signal.entry,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "ai_confidence": signal.ai_confidence,
                "confluence_score": signal.confluence_score,
                "reason": signal.reason,
            })

    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv("reports/backtest_results.csv", index=False)
    print(f"BACKTEST_SUMMARY signals={len(results)} rows={len(m15)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_backtest_session())
