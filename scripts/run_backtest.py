from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.backtest import calculate_performance, resolve_signal
from core.engine import SMCEngine
from data.exchange_api import ExchangeInterface
from utils.config_loader import load_all_configs


def run_backtest_session() -> int:
    config = load_all_configs(require_secrets=False)
    config = copy.deepcopy(config)
    config["model"]["enabled"] = False
    config["model"]["required"] = False

    exchange = ExchangeInterface(config)
    symbol = config["trading"]["symbols"][0]
    tf = config["market_data"]["timeframes"]
    limits = config["market_data"]["limits"]

    daily = exchange.fetch_ohlcv(symbol, tf["daily"], max(500, limits["daily"]))
    h4 = exchange.fetch_ohlcv(symbol, tf["h4"], max(500, limits["h4"]))
    h1 = exchange.fetch_ohlcv(symbol, tf["h1"], max(500, limits["h1"]))
    m15 = exchange.fetch_ohlcv(symbol, tf["m15"], max(1000, limits["m15"]))
    engine = SMCEngine(config)

    signals: list[dict] = []
    outcomes = []

    for i in range(100, len(m15)):
        cutoff = m15.iloc[i]["timestamp"]
        daily_slice = daily[daily["timestamp"] <= cutoff]
        h4_slice = h4[h4["timestamp"] <= cutoff]
        h1_slice = h1[h1["timestamp"] <= cutoff]
        m15_slice = m15.iloc[:i]
        if min(map(len, (daily_slice, h4_slice, h1_slice, m15_slice))) < 50:
            continue

        signal, diagnostic, *_ = engine.process_market(
            daily_slice,
            h4_slice,
            h1_slice,
            m15_slice,
        )
        if signal is None:
            continue

        signal_row = {
            "signal_id": signal.signal_id,
            "timestamp": signal.timestamp,
            "symbol": symbol,
            "side": signal.side,
            "entry": signal.entry,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "ai_confidence": signal.ai_confidence,
            "confluence_score": signal.confluence_score,
            "reason": signal.reason,
        }
        signals.append(signal_row)

        # The resolver receives only candles strictly after the signal timestamp.
        # It therefore cannot inspect the candle that produced the signal.
        outcome = resolve_signal(signal, m15.iloc[i + 1 :])
        outcomes.append(outcome)

    performance = calculate_performance(outcomes)
    performance["symbol"] = symbol
    performance["timeframe"] = tf["m15"]
    performance["strategy"] = "deterministic_smc"
    performance["ai_dependency"] = False
    performance["ai_enabled"] = False
    performance["model_artifacts_required"] = False

    outcome_rows = [outcome.to_dict() for outcome in outcomes]
    results = pd.DataFrame(outcome_rows)
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/backtest_results.csv", index=False)
    Path("reports/backtest_baseline.json").write_text(
        json.dumps(performance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "BACKTEST_SUMMARY "
        f"signals={len(signals)} "
        f"resolved={performance['resolved_trades']} "
        f"wins={performance['wins']} "
        f"losses={performance['losses']} "
        f"unresolved={performance['unresolved']} "
        f"net_r={performance['net_r']:.4f} "
        f"win_rate={performance['win_rate']:.4f} "
        f"expectancy_r={performance['expectancy_r']:.4f} "
        f"max_drawdown_r={performance['max_drawdown_r']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_backtest_session())
