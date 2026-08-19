from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import SMCEngine
from data.exchange_api import ExchangeInterface
from utils.config_loader import load_all_configs


@dataclass(frozen=True)
class TradeOutcome:
    signal_timestamp: str
    resolution_timestamp: str | None
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    outcome: str
    r_multiple: float


def _normalise_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame is missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["timestamp", "open", "high", "low", "close"])
    result = result.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return result


def resolve_signal(signal: Any, future_m15: pd.DataFrame) -> TradeOutcome:
    entry = float(signal.entry)
    stop = float(signal.stop_loss)
    target = float(signal.take_profit)
    side = str(signal.side)
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("Signal has non-positive stop distance")

    for row in future_m15.itertuples(index=False):
        high = float(row.high)
        low = float(row.low)
        timestamp = pd.Timestamp(row.timestamp).isoformat()

        if side == "LONG":
            stop_hit = low <= stop
            target_hit = high >= target
        elif side == "SHORT":
            stop_hit = high >= stop
            target_hit = low <= target
        else:
            raise ValueError(f"Unsupported signal side: {side}")

        if stop_hit and target_hit:
            # Intrabar ordering is unknowable from OHLCV. Use the conservative
            # assumption that the stop was reached first.
            return TradeOutcome(
                signal_timestamp=pd.Timestamp(signal.timestamp).isoformat(),
                resolution_timestamp=timestamp,
                side=side,
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                outcome="LOSS",
                r_multiple=-1.0,
            )
        if stop_hit:
            return TradeOutcome(
                signal_timestamp=pd.Timestamp(signal.timestamp).isoformat(),
                resolution_timestamp=timestamp,
                side=side,
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                outcome="LOSS",
                r_multiple=-1.0,
            )
        if target_hit:
            reward = abs(target - entry)
            return TradeOutcome(
                signal_timestamp=pd.Timestamp(signal.timestamp).isoformat(),
                resolution_timestamp=timestamp,
                side=side,
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                outcome="WIN",
                r_multiple=reward / risk,
            )

    return TradeOutcome(
        signal_timestamp=pd.Timestamp(signal.timestamp).isoformat(),
        resolution_timestamp=None,
        side=side,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        outcome="UNRESOLVED",
        r_multiple=0.0,
    )


def calculate_metrics(outcomes: list[TradeOutcome]) -> dict[str, float | int]:
    completed = [item for item in outcomes if item.outcome in {"WIN", "LOSS"}]
    wins = [item for item in completed if item.outcome == "WIN"]
    losses = [item for item in completed if item.outcome == "LOSS"]
    r_values = [item.r_multiple for item in completed]

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in r_values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    gross_profit = sum(item.r_multiple for item in wins)
    gross_loss = abs(sum(item.r_multiple for item in losses))
    completed_count = len(completed)

    return {
        "signals": len(outcomes),
        "completed": completed_count,
        "wins": len(wins),
        "losses": len(losses),
        "unresolved": len(outcomes) - completed_count,
        "win_rate": len(wins) / completed_count if completed_count else 0.0,
        "average_r": sum(r_values) / completed_count if completed_count else 0.0,
        "expectancy_r": sum(r_values) / completed_count if completed_count else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else float("inf") if gross_profit else 0.0,
        "net_r": sum(r_values),
        "max_drawdown_r": max_drawdown,
    }


def run_backtest_session(symbol: str | None = None, output_dir: str = "reports") -> int:
    config = load_all_configs(require_secrets=False)
    exchange = ExchangeInterface(config)
    symbol = symbol or config["trading"]["symbols"][0]
    tf = config["market_data"]["timeframes"]
    limits = config["market_data"]["limits"]

    daily = _normalise_ohlcv(exchange.fetch_ohlcv(symbol, tf["daily"], max(500, limits["daily"])))
    h4 = _normalise_ohlcv(exchange.fetch_ohlcv(symbol, tf["h4"], max(500, limits["h4"])))
    h1 = _normalise_ohlcv(exchange.fetch_ohlcv(symbol, tf["h1"], max(500, limits["h1"])))
    m15 = _normalise_ohlcv(exchange.fetch_ohlcv(symbol, tf["m15"], max(1000, limits["m15"])))
    engine = SMCEngine(config)
    outcomes: list[TradeOutcome] = []

    for i in range(100, len(m15)):
        cutoff = m15.iloc[i]["timestamp"]
        daily_slice = daily[daily["timestamp"] <= cutoff]
        h4_slice = h4[h4["timestamp"] <= cutoff]
        h1_slice = h1[h1["timestamp"] <= cutoff]
        m15_slice = m15.iloc[:i]
        if min(map(len, (daily_slice, h4_slice, h1_slice, m15_slice))) < 50:
            continue

        signal, *_ = engine.process_market(
            daily_slice,
            h4_slice,
            h1_slice,
            m15_slice,
        )
        if signal is None:
            continue

        future = m15[m15["timestamp"] > cutoff]
        outcomes.append(resolve_signal(signal, future))

    metrics = calculate_metrics(outcomes)
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([asdict(item) for item in outcomes]).to_csv(
        report_dir / "backtest_trades.csv", index=False
    )
    report = {
        "engine": "deterministic_smc",
        "ai_enabled": bool(config.get("model", {}).get("enabled", False)),
        "symbol": symbol,
        "timeframe": tf["m15"],
        "assumptions": {
            "entry": "signal entry price",
            "stop": "configured signal stop",
            "target": "configured signal target",
            "same_bar_stop_and_target": "stop_first_conservative",
            "fees_and_slippage": "not included",
            "future_data": "excluded from signal generation",
        },
        "metrics": metrics,
    }
    pd.Series(report).to_json(report_dir / "backtest_baseline.json", indent=2)
    print(f"BACKTEST_SUMMARY {report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic SMC baseline backtest")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    return run_backtest_session(args.symbol, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
