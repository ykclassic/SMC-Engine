from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TradeOutcome:
    signal_id: str
    signal_timestamp: str
    resolution_timestamp: str | None
    symbol: str
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    outcome: str
    r_multiple: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        return result.tz_localize("UTC")
    return result.tz_convert("UTC")


def resolve_signal(signal: Any, future_bars: pd.DataFrame) -> TradeOutcome:
    """Resolve one signal using only candles strictly after the signal time.

    Collision policy: if both stop and target are touched in the same candle,
    the stop is considered first. OHLC data cannot reconstruct intrabar order,
    so this conservative rule prevents optimistic backtest bias.
    """
    side = str(signal.side).upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError(f"Unsupported signal side: {signal.side!r}")

    entry = float(signal.entry)
    stop = float(signal.stop_loss)
    target = float(signal.take_profit)
    if side == "LONG" and not (stop < entry < target):
        raise ValueError("Invalid LONG entry/SL/TP geometry")
    if side == "SHORT" and not (target < entry < stop):
        raise ValueError("Invalid SHORT entry/SL/TP geometry")

    signal_time = _timestamp(signal.timestamp)
    if future_bars.empty:
        return _unresolved(signal)

    frame = future_bars.copy()
    required_columns = {"timestamp", "high", "low"}
    if not required_columns.issubset(frame.columns):
        raise ValueError("future_bars must contain timestamp, high, and low columns")

    frame["_ts"] = frame["timestamp"].map(_timestamp)
    frame = frame.loc[frame["_ts"] > signal_time].sort_values("_ts")

    risk = abs(entry - stop)
    reward = abs(target - entry)
    target_r = reward / risk

    # Use positional tuples so the private helper column name cannot be
    # rewritten by pandas' namedtuple field sanitisation.
    for timestamp, high, low in frame[["_ts", "high", "low"]].itertuples(
        index=False, name=None
    ):
        timestamp = pd.Timestamp(timestamp)
        high = float(high)
        low = float(low)

        if high < low:
            raise ValueError(
                f"Invalid OHLC row at {timestamp.isoformat()}: high={high} < low={low}"
            )

        if side == "LONG":
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            stop_hit = high >= stop
            target_hit = low <= target

        if stop_hit and target_hit:
            return TradeOutcome(
                signal_id=str(signal.signal_id),
                signal_timestamp=signal_time.isoformat(),
                resolution_timestamp=timestamp.isoformat(),
                symbol=str(signal.symbol),
                side=side,
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                outcome="LOSS",
                r_multiple=-1.0,
            )
        if stop_hit:
            return _resolved(signal, timestamp, "LOSS", -1.0)
        if target_hit:
            return _resolved(signal, timestamp, "WIN", target_r)

    return _unresolved(signal)


def _resolved(
    signal: Any,
    timestamp: pd.Timestamp,
    outcome: str,
    r_multiple: float,
) -> TradeOutcome:
    return TradeOutcome(
        signal_id=str(signal.signal_id),
        signal_timestamp=_timestamp(signal.timestamp).isoformat(),
        resolution_timestamp=timestamp.isoformat(),
        symbol=str(signal.symbol),
        side=str(signal.side).upper(),
        entry=float(signal.entry),
        stop_loss=float(signal.stop_loss),
        take_profit=float(signal.take_profit),
        outcome=outcome,
        r_multiple=float(r_multiple),
    )


def _unresolved(signal: Any) -> TradeOutcome:
    return TradeOutcome(
        signal_id=str(signal.signal_id),
        signal_timestamp=_timestamp(signal.timestamp).isoformat(),
        resolution_timestamp=None,
        symbol=str(signal.symbol),
        side=str(signal.side).upper(),
        entry=float(signal.entry),
        stop_loss=float(signal.stop_loss),
        take_profit=float(signal.take_profit),
        outcome="UNRESOLVED",
        r_multiple=None,
    )


def calculate_performance(outcomes: list[TradeOutcome]) -> dict[str, Any]:
    """Calculate deterministic performance statistics in R-multiple space."""
    resolved = [o for o in outcomes if o.outcome in {"WIN", "LOSS"}]
    wins = [o for o in resolved if o.outcome == "WIN"]
    losses = [o for o in resolved if o.outcome == "LOSS"]
    r_values = [float(o.r_multiple) for o in resolved if o.r_multiple is not None]

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in r_values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    gross_profit = sum(v for v in r_values if v > 0)
    gross_loss = -sum(v for v in r_values if v < 0)
    total = len(outcomes)
    resolved_count = len(resolved)

    return {
        "total_signals": total,
        "resolved_trades": resolved_count,
        "wins": len(wins),
        "losses": len(losses),
        "unresolved": total - resolved_count,
        "win_rate": len(wins) / resolved_count if resolved_count else 0.0,
        "net_r": sum(r_values),
        "expectancy_r": sum(r_values) / resolved_count if resolved_count else 0.0,
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss
            else (float("inf") if gross_profit else 0.0)
        ),
        "max_drawdown_r": max_drawdown,
        "long_trades": sum(o.side == "LONG" for o in resolved),
        "short_trades": sum(o.side == "SHORT" for o in resolved),
    }
