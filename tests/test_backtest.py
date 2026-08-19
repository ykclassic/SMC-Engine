from types import SimpleNamespace

import pandas as pd

from core.backtest import calculate_performance, resolve_signal


def _signal(side: str = "LONG") -> SimpleNamespace:
    return SimpleNamespace(
        signal_id="test-1",
        symbol="BTC/USDT:USDT",
        side=side,
        timestamp="2026-01-01T00:00:00+00:00",
        entry=100.0,
        stop_loss=99.0 if side == "LONG" else 101.0,
        take_profit=102.0 if side == "LONG" else 98.0,
    )


def _future(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["timestamp", "high", "low"])


def test_resolve_long_target_without_future_leakage() -> None:
    outcome = resolve_signal(
        _signal("LONG"),
        _future(
            [
                ("2026-01-01 00:15:00", 100.5, 99.5),
                ("2026-01-01 00:30:00", 102.1, 100.0),
            ]
        ),
    )
    assert outcome.outcome == "WIN"
    assert outcome.r_multiple == 2.0
    assert outcome.resolution_timestamp == "2026-01-01T00:30:00+00:00"


def test_resolve_short_stop_and_target_correctly() -> None:
    outcome = resolve_signal(
        _signal("SHORT"),
        _future(
            [
                ("2026-01-01 00:15:00", 100.5, 99.5),
                ("2026-01-01 00:30:00", 99.0, 98.0),
            ]
        ),
    )
    assert outcome.outcome == "WIN"
    assert outcome.r_multiple == 2.0


def test_same_candle_collision_is_conservative_stop_first() -> None:
    outcome = resolve_signal(
        _signal("LONG"),
        _future([("2026-01-01 00:15:00", 102.5, 98.5)]),
    )
    assert outcome.outcome == "LOSS"
    assert outcome.r_multiple == -1.0


def test_signal_candle_is_excluded() -> None:
    outcome = resolve_signal(
        _signal("LONG"),
        _future([("2026-01-01 00:00:00", 102.5, 98.5)]),
    )
    assert outcome.outcome == "UNRESOLVED"
    assert outcome.r_multiple is None


def test_unresolved_trade_is_explicit() -> None:
    outcome = resolve_signal(
        _signal("LONG"),
        _future([("2026-01-01 00:15:00", 101.0, 99.5)]),
    )
    assert outcome.outcome == "UNRESOLVED"
    assert outcome.resolution_timestamp is None
    assert outcome.r_multiple is None


def test_performance_metrics_are_r_based() -> None:
    outcomes = [
        resolve_signal(
            _signal("LONG"),
            _future([("2026-01-01 00:15:00", 102.0, 100.0)]),
        ),
        resolve_signal(
            _signal("SHORT"),
            _future([("2026-01-01 00:15:00", 101.0, 99.0)]),
        ),
        resolve_signal(
            _signal("LONG"),
            _future([("2026-01-01 00:15:00", 100.5, 99.0)]),
        ),
    ]
    metrics = calculate_performance(outcomes)
    assert metrics["total_signals"] == 3
    assert metrics["resolved_trades"] == 3
    assert metrics["wins"] == 2
    assert metrics["losses"] == 1
    assert metrics["unresolved"] == 0
    assert metrics["win_rate"] == 2 / 3
    assert metrics["net_r"] == 3.0
    assert metrics["expectancy_r"] == 1.0
    assert metrics["profit_factor"] == 4.0
    assert metrics["max_drawdown_r"] == 1.0
