from types import SimpleNamespace

import pandas as pd

from scripts.run_backtest import calculate_metrics, resolve_signal


def _future(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(timestamp, tz="UTC"),
                "open": 100.0,
                "high": high,
                "low": low,
                "close": 100.0,
            }
            for timestamp, high, low in rows
        ]
    )


def _signal(side: str = "LONG") -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        side=side,
        entry=100.0,
        stop_loss=99.0 if side == "LONG" else 101.0,
        take_profit=102.0 if side == "LONG" else 98.0,
    )


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
    assert outcome.r_multiple == 2.1


def test_resolve_short_stop() -> None:
    outcome = resolve_signal(
        _signal("SHORT"),
        _future([("2026-01-01 00:15:00", 101.2, 99.5)]),
    )
    assert outcome.outcome == "LOSS"
    assert outcome.r_multiple == -1.0


def test_same_bar_stop_and_target_is_conservative_loss() -> None:
    outcome = resolve_signal(
        _signal("LONG"),
        _future([("2026-01-01 00:15:00", 102.2, 98.8)]),
    )
    assert outcome.outcome == "LOSS"
    assert outcome.r_multiple == -1.0


def test_unresolved_trade_is_reported_separately() -> None:
    outcome = resolve_signal(
        _signal("LONG"),
        _future([("2026-01-01 00:15:00", 100.5, 99.5)]),
    )
    assert outcome.outcome == "UNRESOLVED"
    assert outcome.r_multiple == 0.0


def test_metrics_calculate_win_rate_expectancy_and_drawdown() -> None:
    outcomes = [
        SimpleNamespace(outcome="WIN", r_multiple=2.0),
        SimpleNamespace(outcome="LOSS", r_multiple=-1.0),
        SimpleNamespace(outcome="LOSS", r_multiple=-1.0),
        SimpleNamespace(outcome="UNRESOLVED", r_multiple=0.0),
    ]
    metrics = calculate_metrics(outcomes)
    assert metrics["signals"] == 4
    assert metrics["completed"] == 3
    assert metrics["wins"] == 1
    assert metrics["losses"] == 2
    assert metrics["unresolved"] == 1
    assert metrics["win_rate"] == 1 / 3
    assert metrics["net_r"] == 0.0
    assert metrics["expectancy_r"] == 0.0
    assert metrics["profit_factor"] == 1.0
    assert metrics["max_drawdown_r"] == 2.0
