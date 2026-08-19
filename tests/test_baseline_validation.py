from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from core.backtest import TradeOutcome
from scripts.run_baseline_validation import (
    EVALUATION_START_INDEX,
    MIN_ENGINE_CONTEXT_ROWS,
    VALIDATION_LIMITS,
    _canonical_signal_id,
    _canonical_outcome,
    _causal_slice,
    _validation_limits,
    build_report,
)


def _config() -> dict:
    return {
        "trading": {
            "exchange": "bitget",
            "fallback_exchange_enabled": False,
            "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        },
        "market_data": {
            "timeframes": {"m15": "15m"},
            "limits": {"daily": 300, "h4": 400, "h1": 400, "m15": 500},
        },
    }


def _outcome(
    signal_id: str,
    symbol: str = "BTC/USDT:USDT",
    side: str = "LONG",
    timestamp: str = "2026-01-01T00:00:00+00:00",
    r_multiple: float = 2.0,
    outcome: str = "WIN",
) -> TradeOutcome:
    return TradeOutcome(
        signal_id=signal_id,
        signal_timestamp=timestamp,
        resolution_timestamp="2026-01-01T00:15:00+00:00" if outcome != "UNRESOLVED" else None,
        symbol=symbol,
        side=side,
        entry=100.0,
        stop_loss=99.0 if side == "LONG" else 101.0,
        take_profit=102.0 if side == "LONG" else 98.0,
        outcome=outcome,
        r_multiple=None if outcome == "UNRESOLVED" else r_multiple,
    )


def test_canonical_signal_id_is_stable() -> None:
    first = _canonical_signal_id("BTC/USDT:USDT", "2026-01-01T00:00:00+00:00", "LONG")
    second = _canonical_signal_id("BTC/USDT:USDT", "2026-01-01T00:00:00+00:00", "LONG")
    assert first == second
    assert first.startswith("baseline-")


def test_canonical_outcome_removes_uuid_nondeterminism() -> None:
    original = _outcome("random-uuid")
    canonical = _canonical_outcome(original)
    assert canonical.signal_id == _canonical_signal_id(
        original.symbol,
        original.signal_timestamp,
        original.side,
    )
    assert canonical.outcome == original.outcome
    assert canonical.r_multiple == original.r_multiple


def test_validation_limits_never_exceed_configured_limits() -> None:
    limits = _validation_limits(_config())
    assert limits == {"daily": 120, "h4": 160, "h1": 200, "m15": 240}
    assert all(limits[key] <= _config()["market_data"]["limits"][key] for key in limits)


def test_causal_slice_is_bounded_and_excludes_future_rows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC"),
            "value": range(10),
        }
    )
    cutoff = frame.iloc[7]["timestamp"]
    sliced = _causal_slice(frame, cutoff, 3)

    assert len(sliced) == 3
    assert sliced["value"].tolist() == [5, 6, 7]
    assert sliced["timestamp"].max() <= cutoff
    assert sliced["value"].max() < 8


def test_phase4_runtime_policy_is_explicit_and_causal() -> None:
    assert EVALUATION_START_INDEX == 100
    assert MIN_ENGINE_CONTEXT_ROWS == 50
    assert VALIDATION_LIMITS["m15"] == 240
    assert VALIDATION_LIMITS["m15"] >= EVALUATION_START_INDEX


def test_phase4_report_contains_provenance_and_breakdowns() -> None:
    outcomes = [
        _outcome("uuid-a", "BTC/USDT:USDT", "LONG"),
        _outcome(
            "uuid-b",
            "BTC/USDT:USDT",
            "SHORT",
            "2026-02-01T00:00:00+00:00",
            -1.0,
            "LOSS",
        ),
        _outcome(
            "uuid-c",
            "ETH/USDT:USDT",
            "LONG",
            "2026-02-01T00:00:00+00:00",
            2.0,
            "WIN",
        ),
    ]
    report = build_report(
        _config(),
        outcomes,
        [{"signal_id": "x"}] * 3,
        {"BTC/USDT:USDT": {"m15_start": "2026-01-01"}},
    )

    assert report["schema_version"] == "phase4-baseline-v1"
    assert report["provenance"]["exchange"] == "bitget"
    assert report["provenance"]["ai_enabled"] is False
    assert report["provenance"]["ai_dependency"] is False
    assert report["provenance"]["model_artifacts_required"] is False
    assert report["provenance"]["validation_policy"] == "bounded_causal_window"
    assert report["provenance"]["validation_limits"] == VALIDATION_LIMITS
    assert set(report["by_symbol"]) == {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    assert set(report["by_side"]) == {"LONG", "SHORT"}
    assert set(report["by_period"]) == {"2026-01", "2026-02"}
    assert report["aggregate"]["total_signals"] == 3


def test_phase4_report_is_json_serializable() -> None:
    report = build_report(
        _config(),
        [_outcome("uuid")],
        [],
        {},
    )
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == "phase4-baseline-v1"


def test_phase4_does_not_modify_phase3_artifact(tmp_path: Path) -> None:
    phase3 = tmp_path / "backtest_baseline.json"
    phase3.write_text('{"net_r": 18.0}\n', encoding="utf-8")
    before = phase3.read_bytes()

    from scripts import run_baseline_validation as validation

    assert validation.ARTIFACT_PATH.name == "phase4_baseline_validation.json"
    assert validation.RESULTS_PATH.name == "phase4_baseline_validation.csv"
    assert phase3.read_bytes() == before


def test_phase4_data_window_timestamps_are_iso_compatible() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"])}
    )
    assert frame["timestamp"].min().isoformat() == "2026-01-01T00:00:00+00:00"
