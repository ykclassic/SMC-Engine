from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any, Callable

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.backtest import TradeOutcome, calculate_performance, resolve_signal
from core.engine import SMCEngine
from data.exchange_api import ExchangeInterface
from utils.config_loader import load_all_configs

REPORT_DIR = Path("reports")
ARTIFACT_PATH = REPORT_DIR / "phase4_baseline_validation.json"
RESULTS_PATH = REPORT_DIR / "phase4_baseline_validation.csv"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_signal_id(symbol: str, timestamp: str, side: str) -> str:
    raw = f"{symbol}|{timestamp}|{side}".encode("utf-8")
    return "baseline-" + hashlib.sha256(raw).hexdigest()[:20]


def _canonical_outcome(outcome: TradeOutcome) -> TradeOutcome:
    return replace(
        outcome,
        signal_id=_canonical_signal_id(
            outcome.symbol,
            outcome.signal_timestamp,
            outcome.side,
        ),
    )


def _validation_limits(config: dict[str, Any]) -> dict[str, int]:
    configured = config["market_data"]["limits"]
    return {
        "daily": max(500, min(int(configured["daily"]), 1000)),
        "h4": max(500, min(int(configured["h4"]), 1000)),
        "h1": max(500, min(int(configured["h1"]), 1000)),
        "m15": max(1000, min(int(configured["m15"]), 1000)),
    }


def _run_symbol(
    config: dict[str, Any],
    exchange: ExchangeInterface,
    symbol: str,
    engine_factory: Callable[[dict[str, Any]], SMCEngine] = SMCEngine,
) -> tuple[list[TradeOutcome], list[dict[str, Any]], dict[str, Any]]:
    timeframes = config["market_data"]["timeframes"]
    limits = _validation_limits(config)

    daily = exchange.fetch_ohlcv(symbol, timeframes["daily"], limits["daily"])
    h4 = exchange.fetch_ohlcv(symbol, timeframes["h4"], limits["h4"])
    h1 = exchange.fetch_ohlcv(symbol, timeframes["h1"], limits["h1"])
    m15 = exchange.fetch_ohlcv(symbol, timeframes["m15"], limits["m15"])

    engine = engine_factory(config)
    outcomes: list[TradeOutcome] = []
    signal_rows: list[dict[str, Any]] = []

    for index in range(100, len(m15)):
        cutoff = m15.iloc[index]["timestamp"]
        daily_slice = daily[daily["timestamp"] <= cutoff]
        h4_slice = h4[h4["timestamp"] <= cutoff]
        h1_slice = h1[h1["timestamp"] <= cutoff]
        m15_slice = m15.iloc[:index]

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
            "signal_id": _canonical_signal_id(symbol, signal.timestamp, signal.side),
            "timestamp": signal.timestamp,
            "symbol": symbol,
            "side": signal.side,
            "entry": float(signal.entry),
            "stop_loss": float(signal.stop_loss),
            "take_profit": float(signal.take_profit),
            "confluence_score": float(signal.confluence_score),
            "ai_enabled": False,
            "reason": signal.reason,
            "diagnostic_reason": diagnostic.get("reason"),
        }
        signal_rows.append(signal_row)

        # Only candles strictly after the signal timestamp are supplied to the
        # resolver. The signal-producing candle is therefore never resolved
        # using information from itself or from a future decision window.
        outcome = resolve_signal(signal, m15.iloc[index + 1 :])
        outcomes.append(_canonical_outcome(outcome))

    data_window = {
        "daily_start": daily["timestamp"].min().isoformat() if not daily.empty else None,
        "daily_end": daily["timestamp"].max().isoformat() if not daily.empty else None,
        "h4_start": h4["timestamp"].min().isoformat() if not h4.empty else None,
        "h4_end": h4["timestamp"].max().isoformat() if not h4.empty else None,
        "h1_start": h1["timestamp"].min().isoformat() if not h1.empty else None,
        "h1_end": h1["timestamp"].max().isoformat() if not h1.empty else None,
        "m15_start": m15["timestamp"].min().isoformat() if not m15.empty else None,
        "m15_end": m15["timestamp"].max().isoformat() if not m15.empty else None,
        "rows": {
            "daily": len(daily),
            "h4": len(h4),
            "h1": len(h1),
            "m15": len(m15),
        },
    }
    return outcomes, signal_rows, data_window


def _provenance(config: dict[str, Any]) -> dict[str, Any]:
    settings_path = Path("config/settings.yaml")
    return {
        "git_sha": os.getenv("GITHUB_SHA", "unknown"),
        "git_ref": os.getenv("GITHUB_REF", "unknown"),
        "workflow": os.getenv("GITHUB_WORKFLOW", "local"),
        "workflow_run_id": os.getenv("GITHUB_RUN_ID", "local"),
        "repository": os.getenv("GITHUB_REPOSITORY", "local"),
        "exchange": config["trading"]["exchange"],
        "fallback_exchange_enabled": bool(
            config["trading"].get("fallback_exchange_enabled", False)
        ),
        "symbols": list(config["trading"]["symbols"]),
        "strategy": "deterministic_smc",
        "ai_enabled": False,
        "ai_dependency": False,
        "model_artifacts_required": False,
        "settings_sha256": _sha256_file(settings_path),
        "validation_script_sha256": _sha256_file(Path(__file__)),
    }


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Make performance metrics strict-JSON compatible without changing core math.

    ``calculate_performance`` intentionally represents an undefined profit factor
    as positive infinity when there is profit but no loss. JSON has no Infinity
    value, so the Phase 4 artifact records that undefined statistic as ``null``.
    This keeps the reporting boundary strict while leaving the Phase 3 performance
    implementation untouched.
    """
    safe: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not isfinite(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def _aggregate_by_key(
    outcomes: list[TradeOutcome],
    key: Callable[[TradeOutcome], str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[TradeOutcome]] = {}
    for outcome in outcomes:
        groups.setdefault(key(outcome), []).append(outcome)
    return {
        name: _json_safe_metrics(calculate_performance(rows))
        for name, rows in sorted(groups.items())
    }


def _period_key(outcome: TradeOutcome) -> str:
    timestamp = pd.Timestamp(outcome.signal_timestamp)
    return timestamp.strftime("%Y-%m")


def build_report(
    config: dict[str, Any],
    outcomes: list[TradeOutcome],
    signal_rows: list[dict[str, Any]],
    data_windows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    aggregate = _json_safe_metrics(calculate_performance(outcomes))
    aggregate.update(
        {
            "strategy": "deterministic_smc",
            "timeframe": config["market_data"]["timeframes"]["m15"],
            "ai_enabled": False,
            "ai_dependency": False,
            "model_artifacts_required": False,
        }
    )

    return {
        "schema_version": "phase4-baseline-v1",
        "provenance": _provenance(config),
        "data_windows": data_windows,
        "aggregate": aggregate,
        "by_symbol": _aggregate_by_key(outcomes, lambda item: item.symbol),
        "by_side": _aggregate_by_key(outcomes, lambda item: item.side),
        "by_period": _aggregate_by_key(outcomes, _period_key),
        "signal_count": len(signal_rows),
        "resolved_outcome_count": len(outcomes),
    }


def run_validation() -> int:
    config = copy.deepcopy(load_all_configs(require_secrets=False, require_notifications=False))
    config["model"]["enabled"] = False
    config["model"]["required"] = False

    exchange = ExchangeInterface(config)
    all_outcomes: list[TradeOutcome] = []
    all_signal_rows: list[dict[str, Any]] = []
    data_windows: dict[str, dict[str, Any]] = {}

    for symbol in config["trading"]["symbols"]:
        outcomes, signal_rows, window = _run_symbol(config, exchange, symbol)
        all_outcomes.extend(outcomes)
        all_signal_rows.extend(signal_rows)
        data_windows[symbol] = window

    report = build_report(config, all_outcomes, all_signal_rows, data_windows)
    REPORT_DIR.mkdir(exist_ok=True)

    results = pd.DataFrame([item.to_dict() for item in all_outcomes])
    if results.empty:
        results = pd.DataFrame(
            columns=[
                "signal_id",
                "signal_timestamp",
                "resolution_timestamp",
                "symbol",
                "side",
                "entry",
                "stop_loss",
                "take_profit",
                "outcome",
                "r_multiple",
            ]
        )
    results.to_csv(RESULTS_PATH, index=False)
    ARTIFACT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    aggregate = report["aggregate"]
    print(
        "PHASE4_BASELINE_SUMMARY "
        f"signals={aggregate['total_signals']} "
        f"resolved={aggregate['resolved_trades']} "
        f"wins={aggregate['wins']} "
        f"losses={aggregate['losses']} "
        f"net_r={aggregate['net_r']:.4f} "
        f"expectancy_r={aggregate['expectancy_r']:.4f} "
        f"win_rate={aggregate['win_rate']:.4f} "
        f"max_drawdown_r={aggregate['max_drawdown_r']:.4f}"
    )
    print(f"PHASE4_ARTIFACT {ARTIFACT_PATH}")
    print(f"PHASE4_RESULTS {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_validation())
