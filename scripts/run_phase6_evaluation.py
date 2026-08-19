from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "phase6-evaluation-v1"
PRODUCTION_GATES = {
    "minimum_coverage": 0.005,
    "minimum_test_auc": 0.55,
    "minimum_test_precision": 0.45,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _require(data: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _validate_period(
    evaluation: dict[str, Any], oos_start: pd.Timestamp, oos_end: pd.Timestamp
) -> None:
    if evaluation.get("untouched") is not True:
        raise ValueError("AI evaluation must explicitly declare untouched=true")
    start = pd.Timestamp(evaluation.get("start"))
    end = pd.Timestamp(evaluation.get("end"))
    if start != oos_start or end != oos_end:
        raise ValueError(
            "AI evaluation period does not exactly match the requested untouched OOS window: "
            f"AI=[{start.isoformat()}, {end.isoformat()}] "
            f"OOS=[{oos_start.isoformat()}, {oos_end.isoformat()}]"
        )
    if start >= end:
        raise ValueError("Untouched OOS period must have start < end")


def _baseline_control_metrics(
    trades_path: Path, oos_start: pd.Timestamp, oos_end: pd.Timestamp
) -> dict[str, Any]:
    frame = pd.read_csv(trades_path)
    required = {"signal_timestamp", "outcome", "r_multiple"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Baseline trades missing required columns: {sorted(missing)}")

    timestamps = pd.to_datetime(frame["signal_timestamp"], utc=True)
    mask = (timestamps >= oos_start) & (timestamps <= oos_end)
    oos = frame.loc[mask].copy()
    if oos.empty:
        raise ValueError("Frozen baseline contains no trades in the requested OOS window")

    r = pd.to_numeric(oos["r_multiple"], errors="raise")
    wins = int((r > 0).sum())
    losses = int((r < 0).sum())
    resolved = wins + losses
    net_r = float(r.sum())
    equity = r.cumsum()
    drawdown = equity - equity.cummax()

    return {
        "signals": int(len(oos)),
        "resolved_trades": resolved,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / resolved if resolved else 0.0,
        "net_r": net_r,
        "max_drawdown_r": float(abs(drawdown.min())) if not drawdown.empty else 0.0,
        "source_sha256": _sha256_file(trades_path),
    }


def _gate_result(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "coverage": float(metrics["coverage"]) >= PRODUCTION_GATES["minimum_coverage"],
        "roc_auc": float(metrics["roc_auc"]) >= PRODUCTION_GATES["minimum_test_auc"],
        "precision": float(metrics["precision"]) >= PRODUCTION_GATES["minimum_test_precision"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": dict(PRODUCTION_GATES),
    }


def build_report(
    baseline: dict[str, Any],
    baseline_trades: Path,
    ai: dict[str, Any],
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> dict[str, Any]:
    _require(baseline, ("schema_version", "provenance"), "Phase 4 baseline")
    provenance = baseline["provenance"]
    if provenance.get("strategy") != "deterministic_smc":
        raise ValueError("Frozen control is not the deterministic SMC baseline")
    if provenance.get("ai_enabled") is not False or provenance.get("ai_dependency") is not False:
        raise ValueError("Frozen control must be AI-independent")

    _require(ai, ("schema_version", "evaluation", "model_metrics", "trading_metrics"), "AI evaluation")
    if ai["schema_version"] != "phase6-ai-evaluation-v1":
        raise ValueError("Unsupported AI evaluation schema")
    _validate_period(ai["evaluation"], oos_start, oos_end)

    model_metrics = ai["model_metrics"]
    for key in ("coverage", "precision", "roc_auc"):
        if key not in model_metrics:
            raise ValueError(f"AI model_metrics missing {key}")
    gates = _gate_result(model_metrics)

    baseline_control = _baseline_control_metrics(baseline_trades, oos_start, oos_end)

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": {
            "start": oos_start.isoformat(),
            "end": oos_end.isoformat(),
            "untouched": True,
            "same_period_required": True,
        },
        "control": {
            "name": "smc_only",
            "strategy": "deterministic_smc",
            "phase4_baseline_schema": baseline["schema_version"],
            "phase4_baseline_git_sha": provenance.get("git_sha"),
            "phase4_baseline_provenance": provenance,
            "trading_metrics": baseline_control,
        },
        "candidate": {
            "name": "smc_plus_ai",
            "candidate_id": ai["evaluation"].get("candidate_id"),
            "model_metrics": model_metrics,
            "trading_metrics": ai["trading_metrics"],
            "production_gates": gates,
            "eligible_for_production_reintroduction": gates["passed"],
        },
        "provenance": {
            "repository": os.getenv("GITHUB_REPOSITORY", "local"),
            "git_sha": os.getenv("GITHUB_SHA", "unknown"),
            "workflow_run_id": os.getenv("GITHUB_RUN_ID", "local"),
            "baseline_trades_sha256": baseline_control["source_sha256"],
            "baseline_control_frozen": True,
            "baseline_recomputed": False,
            "ai_dependency_in_production": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare SMC-only control with an AI candidate on one untouched OOS window.")
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--baseline-trades", type=Path, required=True)
    parser.add_argument("--ai-evaluation", type=Path, required=True)
    parser.add_argument("--oos-start", required=True)
    parser.add_argument("--oos-end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    oos_start = pd.Timestamp(args.oos_start, tz="UTC")
    oos_end = pd.Timestamp(args.oos_end, tz="UTC")
    report = build_report(
        _load_json(args.baseline_json),
        args.baseline_trades,
        _load_json(args.ai_evaluation),
        oos_start,
        oos_end,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gates = report["candidate"]["production_gates"]
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"PHASE6_AI_PRODUCTION_GATE {'PASS' if gates['passed'] else 'FAIL'}")
    if not gates["passed"]:
        print("AI candidate remains research-only; deterministic SMC control is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
