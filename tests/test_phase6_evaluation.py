from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_phase6_evaluation import build_report


def _baseline() -> dict:
    return {
        "schema_version": "phase4-baseline-v1",
        "provenance": {
            "git_sha": "phase4-commit",
            "strategy": "deterministic_smc",
            "ai_enabled": False,
            "ai_dependency": False,
        },
    }


def _ai(precision: float = 0.50) -> dict:
    return {
        "schema_version": "phase6-ai-evaluation-v1",
        "evaluation": {
            "candidate_id": "candidate-1",
            "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-10T00:00:00+00:00",
            "untouched": True,
        },
        "model_metrics": {
            "coverage": 0.10,
            "precision": precision,
            "roc_auc": 0.60,
        },
        "trading_metrics": {"net_r": 4.0, "expectancy_r": 1.0},
    }


def _trades(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.csv"
    path.write_text(
        "signal_timestamp,outcome,r_multiple\n"
        "2026-08-02T00:00:00+00:00,WIN,2\n"
        "2026-08-03T00:00:00+00:00,LOSS,-1\n",
        encoding="utf-8",
    )
    return path


def test_phase6_requires_untouched_oos(tmp_path: Path) -> None:
    ai = _ai()
    ai["evaluation"]["untouched"] = False
    with pytest.raises(ValueError, match="untouched=true"):
        build_report(
            _baseline(),
            _trades(tmp_path),
            ai,
            pd.Timestamp("2026-08-01", tz="UTC"),
            pd.Timestamp("2026-08-10", tz="UTC"),
        )


def test_phase6_requires_exact_same_oos_period(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exactly match"):
        build_report(
            _baseline(),
            _trades(tmp_path),
            _ai(),
            pd.Timestamp("2026-08-02", tz="UTC"),
            pd.Timestamp("2026-08-10", tz="UTC"),
        )


def test_phase6_fails_precision_gate(tmp_path: Path) -> None:
    report = build_report(
        _baseline(),
        _trades(tmp_path),
        _ai(precision=0.40),
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-10", tz="UTC"),
    )
    assert report["candidate"]["production_gates"]["passed"] is False
    assert report["candidate"]["production_gates"]["checks"]["precision"] is False
    assert report["candidate"]["eligible_for_production_reintroduction"] is False
    assert report["provenance"]["baseline_control_frozen"] is True
    assert report["provenance"]["baseline_recomputed"] is False


def test_phase6_passes_all_gates(tmp_path: Path) -> None:
    report = build_report(
        _baseline(),
        _trades(tmp_path),
        _ai(precision=0.50),
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-10", tz="UTC"),
    )
    assert report["candidate"]["production_gates"]["passed"] is True
    assert report["candidate"]["eligible_for_production_reintroduction"] is True


def test_phase6_report_is_strict_json(tmp_path: Path) -> None:
    report = build_report(
        _baseline(),
        _trades(tmp_path),
        _ai(),
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-10", tz="UTC"),
    )
    json.dumps(report, allow_nan=False, sort_keys=True)
