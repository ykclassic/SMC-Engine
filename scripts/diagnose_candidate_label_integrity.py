"""Diagnose SMC candidate, label, temporal, and separability integrity.

This diagnostic is intentionally read-only with respect to model artifacts. It
fetches the configured real-market dataset, reconstructs the exact production
candidate/label pipeline, and writes a JSON report containing integrity and
separability diagnostics for each configured symbol and the aggregate BTC/ETH
training set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.training_events import EVENT_DIRECTIONS, SMCTrainingEventBuilder
from scripts.train_models import build_dataset, build_labeled_dataset, validate_candidates
from utils.config_loader import load_all_configs


REPORT_PATH = Path("models/weights/candidate_label_integrity.json")


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _basic_distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0}
    quantiles = numeric.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "count": int(len(numeric)),
        "mean": _finite_float(numeric.mean()),
        "std": _finite_float(numeric.std(ddof=0)),
        "min": _finite_float(numeric.min()),
        "p05": _finite_float(quantiles.loc[0.05]),
        "p25": _finite_float(quantiles.loc[0.25]),
        "median": _finite_float(quantiles.loc[0.50]),
        "p75": _finite_float(quantiles.loc[0.75]),
        "p95": _finite_float(quantiles.loc[0.95]),
        "max": _finite_float(numeric.max()),
    }


def _event_collision_report(candidates: pd.DataFrame) -> dict[str, Any]:
    key = ["candidate_index", "direction"]
    grouped = candidates.groupby(key, sort=False)
    multi = grouped["event_type"].agg(lambda values: sorted(set(values)))
    multi = multi[multi.map(len) > 1]

    discarded: dict[str, int] = {}
    for events in multi:
        # Production validation keeps the first stable event for a duplicate
        # directional key. Report all additional event types that are lost.
        for event in events[1:]:
            discarded[event] = discarded.get(event, 0) + 1

    return {
        "raw_candidate_rows": int(len(candidates)),
        "unique_directional_keys": int(len(grouped)),
        "multi_event_directional_keys": int(len(multi)),
        "extra_event_rows_colliding_on_directional_key": int(
            sum(max(0, len(events) - 1) for events in multi)
        ),
        "discarded_event_type_counts_if_first_is_kept": dict(sorted(discarded.items())),
    }


def _label_integrity_report(labels: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "rows": int(len(labels)),
        "null_counts": {
            column: int(labels[column].isna().sum())
            for column in labels.columns
        },
        "invalid_label_values": sorted(
            float(value)
            for value in pd.to_numeric(labels["label"], errors="coerce")
            .dropna()
            .unique()
            if value not in (0.0, 1.0)
        ),
        "duplicate_directional_keys": int(
            labels.duplicated(["candidate_index", "direction"], keep=False).sum()
        ),
        "candidate_index_min": int(labels["candidate_index"].min()),
        "candidate_index_max": int(labels["candidate_index"].max()),
    }
    report["positive_count"] = int((labels["label"] == 1.0).sum())
    report["negative_count"] = int((labels["label"] == 0.0).sum())
    report["positive_rate"] = float(labels["label"].mean())
    return report


def _separability_report(labels: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": {}, "by_direction": {}, "by_event_type": {}, "by_distance": {}}

    def summarize(group: pd.DataFrame) -> dict[str, Any]:
        positives = group.loc[group["label"] == 1.0]
        negatives = group.loc[group["label"] == 0.0]
        return {
            "samples": int(len(group)),
            "positive_rate": float(group["label"].mean()),
            "positive_count": int(len(positives)),
            "negative_count": int(len(negatives)),
        }

    result["overall"] = summarize(labels)
    for direction, group in labels.groupby("direction", sort=True):
        result["by_direction"][str(direction)] = summarize(group)
    for event_type, group in labels.groupby("event_type", sort=True):
        result["by_event_type"][str(event_type)] = summarize(group)
    for distance, group in labels.groupby("distance_from_event", sort=True):
        result["by_distance"][str(int(distance))] = summarize(group)
    return result


def _future_overlap_report(labels: pd.DataFrame, split1: int, split2: int, horizon: int, frame_length: int) -> dict[str, Any]:
    """Quantify labels whose future outcome window crosses a partition boundary."""
    train = labels.iloc[:split1]
    validation = labels.iloc[split1:split2]
    test = labels.iloc[split2:]

    boundaries = {"train_validation": split1, "validation_test": split2}

    def crossing(group: pd.DataFrame, boundary: int) -> pd.Series:
        return group["candidate_index"].astype(int) < boundary, group["candidate_index"].astype(int) + horizon >= boundary

    train_before, train_cross = crossing(train, split1)
    validation_before, validation_cross = crossing(validation, split2)
    _ = train_before, validation_before

    # A candidate exactly at or after a boundary belongs to the later split;
    # only candidates in the earlier split can leak their label outcome across it.
    train_cross_mask = train["candidate_index"].astype(int) + horizon >= split1
    validation_cross_mask = validation["candidate_index"].astype(int) + horizon >= split2
    test_horizon_truncated = test["candidate_index"].astype(int) + horizon >= frame_length

    return {
        "label_horizon": int(horizon),
        "boundaries": boundaries,
        "train_labels_crossing_validation_boundary": int(train_cross_mask.sum()),
        "train_crossing_rate": float(train_cross_mask.mean()) if len(train) else 0.0,
        "validation_labels_crossing_test_boundary": int(validation_cross_mask.sum()),
        "validation_crossing_rate": float(validation_cross_mask.mean()) if len(validation) else 0.0,
        "test_candidates_with_truncated_future_horizon": int(test_horizon_truncated.sum()),
        "test_truncation_rate": float(test_horizon_truncated.mean()) if len(test) else 0.0,
    }


def _symbol_report(frame: pd.DataFrame, labels: pd.DataFrame, config: dict) -> dict[str, Any]:
    model_cfg = config["model"]
    horizon = int(model_cfg.get("label_horizon", 20))
    split1 = int(len(labels) * 0.70)
    split2 = int(len(labels) * 0.85)

    # Reconstruct the raw candidate set independently so the report can
    # quantify how much information is lost before labeling.
    builder = SMCTrainingEventBuilder(config)
    raw_candidates = builder.build(frame)
    canonical_candidates = validate_candidates(raw_candidates, len(frame))

    event_count = int(sum(1 for row in frame.itertuples(index=False) for column in ("choch", "bos", "liquidity_sweep", "fvg", "ob_event") if getattr(row, column, None) in EVENT_DIRECTIONS))

    return {
        "symbol": str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "UNKNOWN",
        "frame_rows": int(len(frame)),
        "event_count_reconstructed": event_count,
        "candidate_builder_stats": builder.last_build_stats,
        "candidate_collision_analysis": _event_collision_report(raw_candidates),
        "canonical_candidates_after_validation": int(len(canonical_candidates)),
        "labeled_candidates": _label_integrity_report(labels),
        "separability": _separability_report(labels),
        "future_label_overlap": _future_overlap_report(labels, split1, split2, horizon, len(frame)),
        "label_positive_rate_by_direction": {
            str(direction): float(group["label"].mean())
            for direction, group in labels.groupby("direction", sort=True)
        },
        "label_positive_rate_by_event_type": {
            str(event_type): float(group["label"].mean())
            for event_type, group in labels.groupby("event_type", sort=True)
        },
        "distance_from_event_distribution": _basic_distribution(labels["distance_from_event"]),
    }


def _aggregate_report(symbol_reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "symbol": report["symbol"],
            "labeled_candidates": report["labeled_candidates"]["rows"],
            "positive_rate": report["labeled_candidates"]["positive_rate"],
            "multi_event_directional_keys": report["candidate_collision_analysis"]["multi_event_directional_keys"],
            "extra_event_collisions": report["candidate_collision_analysis"]["extra_event_rows_colliding_on_directional_key"],
            "train_labels_crossing_boundary": report["future_label_overlap"]["train_labels_crossing_validation_boundary"],
            "validation_labels_crossing_boundary": report["future_label_overlap"]["validation_labels_crossing_test_boundary"],
        }
        for report in symbol_reports
    ]
    total_labeled = sum(row["labeled_candidates"] for row in rows)
    weighted_positive = sum(row["labeled_candidates"] * row["positive_rate"] for row in rows)
    return {
        "symbols": rows,
        "total_labeled_candidates": int(total_labeled),
        "aggregate_positive_rate": float(weighted_positive / total_labeled) if total_labeled else 0.0,
        "total_multi_event_directional_keys": int(sum(row["multi_event_directional_keys"] for row in rows)),
        "total_extra_event_collisions": int(sum(row["extra_event_collisions"] for row in rows)),
        "total_train_labels_crossing_split": int(sum(row["train_labels_crossing_boundary"] for row in rows)),
        "total_validation_labels_crossing_split": int(sum(row["validation_labels_crossing_boundary"] for row in rows)),
    }


def main() -> None:
    config = load_all_configs(require_notifications=False)
    frames = build_dataset(config)
    symbol_reports: list[dict[str, Any]] = []

    for frame in frames:
        _, labels = build_labeled_dataset(frame, config)
        symbol_reports.append(_symbol_report(frame, labels, config))

    report = {
        "schema_version": "candidate-label-integrity-v1",
        "symbols": [item["symbol"] for item in symbol_reports],
        "aggregate": _aggregate_report(symbol_reports),
        "symbol_reports": symbol_reports,
        "production_impact": {
            "raw_candidate_deduplication_is_lossy": any(
                item["candidate_collision_analysis"]["extra_event_rows_colliding_on_directional_key"] > 0
                for item in symbol_reports
            ),
            "future_label_cross_split_risk_present": any(
                item["future_label_overlap"]["train_labels_crossing_validation_boundary"] > 0
                or item["future_label_overlap"]["validation_labels_crossing_test_boundary"] > 0
                for item in symbol_reports
            ),
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
