from __future__ import annotations

import pandas as pd
import pytest

from scripts.train_models_ablation import _context, _integrity_report


class Row:
    def __init__(self, direction: str, event_type: str) -> None:
        self.direction = direction
        self.event_type = event_type


def test_context_modes_preserve_direction_and_event_dimensions():
    row = Row("LONG", "BULLISH_BOS")
    assert _context(row, "baseline").shape == (0,)
    assert _context(row, "direction_only").shape == (1,)
    assert _context(row, "event_only").shape == (10,)
    assert _context(row, "direction_plus_event").shape == (11,)
    assert _context(row, "direction_only")[0] == 1.0


def test_integrity_allows_opposite_directions_on_same_candle():
    labels = pd.DataFrame(
        [
            {"candidate_index": 10, "direction": "LONG", "event_type": "BULLISH_BOS", "label": 1.0},
            {"candidate_index": 10, "direction": "SHORT", "event_type": "BEARISH_BOS", "label": 0.0},
        ]
    )
    report = _integrity_report([("TEST", labels)])
    data = report["symbols"]["TEST"]
    assert data["directional_keys"] == 2
    assert data["duplicate_directional_keys"] == 0
    assert data["same_candle_opposite_direction_rows"] == 2
    assert data["direction_counts"] == {"LONG": 1, "SHORT": 1}


def test_integrity_rejects_duplicate_directional_key():
    labels = pd.DataFrame(
        [
            {"candidate_index": 10, "direction": "LONG", "event_type": "BULLISH_BOS", "label": 1.0},
            {"candidate_index": 10, "direction": "LONG", "event_type": "BULLISH_FVG", "label": 0.0},
        ]
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        _integrity_report([("TEST", labels)])


def test_integrity_report_contains_stable_population_fingerprint():
    labels = pd.DataFrame(
        [
            {"candidate_index": 10, "direction": "LONG", "event_type": "BULLISH_BOS", "label": 1.0},
            {"candidate_index": 11, "direction": "SHORT", "event_type": "BEARISH_SWEEP", "label": 0.0},
        ]
    )
    report = _integrity_report([("TEST", labels)])
    fingerprint = report["symbols"]["TEST"]["label_fingerprint"]
    assert len(fingerprint) == 64
    assert report["population_fingerprints"] == [fingerprint]
