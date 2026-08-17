import pandas as pd

from core.structure import MarketStructureDetector
from core.training_events import SMCTrainingEventBuilder


def make_frame(rows=40):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="15min",
                tz="UTC",
            ),
            "open": [100 + i * 0.05 for i in range(rows)],
            "high": [100.2 + i * 0.05 for i in range(rows)],
            "low": [99.8 + i * 0.05 for i in range(rows)],
            "close": [100.1 + i * 0.05 for i in range(rows)],
            "volume": [1000.0] * rows,
        }
    )


def test_structure_does_not_mark_bos_before_swing_confirmation():
    config = {
        "market_structure": {
            "lookback_period": 2,
            "confirmation_candles": 1,
        }
    }
    detector = MarketStructureDetector(config)
    frame = make_frame()
    analyzed = detector.analyze(frame)

    assert "high_confirmed" in analyzed.columns
    assert "low_confirmed" in analyzed.columns
    assert "bos" in analyzed.columns


def test_training_builder_emits_canonical_candidate_schema():
    config = {
        "training_events": {
            "continuation_window": 8,
            "minimum_candidate_spacing": 3,
            "max_candidates_per_event": 3,
        }
    }

    frame = make_frame(20)
    frame["bos"] = None
    frame["choch"] = None
    frame["liquidity_sweep"] = None
    frame["fvg"] = None
    frame["ob_event"] = None
    frame.loc[5, "bos"] = "BULLISH_BOS"
    frame.loc[12, "liquidity_sweep"] = "BEARISH_SWEEP"

    builder = SMCTrainingEventBuilder(config)
    candidates = builder.build(frame)

    assert not candidates.empty
    assert list(candidates.columns) == [
        "candidate_index",
        "direction",
        "event_type",
        "distance_from_event",
    ]
    assert candidates["candidate_index"].notna().all()
    assert candidates["candidate_index"].dtype.kind in "iu"
    assert set(candidates["direction"]) <= {"LONG", "SHORT"}
    assert candidates[
        ["candidate_index", "direction"]
    ].duplicated().sum() == 0
    assert candidates["distance_from_event"].min() >= 0
    assert candidates["distance_from_event"].max() <= 8


def test_training_builder_labels_use_canonical_direction_values():
    config = {
        "training_events": {
            "continuation_window": 0,
            "minimum_candidate_spacing": 1,
            "max_candidates_per_event": 1,
        }
    }

    frame = make_frame(60)
    frame["bos"] = None
    frame["choch"] = None
    frame["liquidity_sweep"] = None
    frame["fvg"] = None
    frame["ob_event"] = None
    frame.loc[10, "bos"] = "BULLISH_BOS"

    builder = SMCTrainingEventBuilder(config)
    candidates = builder.build(frame)

    atr = pd.Series(1.0, index=frame.index)
    labels = builder.label_candidates(
        frame,
        candidates,
        atr,
        sl_multiplier=1.0,
        rr=1.0,
        horizon=5,
    )

    assert not labels.empty
    assert set(labels["direction"]) <= {"LONG", "SHORT"}
    assert labels["candidate_index"].dtype.kind in "iu"
    assert set(labels["label"].unique()) <= {0.0, 1.0}
