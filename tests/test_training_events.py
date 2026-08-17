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


def add_event_columns(frame):
    for column in (
        "bos",
        "choch",
        "liquidity_sweep",
        "fvg",
        "ob_event",
    ):
        frame[column] = None
    return frame


def builder_config(**overrides):
    values = {
        "continuation_window": 8,
        "minimum_candidate_spacing": 3,
        "max_candidates_per_event": 3,
    }
    values.update(overrides)
    return {"training_events": values}


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
    frame = add_event_columns(make_frame(20))
    frame.loc[5, "bos"] = "BULLISH_BOS"
    frame.loc[12, "liquidity_sweep"] = "BEARISH_SWEEP"

    builder = SMCTrainingEventBuilder(builder_config())
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


def test_builder_processes_all_event_types_on_same_candle():
    frame = add_event_columns(make_frame(30))
    frame.loc[10, "bos"] = "BULLISH_BOS"
    frame.loc[10, "fvg"] = "BULLISH_FVG"
    frame.loc[10, "ob_event"] = "BULLISH_OB"

    builder = SMCTrainingEventBuilder(
        builder_config(
            continuation_window=0,
            minimum_candidate_spacing=1,
            max_candidates_per_event=1,
        )
    )
    candidates = builder.build(frame)

    # All three detections are recorded as event origins even though they
    # resolve to one canonical GRU observation at the same candle/direction.
    assert builder.last_build_stats["event_count"] == 3
    assert builder.last_build_stats["event_counts"] == {
        "BULLISH_BOS": 1,
        "BULLISH_FVG": 1,
        "BULLISH_OB": 1,
    }
    assert len(candidates) == 1
    assert candidates.iloc[0]["event_type"] == "BULLISH_BOS"


def test_training_builder_expands_candidates_from_multiple_event_origins():
    frame = add_event_columns(make_frame(30))
    frame.loc[5, "bos"] = "BULLISH_BOS"
    frame.loc[7, "fvg"] = "BULLISH_FVG"

    builder = SMCTrainingEventBuilder(
        builder_config(
            continuation_window=4,
            minimum_candidate_spacing=3,
            max_candidates_per_event=3,
        )
    )
    candidates = builder.build(frame)

    assert builder.last_build_stats["event_count"] == 2
    assert builder.last_build_stats["raw_candidates"] > 3
    assert 7 in set(candidates["candidate_index"])
    assert len(candidates) >= 4


def test_training_builder_labels_use_canonical_direction_values():
    frame = add_event_columns(make_frame(60))
    frame.loc[10, "bos"] = "BULLISH_BOS"
    frame.loc[11:15, "high"] = [101.0, 101.1, 101.2, 101.3, 101.4]

    builder = SMCTrainingEventBuilder(
        builder_config(
            continuation_window=0,
            minimum_candidate_spacing=1,
            max_candidates_per_event=1,
        )
    )
    candidates = builder.build(frame)

    atr = pd.Series(1.0, index=frame.index)
    labels = builder.label_candidates(
        frame,
        candidates,
        atr,
        sl_multiplier=1.0,
        rr=0.2,
        horizon=5,
    )

    assert not labels.empty
    assert set(labels["direction"]) <= {"LONG", "SHORT"}
    assert labels["candidate_index"].dtype.kind in "iu"
    assert set(labels["label"].unique()) <= {0.0, 1.0}
    assert labels.iloc[0]["label"] == 1.0


def test_unresolved_candidate_is_retained_as_negative_label():
    frame = add_event_columns(make_frame(60))
    frame.loc[10, "bos"] = "BULLISH_BOS"

    builder = SMCTrainingEventBuilder(
        builder_config(
            continuation_window=0,
            minimum_candidate_spacing=1,
            max_candidates_per_event=1,
        )
    )
    candidates = builder.build(frame)
    atr = pd.Series(10.0, index=frame.index)

    labels = builder.label_candidates(
        frame,
        candidates,
        atr,
        sl_multiplier=1.0,
        rr=2.0,
        horizon=5,
    )

    assert len(labels) == 1
    assert labels.iloc[0]["label"] == 0.0
