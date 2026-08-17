import pandas as pd

from core.structure import MarketStructureDetector
from core.training_events import SMCTrainingEventBuilder


def make_frame(rows=40):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC"),
        "open": [100 + i * 0.05 for i in range(rows)],
        "high": [100.2 + i * 0.05 for i in range(rows)],
        "low": [99.8 + i * 0.05 for i in range(rows)],
        "close": [100.1 + i * 0.05 for i in range(rows)],
        "volume": [1000.0] * rows,
    })


def test_structure_does_not_mark_bos_before_swing_confirmation():
    config = {"market_structure": {"lookback_period": 2, "confirmation_candles": 1}}
    detector = MarketStructureDetector(config)
    frame = make_frame()
    analyzed = detector.analyze(frame)
    assert "high_confirmed" in analyzed.columns
    assert "low_confirmed" in analyzed.columns
    assert "bos" in analyzed.columns


def test_training_builder_creates_spaced_continuation_candidates():
    config = {"training_events": {"continuation_window": 8, "minimum_candidate_spacing": 3, "max_candidates_per_event": 3}}
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
    assert candidates["candidate_index"].is_unique
    assert candidates["distance_from_event"].max() <= 8
