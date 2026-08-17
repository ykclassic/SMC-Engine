import pandas as pd

from core.risk import RiskEngine
from core.structure import MarketStructureDetector
from models.feature_engineering import FEATURE_COLUMNS, FeatureEngineer


def test_feature_schema_is_stable():
    rows = []
    for i in range(80):
        price = 100 + i * 0.1
        rows.append({
            "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=15 * i),
            "open": price,
            "high": price + 0.2,
            "low": price - 0.2,
            "close": price + 0.1,
            "volume": 1000 + i,
        })
    df = pd.DataFrame(rows)
    features = FeatureEngineer().build_features(df)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.shape == (80, len(FEATURE_COLUMNS))


def test_risk_levels_have_positive_distance():
    config = {"risk_management": {"atr_sl_multiplier": 1.5, "default_tp_rr": 2.0}}
    risk = RiskEngine(config)
    stop, target = risk.levels("LONG", 100.0, 1.0)
    assert stop < 100.0
    assert target > 100.0
    assert abs(target - 100.0) == 2.0 * abs(100.0 - stop)


def test_structure_detector_does_not_mark_unconfirmed_edge_as_fractal():
    rows = []
    for i in range(10):
        rows.append({"open": i, "high": i + 1, "low": i - 1, "close": i + 0.5, "volume": 1})
    df = pd.DataFrame(rows)
    detector = MarketStructureDetector({"market_structure": {"lookback_period": 2}})
    result = detector.detect_fractals(df)
    assert result.iloc[-1]["is_high"] == 0
    assert result.iloc[-1]["is_low"] == 0
