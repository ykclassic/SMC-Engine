import pandas as pd

from core.confluence import ConfluenceEngine
from core.risk import RiskEngine
from core.structure import MarketStructureDetector
from models.feature_engineering import FEATURE_COLUMNS, FeatureEngineer


def test_feature_schema_is_stable():
    rows = []
    for i in range(80):
        price = 100 + i * 0.1
        rows.append(
            {
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=15 * i),
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price + 0.1,
                "volume": 1000 + i,
            }
        )
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


def _confluence_frame(rows: int = 20) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    close = pd.Series([100.0 + i * 0.1 for i in range(rows)])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.05,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0,
            "bos": [None] * (rows - 1) + ["BULLISH_BOS"],
            "choch": [None] * (rows - 1) + [None],
            "structure_bias": ["BULLISH"] * rows,
            "liquidity_sweep": [None] * (rows - 1) + ["BULLISH_SWEEP"],
            "order_block": [None] * rows,
            "ob_mitigated": [False] * rows,
            "fvg": [None] * rows,
            "ob_top": [float("nan")] * rows,
            "ob_bottom": [float("nan")] * rows,
            "fvg_top": [float("nan")] * rows,
            "fvg_bottom": [float("nan")] * rows,
        }
    )


def test_deterministic_confluence_generates_signal_without_ai():
    config = {
        "confluence": {
            "minimum_score": 0.70,
            "weights": {
                "daily_bias": 0.20,
                "h4_bias": 0.20,
                "h1_setup": 0.20,
                "m15_sweep": 0.20,
                "m15_confirmation": 0.20,
            },
        },
        "model": {"enabled": False, "required": False},
        "risk_management": {"atr_sl_multiplier": 1.5, "default_tp_rr": 2.0},
    }
    engine = ConfluenceEngine(config)
    daily = _confluence_frame()
    h4 = _confluence_frame()
    h1 = _confluence_frame()
    h1.loc[h1.index[-1], "choch"] = "BULLISH_CHOCH"
    m15 = _confluence_frame()

    signal, diagnostic = engine.validate_signal(daily, h4, h1, m15)

    assert signal is not None
    assert signal.side == "LONG"
    assert signal.ai_confidence is None
    assert diagnostic["ai_enabled"] is False
    assert diagnostic["ai_status"] == "DISABLED"
    assert diagnostic["reason"] == "VALIDATED"
    assert signal.stop_loss < signal.entry < signal.take_profit
