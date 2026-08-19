from __future__ import annotations

import numpy as np
import pandas as pd

from models.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION, FeatureEngineer


def make_frame(rows: int = 90) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="15min",
        tz="UTC",
    )
    close = 100.0 + np.arange(rows, dtype=float) * 0.03
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "bos": None,
            "choch": None,
            "fvg": None,
            "fvg_top": np.nan,
            "fvg_bottom": np.nan,
            "ob_event": None,
            "ob_origin_index": np.nan,
            "ob_top": np.nan,
            "ob_bottom": np.nan,
            "liquidity_pool": None,
            "confirmed_swing_high": np.nan,
            "confirmed_swing_low": np.nan,
            "liquidity_sweep": None,
            "structure_bias": None,
        }
    )

    frame.loc[20, "fvg"] = "BULLISH_FVG"
    frame.loc[20, "fvg_top"] = close[20] + 0.20
    frame.loc[20, "fvg_bottom"] = close[20] + 0.05

    frame.loc[30, "ob_event"] = "BULLISH_OB"
    frame.loc[30, "ob_origin_index"] = 25
    frame.loc[25, "ob_top"] = close[25] - 0.02
    frame.loc[25, "ob_bottom"] = close[25] - 0.12

    frame.loc[35, "liquidity_pool"] = "EQH"
    frame.loc[35, "confirmed_swing_high"] = close[35] + 0.20
    frame.loc[40, "liquidity_sweep"] = "BEARISH_SWEEP"

    frame.loc[45, "bos"] = "BULLISH_BOS"
    frame.loc[50, "choch"] = "BEARISH_CHOCH"
    frame.loc[55:, "structure_bias"] = "BULLISH"

    return frame


def test_feature_schema_is_versioned_and_complete() -> None:
    engineer = FeatureEngineer(sequence_length=32)
    features = engineer.build_features(make_frame())

    assert FEATURE_VERSION == "smc-v4"
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 29
    assert features.shape[0] == 90
    assert np.isfinite(features.to_numpy()).all()


def test_high_value_smc_geometry_features_are_populated_causally() -> None:
    engineer = FeatureEngineer(sequence_length=32)
    features = engineer.build_features(make_frame())

    assert features.loc[20, "fvg_size_atr"] > 0
    assert features.loc[20, "fvg_age"] == 0
    assert features.loc[21, "fvg_age"] == 1
    assert features.loc[21, "fvg_distance_atr"] >= 0
    assert 0 <= features.loc[21, "fvg_fill_ratio"] <= 1

    assert features.loc[30, "ob_size_atr"] > 0
    assert features.loc[30, "ob_age"] == 0
    assert features.loc[31, "ob_age"] == 1
    assert features.loc[30, "ob_distance_atr"] >= 0

    assert features.loc[40, "sweep_magnitude_atr"] >= 0
    assert features.loc[40, "liquidity_distance_atr"] >= 0

    assert features.loc[55, "structure_bias_numeric"] == 1.0


def test_future_rows_cannot_change_prior_feature_values() -> None:
    engineer = FeatureEngineer(sequence_length=32)
    base = make_frame(60)
    base_features = engineer.build_features(base)

    extended = make_frame(90)
    extended.loc[60:, "open"] = 500.0
    extended.loc[60:, "high"] = 600.0
    extended.loc[60:, "low"] = 400.0
    extended.loc[60:, "close"] = 550.0
    extended.loc[60:, "volume"] = 9_000_000.0
    extended.loc[60:, "fvg"] = "BEARISH_FVG"
    extended.loc[60:, "fvg_top"] = 600.0
    extended.loc[60:, "fvg_bottom"] = 400.0
    extended.loc[60:, "ob_event"] = "BEARISH_OB"
    extended.loc[60:, "ob_origin_index"] = 60
    extended.loc[60:, "ob_top"] = 600.0
    extended.loc[60:, "ob_bottom"] = 400.0
    extended.loc[60:, "liquidity_sweep"] = "BEARISH_SWEEP"
    extended.loc[60:, "structure_bias"] = "BEARISH"

    extended_features = engineer.build_features(extended)

    np.testing.assert_allclose(
        base_features.to_numpy(),
        extended_features.iloc[:60].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_future_mitigation_fields_are_not_consumed_as_features() -> None:
    engineer = FeatureEngineer(sequence_length=32)
    frame = make_frame()
    frame["ob_mitigated"] = False
    baseline = engineer.build_features(frame)

    frame.loc[30:, "ob_mitigated"] = True
    changed = engineer.build_features(frame)

    np.testing.assert_allclose(
        baseline.to_numpy(),
        changed.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
