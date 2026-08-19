from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_VERSION = "smc-v4"
FEATURE_COLUMNS = [
    "return_1",
    "atr_norm",
    "range_norm",
    "body_ratio",
    "volume_z",
    "trend_distance",
    "bos_numeric",
    "choch_numeric",
    "fvg_numeric",
    "ob_numeric",
    "sweep_numeric",
    "zone_distance",
    "rsi_norm",
    "momentum_norm",
    "hour_sin",
    "hour_cos",
    "displacement_atr",
    "fvg_size_atr",
    "fvg_age",
    "fvg_distance_atr",
    "fvg_fill_ratio",
    "ob_size_atr",
    "ob_age",
    "ob_distance_atr",
    "sweep_magnitude_atr",
    "liquidity_distance_atr",
    "adx_norm",
    "atr_percentile",
    "structure_bias_numeric",
]


class FeatureEngineer:
    """Build point-in-time-safe candle and SMC features.

    Every feature at row ``t`` is computed from data available at or before
    the close of row ``t``. Persistent FVG/OB/liquidity state is maintained
    forward in time rather than inferred from future mitigation outcomes.
    """

    def __init__(self, sequence_length: int = 32) -> None:
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        previous_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - previous_close).abs(),
                (df["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(period, min_periods=period).mean()

    @staticmethod
    def _rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where(
            (up_move > down_move) & (up_move > 0), 0.0
        )
        minus_dm = down_move.where(
            (down_move > up_move) & (down_move > 0), 0.0
        )

        tr_mean = true_range.rolling(period, min_periods=period).mean()
        plus_mean = plus_dm.rolling(period, min_periods=period).mean()
        minus_mean = minus_dm.rolling(period, min_periods=period).mean()

        plus_di = 100 * plus_mean / tr_mean.replace(0, np.nan)
        minus_di = 100 * minus_mean / tr_mean.replace(0, np.nan)
        denominator = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / denominator
        return dx.rolling(period, min_periods=period).mean()

    @staticmethod
    def _rolling_percentile(series: pd.Series, window: int = 100) -> pd.Series:
        """Return the percentile rank of the current value in past data."""

        def percentile(values: np.ndarray) -> float:
            if len(values) == 0:
                return np.nan
            current = values[-1]
            return float(np.mean(values <= current))

        return series.rolling(window, min_periods=max(5, window // 5)).apply(
            percentile,
            raw=True,
        )

    @staticmethod
    def _numeric_event(
        series: pd.Series,
        bullish: str,
        bearish: str,
    ) -> pd.Series:
        return series.map({bullish: 1.0, bearish: -1.0}).fillna(0.0)

    def _smc_state_features(
        self,
        frame: pd.DataFrame,
        atr: pd.Series,
    ) -> pd.DataFrame:
        """Build causal persistent SMC-zone/liquidity geometry."""

        index = frame.index
        result = pd.DataFrame(index=index)
        result["fvg_size_atr"] = 0.0
        result["fvg_age"] = 0.0
        result["fvg_distance_atr"] = 0.0
        result["fvg_fill_ratio"] = 0.0
        result["ob_size_atr"] = 0.0
        result["ob_age"] = 0.0
        result["ob_distance_atr"] = 0.0
        result["sweep_magnitude_atr"] = 0.0
        result["liquidity_distance_atr"] = 0.0

        active_fvg: tuple[float, float, int, str] | None = None
        active_ob: tuple[float, float, int, str] | None = None
        liquidity_level: tuple[float, int, str] | None = None

        for i in index:
            current_atr = float(atr.at[i]) if pd.notna(atr.at[i]) else np.nan
            close = float(frame.at[i, "close"])
            high = float(frame.at[i, "high"])
            low = float(frame.at[i, "low"])
            atr_value = max(current_atr, 1e-12) if np.isfinite(current_atr) else np.nan

            fvg = frame.at[i, "fvg"] if "fvg" in frame else None
            fvg_top = frame.at[i, "fvg_top"] if "fvg_top" in frame else np.nan
            fvg_bottom = frame.at[i, "fvg_bottom"] if "fvg_bottom" in frame else np.nan
            if (
                pd.notna(fvg_top)
                and pd.notna(fvg_bottom)
                and fvg in {"BULLISH_FVG", "BEARISH_FVG"}
            ):
                active_fvg = (
                    float(fvg_top),
                    float(fvg_bottom),
                    int(i),
                    str(fvg),
                )

            if active_fvg is not None and np.isfinite(atr_value):
                top, bottom, origin, direction = active_fvg
                size = max(top - bottom, 0.0)
                midpoint = (top + bottom) / 2.0
                distance = abs(close - midpoint) / atr_value
                if direction == "BULLISH_FVG":
                    fill = np.clip((top - low) / max(size, 1e-12), 0.0, 1.0)
                else:
                    fill = np.clip((high - bottom) / max(size, 1e-12), 0.0, 1.0)
                result.at[i, "fvg_size_atr"] = size / atr_value
                result.at[i, "fvg_age"] = float(i - origin)
                result.at[i, "fvg_distance_atr"] = distance
                result.at[i, "fvg_fill_ratio"] = float(fill)
                if fill >= 1.0:
                    active_fvg = None

            ob_event = frame.at[i, "ob_event"] if "ob_event" in frame else None
            origin_value = frame.at[i, "ob_origin_index"] if "ob_origin_index" in frame else np.nan
            if (
                ob_event in {"BULLISH_OB", "BEARISH_OB"}
                and pd.notna(origin_value)
            ):
                origin = int(origin_value)
                if 0 <= origin < len(frame):
                    top = frame.at[origin, "ob_top"]
                    bottom = frame.at[origin, "ob_bottom"]
                    if pd.notna(top) and pd.notna(bottom):
                        active_ob = (
                            float(top),
                            float(bottom),
                            int(i),
                            str(ob_event),
                        )

            if active_ob is not None and np.isfinite(atr_value):
                top, bottom, origin, _ = active_ob
                size = max(top - bottom, 0.0)
                midpoint = (top + bottom) / 2.0
                result.at[i, "ob_size_atr"] = size / atr_value
                result.at[i, "ob_age"] = float(i - origin)
                result.at[i, "ob_distance_atr"] = abs(close - midpoint) / atr_value

                if low <= bottom or high >= top:
                    # The current candle has interacted with the zone. Keep
                    # the feature on this candle, then retire the zone so
                    # future rows cannot treat an already-consumed OB as live.
                    active_ob = None

            pool = frame.at[i, "liquidity_pool"] if "liquidity_pool" in frame else None
            if pool == "EQH" and pd.notna(frame.at[i, "confirmed_swing_high"]):
                liquidity_level = (
                    float(frame.at[i, "confirmed_swing_high"]),
                    int(i),
                    "EQH",
                )
            elif pool == "EQL" and pd.notna(frame.at[i, "confirmed_swing_low"]):
                liquidity_level = (
                    float(frame.at[i, "confirmed_swing_low"]),
                    int(i),
                    "EQL",
                )

            if liquidity_level is not None and np.isfinite(atr_value):
                level, _, _ = liquidity_level
                result.at[i, "liquidity_distance_atr"] = abs(close - level) / atr_value

            sweep = frame.at[i, "liquidity_sweep"] if "liquidity_sweep" in frame else None
            if liquidity_level is not None and np.isfinite(atr_value):
                level, _, _ = liquidity_level
                if sweep == "BULLISH_SWEEP":
                    result.at[i, "sweep_magnitude_atr"] = max(level - low, 0.0) / atr_value
                elif sweep == "BEARISH_SWEEP":
                    result.at[i, "sweep_magnitude_atr"] = max(high - level, 0.0) / atr_value

        return result

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy().reset_index(drop=True)
        for column in [
            "bos",
            "choch",
            "fvg",
            "ob_event",
            "liquidity_sweep",
            "liquidity_pool",
            "structure_bias",
        ]:
            if column not in frame:
                frame[column] = None

        atr = self._atr(frame)
        returns = frame["close"].pct_change()
        volume_mean = frame["volume"].rolling(30, min_periods=5).mean()
        volume_std = frame["volume"].rolling(30, min_periods=5).std().replace(0, np.nan)
        ema = frame["close"].ewm(span=50, adjust=False).mean()
        hour = (
            pd.to_datetime(frame["timestamp"], utc=True).dt.hour
            if "timestamp" in frame
            else pd.Series(0, index=frame.index)
        )

        features = pd.DataFrame(index=frame.index)
        features["return_1"] = returns
        features["atr_norm"] = atr / frame["close"].replace(0, np.nan)
        features["range_norm"] = (
            frame["high"] - frame["low"]
        ) / frame["close"].replace(0, np.nan)
        features["body_ratio"] = (
            frame["close"] - frame["open"]
        ).abs() / (frame["high"] - frame["low"]).replace(0, np.nan)
        features["volume_z"] = (frame["volume"] - volume_mean) / volume_std
        features["trend_distance"] = (
            frame["close"] - ema
        ) / frame["close"].replace(0, np.nan)
        features["bos_numeric"] = self._numeric_event(
            frame["bos"], "BULLISH_BOS", "BEARISH_BOS"
        )
        features["choch_numeric"] = self._numeric_event(
            frame["choch"], "BULLISH_CHOCH", "BEARISH_CHOCH"
        )
        features["fvg_numeric"] = self._numeric_event(
            frame["fvg"], "BULLISH_FVG", "BEARISH_FVG"
        )
        features["ob_numeric"] = self._numeric_event(
            frame["ob_event"], "BULLISH_OB", "BEARISH_OB"
        )
        features["sweep_numeric"] = self._numeric_event(
            frame["liquidity_sweep"], "BULLISH_SWEEP", "BEARISH_SWEEP"
        )

        # Retain the original point-in-time zone feature for backward
        # interpretability. Persistent FVG geometry is represented separately.
        fvg_mid = (
            frame.get("fvg_top", pd.Series(np.nan, index=frame.index))
            + frame.get("fvg_bottom", pd.Series(np.nan, index=frame.index))
        ) / 2
        features["zone_distance"] = (
            frame["close"] - fvg_mid
        ) / frame["close"].replace(0, np.nan)
        features["rsi_norm"] = (self._rsi(frame) - 50) / 50
        features["momentum_norm"] = frame["close"].pct_change(5)
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        features["displacement_atr"] = (
            (frame["close"] - frame["open"]).abs()
            / atr.replace(0, np.nan)
        )

        state = self._smc_state_features(frame, atr)
        for column in state.columns:
            features[column] = state[column]

        adx = self._adx(frame)
        features["adx_norm"] = adx / 100.0
        features["atr_percentile"] = self._rolling_percentile(
            atr,
            window=100,
        )
        features["structure_bias_numeric"] = frame["structure_bias"].map(
            {"BULLISH": 1.0, "BEARISH": -1.0}
        ).fillna(0.0)

        return (
            features[FEATURE_COLUMNS]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        features = self.build_features(df)
        return self.scaler.fit_transform(features)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        features = self.build_features(df)
        return self.scaler.transform(features)

    def save_scaler(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str) -> None:
        self.scaler = joblib.load(path)

    def prepare_sequence(self, scaled_features: np.ndarray) -> np.ndarray:
        if len(scaled_features) < self.sequence_length:
            raise ValueError(
                f"Need {self.sequence_length} feature rows, "
                f"got {len(scaled_features)}"
            )
        return scaled_features[-self.sequence_length :]
