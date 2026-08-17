from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_VERSION = "smc-v3"
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
]


class FeatureEngineer:
    def __init__(self, sequence_length: int = 32) -> None:
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        previous_close = df["close"].shift(1)
        true_range = pd.concat([
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        return true_range.rolling(period, min_periods=period).mean()

    @staticmethod
    def _rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy().reset_index(drop=True)
        for column in ["bos", "choch", "fvg", "ob_event", "liquidity_sweep"]:
            if column not in frame:
                frame[column] = None

        atr = self._atr(frame)
        returns = frame["close"].pct_change()
        volume_mean = frame["volume"].rolling(30, min_periods=5).mean()
        volume_std = frame["volume"].rolling(30, min_periods=5).std().replace(0, np.nan)
        ema = frame["close"].ewm(span=50, adjust=False).mean()
        hour = pd.to_datetime(frame["timestamp"], utc=True).dt.hour if "timestamp" in frame else pd.Series(0, index=frame.index)

        features = pd.DataFrame(index=frame.index)
        features["return_1"] = returns
        features["atr_norm"] = atr / frame["close"].replace(0, np.nan)
        features["range_norm"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
        features["body_ratio"] = (frame["close"] - frame["open"]).abs() / (frame["high"] - frame["low"]).replace(0, np.nan)
        features["volume_z"] = (frame["volume"] - volume_mean) / volume_std
        features["trend_distance"] = (frame["close"] - ema) / frame["close"].replace(0, np.nan)
        features["bos_numeric"] = frame["bos"].map({"BULLISH_BOS": 1, "BEARISH_BOS": -1}).fillna(0)
        features["choch_numeric"] = frame["choch"].map({"BULLISH_CHOCH": 1, "BEARISH_CHOCH": -1}).fillna(0)
        features["fvg_numeric"] = frame["fvg"].map({"BULLISH_FVG": 1, "BEARISH_FVG": -1}).fillna(0)
        # An order block is only knowable after its BOS confirmation. The
        # origin candle is therefore not used as a feature marker.
        features["ob_numeric"] = frame["ob_event"].map({"BULLISH_OB": 1, "BEARISH_OB": -1}).fillna(0)
        features["sweep_numeric"] = frame["liquidity_sweep"].map({"BULLISH_SWEEP": 1, "BEARISH_SWEEP": -1}).fillna(0)

        # Use only a zone that exists on the current confirmed candle. OB
        # origin levels are deliberately excluded because they are discovered
        # retrospectively when a later BOS occurs.
        fvg_mid = (frame.get("fvg_top", pd.Series(np.nan, index=frame.index)) + frame.get("fvg_bottom", pd.Series(np.nan, index=frame.index))) / 2
        features["zone_distance"] = (frame["close"] - fvg_mid) / frame["close"].replace(0, np.nan)
        features["rsi_norm"] = (self._rsi(frame) - 50) / 50
        features["momentum_norm"] = frame["close"].pct_change(5)
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        return features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
            raise ValueError(f"Need {self.sequence_length} feature rows, got {len(scaled_features)}")
        return scaled_features[-self.sequence_length :]
