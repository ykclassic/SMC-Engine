from __future__ import annotations

import pandas as pd


class MarketStructureDetector:
    """Deterministic, closed-candle market-structure detector."""

    def __init__(self, config: dict) -> None:
        self.lookback = int(config["market_structure"]["lookback_period"])

    def detect_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["is_high"] = 0
        frame["is_low"] = 0
        window = self.lookback * 2 + 1
        if len(frame) < window:
            return frame

        for i in range(self.lookback, len(frame) - self.lookback):
            high_window = frame["high"].iloc[i - self.lookback : i + self.lookback + 1]
            low_window = frame["low"].iloc[i - self.lookback : i + self.lookback + 1]
            if frame["high"].iloc[i] == high_window.max() and (high_window == frame["high"].iloc[i]).sum() == 1:
                frame.at[frame.index[i], "is_high"] = 1
            if frame["low"].iloc[i] == low_window.min() and (low_window == frame["low"].iloc[i]).sum() == 1:
                frame.at[frame.index[i], "is_low"] = 1
        return frame

    def get_structure_points(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = self.detect_fractals(df)
        frame["label"] = None
        last_high = None
        last_low = None
        for idx, row in frame.iterrows():
            if row["is_high"] == 1:
                frame.at[idx, "label"] = "HH" if last_high is not None and row["high"] > last_high else "LH"
                last_high = float(row["high"])
            elif row["is_low"] == 1:
                frame.at[idx, "label"] = "HL" if last_low is not None and row["low"] > last_low else "LL"
                last_low = float(row["low"])
        return frame

    def detect_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["bos"] = None
        frame["choch"] = None
        protected_high = None
        protected_low = None
        trend = None

        for idx, row in frame.iterrows():
            label = row.get("label")
            if label in ("HH", "LH"):
                protected_high = float(row["high"])
            elif label in ("HL", "LL"):
                protected_low = float(row["low"])

            close = float(row["close"])
            if protected_high is not None and close > protected_high:
                frame.at[idx, "bos"] = "BULLISH_BOS"
                if trend == "BEARISH":
                    frame.at[idx, "choch"] = "BULLISH_CHOCH"
                trend = "BULLISH"
                protected_high = None
            elif protected_low is not None and close < protected_low:
                frame.at[idx, "bos"] = "BEARISH_BOS"
                if trend == "BULLISH":
                    frame.at[idx, "choch"] = "BEARISH_CHOCH"
                trend = "BEARISH"
                protected_low = None

        frame["structure_bias"] = trend
        return frame

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.detect_bos(self.get_structure_points(df))


# Backward-compatible name used by existing modules.
MarketStructure = MarketStructureDetector
