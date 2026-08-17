from __future__ import annotations

import pandas as pd


class MarketStructureDetector:
    """Deterministic SMC structure detector with no future-bar leakage."""

    def __init__(self, config: dict) -> None:
        self.lookback = max(1, int(config["market_structure"]["lookback_period"]))
        self.confirmation_candles = max(1, int(config["market_structure"].get("confirmation_candles", 1)))

    def detect_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy().reset_index(drop=True)
        frame["is_high"] = 0
        frame["is_low"] = 0
        frame["high_confirmed"] = 0
        frame["low_confirmed"] = 0
        frame["confirmed_swing_high"] = float("nan")
        frame["confirmed_swing_low"] = float("nan")
        window = self.lookback * 2 + 1
        if len(frame) < window:
            return frame

        # A pivot at p is only known at p + lookback. The pivot marker is
        # retained for historical compatibility, but all actionable fields
        # are written on the confirmation candle.
        for pivot in range(self.lookback, len(frame) - self.lookback):
            high_window = frame["high"].iloc[pivot - self.lookback : pivot + self.lookback + 1]
            low_window = frame["low"].iloc[pivot - self.lookback : pivot + self.lookback + 1]
            high = float(frame.at[pivot, "high"])
            low = float(frame.at[pivot, "low"])
            is_high = high == float(high_window.max()) and int((high_window == high).sum()) == 1
            is_low = low == float(low_window.min()) and int((low_window == low).sum()) == 1
            if is_high:
                frame.at[pivot, "is_high"] = 1
                confirmation = pivot + self.lookback
                frame.at[confirmation, "high_confirmed"] = 1
                frame.at[confirmation, "confirmed_swing_high"] = high
            if is_low:
                frame.at[pivot, "is_low"] = 1
                confirmation = pivot + self.lookback
                frame.at[confirmation, "low_confirmed"] = 1
                frame.at[confirmation, "confirmed_swing_low"] = low
        return frame

    def get_structure_points(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = self.detect_fractals(df)
        frame["label"] = None
        last_high = None
        last_low = None
        for idx, row in frame.iterrows():
            if row["high_confirmed"] == 1 and pd.notna(row["confirmed_swing_high"]):
                level = float(row["confirmed_swing_high"])
                frame.at[idx, "label"] = "HH" if last_high is not None and level > last_high else "LH"
                last_high = level
            if row["low_confirmed"] == 1 and pd.notna(row["confirmed_swing_low"]):
                level = float(row["confirmed_swing_low"])
                if frame.at[idx, "label"] is None:
                    frame.at[idx, "label"] = "HL" if last_low is not None and level > last_low else "LL"
                last_low = level
        return frame

    def detect_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["bos"] = None
        frame["choch"] = None
        frame["bos_level"] = float("nan")
        frame["structure_bias"] = None
        protected_high = None
        protected_low = None
        trend = None

        for idx, row in frame.iterrows():
            # A newly confirmed swing becomes a level that can be broken only
            # by a later closed candle. Do not treat the confirmation candle
            # itself as a break of the newly confirmed level.
            if row.get("high_confirmed") == 1 and pd.notna(row.get("confirmed_swing_high")):
                protected_high = float(row["confirmed_swing_high"])
            if row.get("low_confirmed") == 1 and pd.notna(row.get("confirmed_swing_low")):
                protected_low = float(row["confirmed_swing_low"])

            close = float(row["close"])
            if protected_high is not None and close > protected_high:
                frame.at[idx, "bos"] = "BULLISH_BOS"
                frame.at[idx, "bos_level"] = protected_high
                if trend == "BEARISH":
                    frame.at[idx, "choch"] = "BULLISH_CHOCH"
                trend = "BULLISH"
                protected_high = None
            elif protected_low is not None and close < protected_low:
                frame.at[idx, "bos"] = "BEARISH_BOS"
                frame.at[idx, "bos_level"] = protected_low
                if trend == "BULLISH":
                    frame.at[idx, "choch"] = "BEARISH_CHOCH"
                trend = "BEARISH"
                protected_low = None
            frame.at[idx, "structure_bias"] = trend
        return frame

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        return self.detect_bos(self.get_structure_points(df))


MarketStructure = MarketStructureDetector
