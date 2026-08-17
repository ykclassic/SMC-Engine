from __future__ import annotations

import logging

import pandas as pd


class LiquidityEngine:
    def __init__(self, config: dict) -> None:
        liquidity = config["liquidity"]
        self.threshold = float(liquidity.get("eq_threshold", 0.001))
        self.sweep_buffer = float(liquidity.get("sweep_buffer", 0.0005))
        self.search_depth = max(5, int(liquidity.get("search_depth", 50)))
        self.logger = logging.getLogger("SMC-Liquidity")

    def identify_liquidity_pools(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy().reset_index(drop=True)
        frame["liquidity_pool"] = None
        confirmed_highs = []
        confirmed_lows = []
        for i in range(len(frame)):
            if frame.at[i, "high_confirmed"] == 1:
                confirmed_highs.append((i, float(frame.at[i, "confirmed_swing_high"])))
            if frame.at[i, "low_confirmed"] == 1:
                confirmed_lows.append((i, float(frame.at[i, "confirmed_swing_low"])))

            recent_highs = confirmed_highs[-2:]
            if len(recent_highs) == 2:
                (_, first), (second_i, second) = recent_highs
                if abs(second - first) / max(abs(first), 1e-12) <= self.threshold:
                    frame.at[second_i, "liquidity_pool"] = "EQH"

            recent_lows = confirmed_lows[-2:]
            if len(recent_lows) == 2:
                (_, first), (second_i, second) = recent_lows
                if abs(second - first) / max(abs(first), 1e-12) <= self.threshold:
                    frame.at[second_i, "liquidity_pool"] = "EQL"
        return frame

    def detect_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy().reset_index(drop=True)
        frame["liquidity_sweep"] = None
        recent_high = None
        recent_low = None
        recent_high_index = None
        recent_low_index = None

        for i in range(len(frame)):
            start = max(0, i - self.search_depth)
            history = frame.iloc[start:i]
            confirmed_highs = history[history["high_confirmed"] == 1]
            confirmed_lows = history[history["low_confirmed"] == 1]
            if not confirmed_highs.empty:
                recent_high_index = int(confirmed_highs.index[-1])
                recent_high = float(confirmed_highs.iloc[-1]["confirmed_swing_high"])
            if not confirmed_lows.empty:
                recent_low_index = int(confirmed_lows.index[-1])
                recent_low = float(confirmed_lows.iloc[-1]["confirmed_swing_low"])

            row = frame.iloc[i]
            # Buffer is interpreted as a percentage of the swept level.
            if recent_low is not None and recent_low_index is not None:
                swept = float(row["low"]) < recent_low * (1.0 - self.sweep_buffer)
                reclaimed = float(row["close"]) > recent_low
                if swept and reclaimed:
                    frame.at[i, "liquidity_sweep"] = "BULLISH_SWEEP"
            if recent_high is not None and recent_high_index is not None:
                swept = float(row["high"]) > recent_high * (1.0 + self.sweep_buffer)
                rejected = float(row["close"]) < recent_high
                if swept and rejected:
                    frame.at[i, "liquidity_sweep"] = "BEARISH_SWEEP"
        return frame
