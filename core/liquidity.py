from __future__ import annotations

import logging

import pandas as pd


class LiquidityEngine:
    def __init__(self, config: dict) -> None:
        liquidity = config["liquidity"]
        self.threshold = float(liquidity.get("eq_threshold", 0.001))
        self.sweep_buffer = float(liquidity.get("sweep_buffer", 0.0005))
        self.search_depth = int(liquidity.get("search_depth", 50))
        self.logger = logging.getLogger("SMC-Liquidity")

    def identify_liquidity_pools(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["liquidity_pool"] = None
        highs = frame[frame["is_high"] == 1]
        lows = frame[frame["is_low"] == 1]

        for current, previous in zip(highs.iloc[1:].itertuples(), highs.iloc[:-1].itertuples()):
            if abs(current.high - previous.high) / max(abs(previous.high), 1e-12) <= self.threshold:
                frame.at[current.Index, "liquidity_pool"] = "EQH"
        for current, previous in zip(lows.iloc[1:].itertuples(), lows.iloc[:-1].itertuples()):
            if abs(current.low - previous.low) / max(abs(previous.low), 1e-12) <= self.threshold:
                frame.at[current.Index, "liquidity_pool"] = "EQL"
        return frame

    def detect_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["liquidity_sweep"] = None
        recent_high = None
        recent_low = None

        for i in range(len(frame)):
            if i > 0:
                start = max(0, i - self.search_depth)
                history = frame.iloc[start:i]
                swing_highs = history[history["is_high"] == 1]
                swing_lows = history[history["is_low"] == 1]
                if not swing_highs.empty:
                    recent_high = float(swing_highs.iloc[-1]["high"])
                if not swing_lows.empty:
                    recent_low = float(swing_lows.iloc[-1]["low"])

                row = frame.iloc[i]
                if recent_low is not None:
                    swept = row["low"] < recent_low * (1.0 - self.sweep_buffer)
                    reclaimed = row["close"] > recent_low
                    if swept and reclaimed:
                        frame.at[frame.index[i], "liquidity_sweep"] = "BULLISH_SWEEP"
                if recent_high is not None:
                    swept = row["high"] > recent_high * (1.0 + self.sweep_buffer)
                    rejected = row["close"] < recent_high
                    if swept and rejected:
                        frame.at[frame.index[i], "liquidity_sweep"] = "BEARISH_SWEEP"

            if frame.iloc[i]["is_high"] == 1:
                recent_high = float(frame.iloc[i]["high"])
            if frame.iloc[i]["is_low"] == 1:
                recent_low = float(frame.iloc[i]["low"])
        return frame
