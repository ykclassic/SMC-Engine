from __future__ import annotations

import numpy as np
import pandas as pd


class ZoneEngine:
    def __init__(self, config: dict) -> None:
        self.min_gap = float(config["imbalance"]["min_gap_size"])
        self.require_fvg = bool(config["order_blocks"]["require_fvg"])
        self.lookback = max(5, int(config["order_blocks"]["lookback_candles"]))

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy().reset_index(drop=True)
        frame["fvg"] = None; frame["fvg_top"] = np.nan; frame["fvg_bottom"] = np.nan; frame["fvg_origin"] = np.nan
        for i in range(2, len(frame)):
            left_high = float(frame.at[i - 2, "high"]); left_low = float(frame.at[i - 2, "low"])
            low = float(frame.at[i, "low"]); high = float(frame.at[i, "high"])
            if low > left_high and (low - left_high) / max(abs(left_high), 1e-12) >= self.min_gap:
                frame.at[i, "fvg"] = "BULLISH_FVG"; frame.at[i, "fvg_top"] = low; frame.at[i, "fvg_bottom"] = left_high; frame.at[i, "fvg_origin"] = i - 1
            elif high < left_low and (left_low - high) / max(abs(left_low), 1e-12) >= self.min_gap:
                frame.at[i, "fvg"] = "BEARISH_FVG"; frame.at[i, "fvg_top"] = left_low; frame.at[i, "fvg_bottom"] = high; frame.at[i, "fvg_origin"] = i - 1
        return frame

    def find_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy().reset_index(drop=True)
        frame["order_block"] = None; frame["ob_top"] = np.nan; frame["ob_bottom"] = np.nan; frame["ob_mitigated"] = False; frame["ob_event"] = None; frame["ob_origin_index"] = np.nan
        for idx in frame.index[frame["bos"].notna()].tolist():
            wanted = "BULLISH_OB" if frame.at[idx, "bos"] == "BULLISH_BOS" else "BEARISH_OB" if frame.at[idx, "bos"] == "BEARISH_BOS" else None
            if wanted is None: continue
            start = max(0, int(idx) - self.lookback)
            for candidate_idx, candle in frame.iloc[start:idx].iloc[::-1].iterrows():
                is_origin = float(candle["close"]) < float(candle["open"]) if wanted == "BULLISH_OB" else float(candle["close"]) > float(candle["open"])
                if not is_origin: continue
                if self.require_fvg:
                    fvg_match = "BULLISH_FVG" if wanted == "BULLISH_OB" else "BEARISH_FVG"
                    if not frame.iloc[start:idx]["fvg"].eq(fvg_match).any(): continue
                frame.at[candidate_idx, "order_block"] = wanted; frame.at[candidate_idx, "ob_top"] = float(candle["high"]); frame.at[candidate_idx, "ob_bottom"] = float(candle["low"])
                frame.at[idx, "ob_event"] = wanted; frame.at[idx, "ob_origin_index"] = candidate_idx
                break
        return self._check_mitigation(frame)

    def _check_mitigation(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        for idx in frame.index[frame["order_block"].notna()]:
            future = frame.iloc[int(idx) + 1:]
            if future.empty: continue
            top = float(frame.at[idx, "ob_top"]); bottom = float(frame.at[idx, "ob_bottom"])
            frame.at[idx, "ob_mitigated"] = bool(((future["low"] <= top) & (future["high"] >= bottom)).any())
        return frame
