from __future__ import annotations

import numpy as np
import pandas as pd


class ZoneEngine:
    def __init__(self, config: dict) -> None:
        self.min_gap = float(config["imbalance"]["min_gap_size"])
        self.require_fvg = bool(config["order_blocks"]["require_fvg"])
        self.lookback = int(config["order_blocks"]["lookback_candles"])

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["fvg"] = None
        frame["fvg_top"] = np.nan
        frame["fvg_bottom"] = np.nan
        for i in range(2, len(frame)):
            if frame.at[i, "low"] > frame.at[i - 2, "high"]:
                gap = (frame.at[i, "low"] - frame.at[i - 2, "high"]) / frame.at[i - 2, "high"]
                if gap >= self.min_gap:
                    frame.at[i - 1, "fvg"] = "BULLISH_FVG"
                    frame.at[i - 1, "fvg_top"] = frame.at[i, "low"]
                    frame.at[i - 1, "fvg_bottom"] = frame.at[i - 2, "high"]
            elif frame.at[i, "high"] < frame.at[i - 2, "low"]:
                gap = (frame.at[i - 2, "low"] - frame.at[i, "high"]) / frame.at[i - 2, "low"]
                if gap >= self.min_gap:
                    frame.at[i - 1, "fvg"] = "BEARISH_FVG"
                    frame.at[i - 1, "fvg_top"] = frame.at[i - 2, "low"]
                    frame.at[i - 1, "fvg_bottom"] = frame.at[i, "high"]
        return frame

    def find_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["order_block"] = None
        frame["ob_top"] = np.nan
        frame["ob_bottom"] = np.nan
        frame["ob_mitigated"] = False

        bos_rows = frame[frame["bos"].notna()]
        for idx in bos_rows.index:
            position = frame.index.get_loc(idx)
            bos_type = frame.at[idx, "bos"]
            start = max(0, position - self.lookback)
            candidates = frame.iloc[start:position].iloc[::-1]
            wanted = "BULLISH_OB" if bos_type == "BULLISH_BOS" else "BEARISH_OB" if bos_type == "BEARISH_BOS" else None
            if wanted is None:
                continue
            for candidate_idx, candle in candidates.iterrows():
                is_origin = candle["close"] < candle["open"] if wanted == "BULLISH_OB" else candle["close"] > candle["open"]
                if not is_origin:
                    continue
                frame.at[candidate_idx, "order_block"] = wanted
                frame.at[candidate_idx, "ob_top"] = float(candle["high"])
                frame.at[candidate_idx, "ob_bottom"] = float(candle["low"])
                break

        return self._check_mitigation(frame)

    def _check_mitigation(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        for idx in frame.index[frame["order_block"].notna()]:
            position = frame.index.get_loc(idx)
            future = frame.iloc[position + 1 :]
            if future.empty:
                continue
            top = float(frame.at[idx, "ob_top"])
            bottom = float(frame.at[idx, "ob_bottom"])
            touched = ((future["low"] <= top) & (future["high"] >= bottom)).any()
            frame.at[idx, "ob_mitigated"] = bool(touched)
        return frame
