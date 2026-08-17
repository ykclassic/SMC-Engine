from __future__ import annotations

import logging
import uuid

import pandas as pd

from core.risk import RiskEngine
from models.inference import ModelInference
from models.signal import TradingSignal


class ConfluenceEngine:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.minimum_score = float(config["confluence"]["minimum_score"])
        self.weights = config["confluence"]["weights"]
        self.min_ai_confidence = float(config["model"]["min_confidence"])
        self.ai_engine = ModelInference(config)
        self.risk = RiskEngine(config)
        self.logger = logging.getLogger("SMC-Confluence")

    @staticmethod
    def _bias(df: pd.DataFrame) -> str:
        if df.empty:
            return "UNKNOWN"
        bos = df["bos"].dropna()
        if not bos.empty:
            latest = bos.iloc[-1]
            if latest == "BULLISH_BOS":
                return "BULLISH"
            if latest == "BEARISH_BOS":
                return "BEARISH"
        bias = df["structure_bias"].dropna() if "structure_bias" in df else pd.Series(dtype=object)
        return str(bias.iloc[-1]) if not bias.empty else "UNKNOWN"

    @staticmethod
    def _latest_aligned_zone(df: pd.DataFrame, side: str) -> dict | None:
        if df.empty:
            return None
        wanted_ob = "BULLISH_OB" if side == "LONG" else "BEARISH_OB"
        wanted_fvg = "BULLISH_FVG" if side == "LONG" else "BEARISH_FVG"
        recent = df.tail(100)
        obs = recent[(recent["order_block"] == wanted_ob) & (~recent["ob_mitigated"])]
        fvgs = recent[recent["fvg"] == wanted_fvg]
        zone = obs.iloc[-1] if not obs.empty else fvgs.iloc[-1] if not fvgs.empty else None
        if zone is None:
            return None
        top = zone.get("ob_top", zone.get("fvg_top"))
        bottom = zone.get("ob_bottom", zone.get("fvg_bottom"))
        return {"top": float(top), "bottom": float(bottom)} if pd.notna(top) and pd.notna(bottom) else None

    def validate_signal(self, daily_df: pd.DataFrame, h4_df: pd.DataFrame, h1_df: pd.DataFrame, m15_df: pd.DataFrame):
        diagnostic = {"decision": "REJECTED", "reason": "NO_SETUP"}
        daily_bias = self._bias(daily_df)
        h4_bias = self._bias(h4_df)
        recent = m15_df.tail(3)
        sweep_rows = recent[recent["liquidity_sweep"].notna()]
        if sweep_rows.empty:
            diagnostic.update({"daily_bias": daily_bias, "h4_bias": h4_bias, "reason": "NO_M15_SWEEP"})
            return None, diagnostic

        sweep = sweep_rows.iloc[-1]["liquidity_sweep"]
        side = "LONG" if sweep == "BULLISH_SWEEP" else "SHORT"
        required_bias = "BULLISH" if side == "LONG" else "BEARISH"
        h1_zone = self._latest_aligned_zone(h1_df, side)
        h1_choch = h1_df["choch"].dropna().iloc[-1] if not h1_df["choch"].dropna().empty else None
        m15_bos = m15_df["bos"].dropna().iloc[-1] if not m15_df["bos"].dropna().empty else None
        m15_choch = m15_df["choch"].dropna().iloc[-1] if not m15_df["choch"].dropna().empty else None

        checks = {
            "daily_bias": daily_bias == required_bias,
            "h4_bias": h4_bias == required_bias,
            "h1_setup": h1_zone is not None or h1_choch == ("BULLISH_CHOCH" if side == "LONG" else "BEARISH_CHOCH"),
            "m15_sweep": True,
            "m15_confirmation": m15_bos == ("BULLISH_BOS" if side == "LONG" else "BEARISH_BOS")
            or m15_choch == ("BULLISH_CHOCH" if side == "LONG" else "BEARISH_CHOCH"),
        }
        score = sum(float(self.weights[key]) for key, passed in checks.items() if passed)
        diagnostic.update({"daily_bias": daily_bias, "h4_bias": h4_bias, "side": side, "checks": checks, "confluence_score": score})
        if score < self.minimum_score:
            diagnostic["reason"] = "CONFLUENCE_BELOW_THRESHOLD"
            self.logger.info("Rejected %s: confluence %.2f < %.2f", side, score, self.minimum_score)
            return None, diagnostic

        confidence = self.ai_engine.predict_confidence(m15_df)
        diagnostic["ai_confidence"] = confidence
        if confidence < self.min_ai_confidence:
            diagnostic["reason"] = "AI_CONFIDENCE_BELOW_THRESHOLD"
            self.logger.info("Rejected %s: AI %.3f < %.3f", side, confidence, self.min_ai_confidence)
            return None, diagnostic

        entry = float(sweep_rows.iloc[-1]["close"])
        atr = float(self.ai_engine.features._atr(m15_df).iloc[-1])
        zone = h1_zone
        zone_limit = zone["bottom"] if side == "LONG" and zone else zone["top"] if zone else None
        stop_loss, take_profit = self.risk.levels(side, entry, atr, zone_limit)
        signal = TradingSignal(
            signal_id=str(uuid.uuid4()),
            symbol="",
            side=side,
            timestamp=str(sweep_rows.iloc[-1]["timestamp"]),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=self.risk.rr,
            timeframe="15m",
            daily_bias=daily_bias,
            h4_bias=h4_bias,
            h1_setup="POI/CHOCH",
            m15_trigger=sweep,
            confluence_score=score,
            ai_confidence=confidence,
            reason=f"{side} liquidity sweep with top-down MTF confluence",
        )
        diagnostic.update({"decision": "SIGNAL", "reason": "VALIDATED"})
        return signal, diagnostic
