from __future__ import annotations

import logging
import uuid

import pandas as pd

from core.risk import RiskEngine
from models.signal import TradingSignal


class ConfluenceEngine:
    """Validate deterministic SMC confluence with optional AI enhancement.

    The trading decision is intentionally independent of the AI subsystem.
    AI is an optional enhancement and is never required for deterministic
    signal generation when disabled in configuration.
    """

    def __init__(self, config: dict) -> None:
        self.minimum_score = float(config["confluence"]["minimum_score"])
        self.weights = config["confluence"]["weights"]
        self.risk = RiskEngine(config)
        self.logger = logging.getLogger("SMC-Confluence")
        self.ai_engine = None

        model_config = config.get("model", {})
        self.ai_enabled = bool(model_config.get("enabled", False))
        if self.ai_enabled:
            # Keep the live deterministic engine independent of the AI runtime.
            from models.inference import ModelInference

            self.ai_engine = ModelInference(config)
            self.logger.info(
                "AI enhancement enabled; available=%s",
                self.ai_engine.available,
            )
        else:
            self.logger.info("AI enhancement disabled; using deterministic SMC engine")

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
        if zone["order_block"] == wanted_ob:
            top, bottom = zone["ob_top"], zone["ob_bottom"]
        else:
            top, bottom = zone["fvg_top"], zone["fvg_bottom"]
        return {"top": float(top), "bottom": float(bottom)} if pd.notna(top) and pd.notna(bottom) else None

    def validate_signal(
        self,
        daily_df: pd.DataFrame,
        h4_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        m15_df: pd.DataFrame,
    ):
        diagnostic = {"decision": "REJECTED", "reason": "NO_SETUP"}
        daily_bias = self._bias(daily_df)
        h4_bias = self._bias(h4_df)
        sweep_tail = m15_df.tail(3)
        sweep_rows = sweep_tail[sweep_tail["liquidity_sweep"].notna()]
        if sweep_rows.empty:
            diagnostic.update(
                {
                    "daily_bias": daily_bias,
                    "h4_bias": h4_bias,
                    "reason": "NO_M15_SWEEP",
                }
            )
            return None, diagnostic

        sweep = sweep_rows.iloc[-1]["liquidity_sweep"]
        side = "LONG" if sweep == "BULLISH_SWEEP" else "SHORT"
        required_bias = "BULLISH" if side == "LONG" else "BEARISH"
        h1_zone = self._latest_aligned_zone(h1_df, side)
        h1_choch_series = h1_df["choch"].dropna()
        m15_bos_series = m15_df["bos"].dropna()
        m15_choch_series = m15_df["choch"].dropna()
        h1_choch = h1_choch_series.iloc[-1] if not h1_choch_series.empty else None
        m15_bos = m15_bos_series.iloc[-1] if not m15_bos_series.empty else None
        m15_choch = m15_choch_series.iloc[-1] if not m15_choch_series.empty else None
        checks = {
            "daily_bias": daily_bias == required_bias,
            "h4_bias": h4_bias == required_bias,
            "h1_setup": h1_zone is not None
            or h1_choch == ("BULLISH_CHOCH" if side == "LONG" else "BEARISH_CHOCH"),
            "m15_sweep": True,
            "m15_confirmation": m15_bos == ("BULLISH_BOS" if side == "LONG" else "BEARISH_BOS")
            or m15_choch == ("BULLISH_CHOCH" if side == "LONG" else "BEARISH_CHOCH"),
        }
        score = sum(float(self.weights[key]) for key, passed in checks.items() if passed)
        diagnostic.update(
            {
                "daily_bias": daily_bias,
                "h4_bias": h4_bias,
                "side": side,
                "checks": checks,
                "confluence_score": score,
                "ai_enabled": self.ai_enabled,
            }
        )
        if score < self.minimum_score:
            diagnostic["reason"] = "CONFLUENCE_BELOW_THRESHOLD"
            return None, diagnostic

        ai_confidence = None
        if self.ai_enabled and self.ai_engine is not None and self.ai_engine.available:
            ai_confidence = self.ai_engine.predict_confidence(m15_df)
            diagnostic["ai_confidence"] = ai_confidence
            threshold = self.ai_engine.decision_threshold
            if ai_confidence < threshold:
                diagnostic["reason"] = "AI_CONFIDENCE_BELOW_THRESHOLD"
                return None, diagnostic
        else:
            diagnostic["ai_status"] = "DISABLED" if not self.ai_enabled else "UNAVAILABLE_OPTIONAL"

        entry = float(sweep_rows.iloc[-1]["close"])
        if self.ai_engine is not None:
            atr_series = self.ai_engine.features._atr(m15_df)
        else:
            high = pd.to_numeric(m15_df["high"], errors="coerce")
            low = pd.to_numeric(m15_df["low"], errors="coerce")
            close = pd.to_numeric(m15_df["close"], errors="coerce")
            previous_close = close.shift(1)
            true_range = pd.concat(
                [
                    high - low,
                    (high - previous_close).abs(),
                    (low - previous_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_series = true_range.rolling(14, min_periods=14).mean()

        atr = float(atr_series.iloc[-1])
        if not pd.notna(atr) or atr <= 0:
            diagnostic["reason"] = "INVALID_ATR"
            return None, diagnostic

        zone_limit = (
            h1_zone["bottom"]
            if side == "LONG" and h1_zone
            else h1_zone["top"]
            if h1_zone
            else None
        )
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
            ai_confidence=ai_confidence,
            reason=f"{side} liquidity sweep with top-down MTF confluence",
        )
        diagnostic.update({"decision": "SIGNAL", "reason": "VALIDATED"})
        return signal, diagnostic
