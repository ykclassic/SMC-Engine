from __future__ import annotations

import math


class RiskEngine:
    def __init__(self, config: dict) -> None:
        risk = config["risk_management"]
        self.atr_multiplier = float(risk["atr_sl_multiplier"])
        self.rr = float(risk["default_tp_rr"])

    def levels(self, side: str, entry: float, atr: float, zone_limit: float | None = None) -> tuple[float, float]:
        if entry <= 0 or atr <= 0 or not math.isfinite(entry) or not math.isfinite(atr):
            raise ValueError("Entry and ATR must be positive finite values")
        distance = atr * self.atr_multiplier
        if zone_limit is not None and math.isfinite(zone_limit):
            if side == "LONG":
                stop = min(entry - distance, zone_limit)
            else:
                stop = max(entry + distance, zone_limit)
        else:
            stop = entry - distance if side == "LONG" else entry + distance
        risk = abs(entry - stop)
        if risk <= 0:
            raise ValueError("Risk distance must be positive")
        target = entry + risk * self.rr if side == "LONG" else entry - risk * self.rr
        return float(stop), float(target)
