from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TradingSignal:
    signal_id: str
    symbol: str
    side: str
    timestamp: str
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    timeframe: str
    daily_bias: str
    h4_bias: str
    h1_setup: str
    m15_trigger: str
    confluence_score: float
    ai_confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
