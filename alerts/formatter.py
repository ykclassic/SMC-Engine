from __future__ import annotations

from datetime import datetime, timezone

from models.signal import TradingSignal


class SignalFormatter:
    @staticmethod
    def format_discord_embed(signal: TradingSignal | dict, config: dict) -> dict:
        data = signal.to_dict() if isinstance(signal, TradingSignal) else signal
        color = config["discord"]["color_long"] if data["side"] == "LONG" else config["discord"]["color_short"]
        return {
            "title": f"SMC {data['side']} SIGNAL: {data['symbol']}",
            "description": f"**Reasoning:** {data['reason']}",
            "color": color,
            "fields": [
                {"name": "Entry", "value": f"`{data['entry']:.6f}`", "inline": True},
                {"name": "Stop Loss", "value": f"`{data['stop_loss']:.6f}`", "inline": True},
                {"name": "Take Profit", "value": f"`{data['take_profit']:.6f}`", "inline": True},
                {"name": "Risk/Reward", "value": f"1:{data['risk_reward']:.2f}", "inline": True},
                {"name": "AI Confidence", "value": f"{data['ai_confidence'] * 100:.1f}%", "inline": True},
                {"name": "Confluence", "value": f"{data['confluence_score'] * 100:.1f}%", "inline": True},
                {"name": "MTF Bias", "value": f"Daily: {data['daily_bias']} | H4: {data['h4_bias']}", "inline": False},
                {"name": "Trigger", "value": f"{data['m15_trigger']} | {data['timeframe']}", "inline": False},
            ],
            "footer": {"text": "SMC Engine | TechSolute"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
