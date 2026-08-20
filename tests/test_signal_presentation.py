from alerts.formatter import SignalFormatter
from models.signal import TradingSignal


def _signal() -> TradingSignal:
    return TradingSignal(
        signal_id="signal-1",
        symbol="ETH/USDT:USDT",
        side="LONG",
        timestamp="2026-08-20T07:03:00+00:00",
        entry=2247.79,
        stop_loss=2104.71,
        take_profit=2533.95,
        risk_reward=2.0,
        timeframe="15m",
        daily_bias="BULLISH",
        h4_bias="BULLISH",
        h1_setup="POI/CHOCH",
        m15_trigger="BULLISH_SWEEP",
        confluence_score=0.80,
        ai_confidence=1.0,
        reason="LONG liquidity sweep with top-down MTF confluence",
    )


def test_deterministic_alert_does_not_claim_ai_confidence() -> None:
    config = {
        "model": {"enabled": False},
        "discord": {"color_long": 3066993, "color_short": 15158332},
    }

    embed = SignalFormatter.format_discord_embed(_signal(), config)
    field_names = [field["name"] for field in embed["fields"]]

    assert "AI Confidence" not in field_names
    mode = next(field for field in embed["fields"] if field["name"] == "Signal Mode")
    assert mode["value"] == "DETERMINISTIC SMC | AI DISABLED"


def test_ai_enabled_alert_preserves_ai_confidence() -> None:
    config = {
        "model": {"enabled": True},
        "discord": {"color_long": 3066993, "color_short": 15158332},
    }

    embed = SignalFormatter.format_discord_embed(_signal(), config)
    confidence = next(
        field for field in embed["fields"] if field["name"] == "AI Confidence"
    )

    assert confidence["value"] == "100.0%"
