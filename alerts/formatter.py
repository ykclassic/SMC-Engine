import datetime

class SignalFormatter:
    @staticmethod
    def format_discord_embed(signal, symbol, config):
        """Creates a professional Discord Embed for the signal."""
        color = config['discord']['color_long'] if signal['type'] == "LONG" else config['discord']['color_short']
        
        # Calculate Risk/Reward Levels
        entry = signal['entry']
        zone_limit = signal['zone_limit']
        
        # SL is slightly beyond the OB limit
        stop_loss = zone_limit * (0.999 if signal['type'] == "LONG" else 1.001)
        risk = abs(entry - stop_loss)
        take_profit = entry + (risk * config['risk_management']['default_tp_rr']) if signal['type'] == "LONG" else entry - (risk * config['risk_management']['default_tp_rr'])

        embed = {
            "title": f"🚀 SMC {signal['type']} SIGNAL: {symbol}",
            "description": f"**Reasoning:** {signal['reason']}",
            "color": color,
            "fields": [
                {"name": "🎯 Entry Price", "value": f"`{entry:.2f}`", "inline": True},
                {"name": "🛡️ Stop Loss", "value": f"`{stop_loss:.2f}`", "inline": True},
                {"name": "💰 Take Profit", "value": f"`{take_profit:.2f}`", "inline": True},
                {"name": "📊 Risk/Reward", "value": f"1:{config['risk_management']['default_tp_rr']}", "inline": True},
                {"name": "🕒 Timeframe", "value": f"Macro: {config['timeframes']['macro']} | Micro: {config['timeframes']['micro']}", "inline": False}
            ],
            "footer": {"text": "Nexus Intelligence Suite | TechSolute"},
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        return embed
