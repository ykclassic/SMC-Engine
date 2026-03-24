import requests
import datetime

class DiscordNotifier:
    def __init__(self, config):
        # Ensure we pull from the correct Env Var naming
        self.webhook_url = config.get('discord_webhook_url') 
        self.logger = logging.getLogger("Nexus-Discord")

    def send_heartbeat(self, symbol, macro_bias, ai_status):
        """Sends a periodic status update to Discord."""
        if not self.webhook_url:
            self.logger.error("No Webhook URL found. Heartbeat aborted.")
            return

        payload = {
            "username": "Nexus Guardian",
            "embeds": [{
                "title": "💓 Nexus System Heartbeat",
                "description": f"Monitoring **{symbol}** with institutional precision.",
                "color": 0x5865F2, # Discord Blue
                "fields": [
                    {"name": "📈 Macro Trend", "value": f"`{macro_bias}`", "inline": True},
                    {"name": "🤖 AI Engine", "value": f"`{ai_status}`", "inline": True},
                    {"name": "🕒 Last Check", "value": datetime.datetime.now().strftime("%H:%M:%S"), "inline": True}
                ],
                "footer": {"text": "TechSolute | Smart Money Concept App"}
            }]
        }
        requests.post(self.webhook_url, json=payload)
