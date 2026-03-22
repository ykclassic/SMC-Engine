import requests
import logging

class DiscordNotifier:
    def __init__(self, config):
        self.webhook_url = config.get('discord_webhook') # Loaded from .env
        self.logger = logging.getLogger(__name__)

    def send_signal(self, embed):
        """Sends the formatted embed to the Discord Webhook."""
        payload = {
            "username": "Nexus SMC Engine",
            "embeds": [embed]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            self.logger.info("Signal successfully pushed to Discord.")
        except Exception as e:
            self.logger.error(f"Failed to send Discord alert: {e}")
