import requests
import logging

class DiscordNotifier:
    def __init__(self, config):
        self.webhook_url = config.get('discord_webhook_url')
        self.logger = logging.getLogger("Nexus-Discord")

    def send_signal(self, embed):
        """Sends a trading signal to Discord."""
        if not self.webhook_url:
            self.logger.error("No Webhook URL found. Signal not sent.")
            return

        payload = {
            "username": "Nexus Intelligence Suite",
            "embeds": [embed]
        }

        try:
            response = requests.post(self.webhook_url, json=payload)

            if response.status_code == 204:
                self.logger.info("✅ Signal sent to Discord successfully.")
            else:
                self.logger.error(f"❌ Discord error: {response.status_code} - {response.text}")

        except Exception as e:
            self.logger.error(f"❌ Failed to send signal: {e}")
