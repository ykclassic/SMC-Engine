from __future__ import annotations

import logging
import time

import requests


class DiscordNotifier:
    def __init__(self, config: dict) -> None:
        self.webhook_url = config.get("discord_webhook_url")
        notifications = config.get("notifications", {})
        self.timeout = float(notifications.get("timeout_seconds", 10))
        self.max_retries = int(notifications.get("max_retries", 4))
        self.backoff = float(notifications.get("backoff_seconds", 2))
        self.logger = logging.getLogger("SMC-Discord")
        self.session = requests.Session()

    def send_signal(self, embed: dict) -> tuple[bool, str]:
        if not self.webhook_url:
            return False, "WEBHOOK_MISSING"
        payload = {"username": "SMC Engine", "allowed_mentions": {"parse": []}, "embeds": [embed]}
        last_status = "REQUEST_FAILED"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(self.webhook_url, json=payload, timeout=self.timeout)
                if response.status_code in (200, 204):
                    self.logger.info("Discord signal delivered on attempt %d.", attempt)
                    return True, "DELIVERED"
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", self.backoff))
                    last_status = "RATE_LIMITED"
                    time.sleep(min(retry_after, 60.0))
                    continue
                last_status = f"HTTP_{response.status_code}"
                self.logger.error("Discord returned %s: %s", response.status_code, response.text[:500])
            except requests.RequestException as exc:
                last_status = "REQUEST_FAILED"
                self.logger.error("Discord request failed on attempt %d: %s", attempt, exc)
            if attempt < self.max_retries:
                time.sleep(self.backoff * (2 ** (attempt - 1)))
        return False, last_status
