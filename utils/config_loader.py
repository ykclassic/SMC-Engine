from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


EXCHANGE_SECRETS = {
    "EXCHANGE_API_KEY": "api_key",
    "EXCHANGE_API_SECRET": "api_secret",
    "EXCHANGE_PASSPHRASE": "passphrase",
}

NOTIFICATION_SECRETS = {"DISCORD_WEBHOOK_URL": "discord_webhook_url"}


def load_all_configs(require_secrets: bool = True, require_notifications: bool = True) -> dict:
    load_dotenv()
    logger = logging.getLogger("SMC-Config")
    config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    for section in ("trading", "market_data", "model", "risk_management", "notifications", "journal"):
        config.setdefault(section, {})
    config["market_data"].setdefault("timeframes", {})
    config["market_data"].setdefault("limits", {})

    for env_name, key in {**EXCHANGE_SECRETS, **NOTIFICATION_SECRETS}.items():
        config[key] = os.getenv(env_name)

    required = {}
    if require_secrets:
        required.update(EXCHANGE_SECRETS)
    if require_notifications:
        required.update(NOTIFICATION_SECRETS)
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    if not config["trading"].get("symbols"):
        raise ValueError("trading.symbols must contain at least one CCXT symbol")
    for timeframe_key in ("daily", "h4", "h1", "m15"):
        if timeframe_key not in config["market_data"]["timeframes"]:
            raise ValueError(f"market_data.timeframes.{timeframe_key} is required")

    logger.info("Configuration loaded successfully.")
    return config
