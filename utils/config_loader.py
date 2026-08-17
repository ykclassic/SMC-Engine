from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


REQUIRED_SECRETS = {
    "EXCHANGE_API_KEY": "api_key",
    "EXCHANGE_API_SECRET": "api_secret",
    "EXCHANGE_PASSPHRASE": "passphrase",
    "DISCORD_WEBHOOK_URL": "discord_webhook_url",
}


def load_all_configs(require_secrets: bool = True) -> dict:
    """Load YAML configuration and optionally validate runtime secrets."""
    load_dotenv()
    logger = logging.getLogger("SMC-Config")
    config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    config.setdefault("trading", {})
    config.setdefault("market_data", {})
    config["market_data"].setdefault("timeframes", {})
    config["market_data"].setdefault("limits", {})
    config.setdefault("model", {})
    config.setdefault("risk_management", {})
    config.setdefault("notifications", {})
    config.setdefault("journal", {})

    for env_name, config_key in REQUIRED_SECRETS.items():
        config[config_key] = os.getenv(env_name)

    if require_secrets:
        missing = [name for name in REQUIRED_SECRETS if not os.getenv(name)]
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    symbols = config["trading"].get("symbols", [])
    if not symbols:
        raise ValueError("trading.symbols must contain at least one CCXT symbol")
    for timeframe_key in ("daily", "h4", "h1", "m15"):
        if timeframe_key not in config["market_data"]["timeframes"]:
            raise ValueError(f"market_data.timeframes.{timeframe_key} is required")

    logger.info("Configuration loaded successfully.")
    return config
