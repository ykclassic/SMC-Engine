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

FALLBACK_EXCHANGE_SECRETS = {
    "XT_API_KEY": "fallback_api_key",
    "XT_API_SECRET": "fallback_api_secret",
    "XT_PASSPHRASE": "fallback_passphrase",
}

NOTIFICATION_SECRETS = {"DISCORD_WEBHOOK_URL": "discord_webhook_url"}


def load_all_configs(
    require_secrets: bool = True,
    require_notifications: bool = True,
) -> dict:
    load_dotenv()
    logger = logging.getLogger("SMC-Config")
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "settings.yaml"
    )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    for section in (
        "trading",
        "market_data",
        "model",
        "risk_management",
        "notifications",
        "journal",
    ):
        config.setdefault(section, {})

    config["market_data"].setdefault("timeframes", {})
    config["market_data"].setdefault("limits", {})

    for env_name, key in EXCHANGE_SECRETS.items():
        config[key] = os.getenv(env_name)

    for env_name, key in FALLBACK_EXCHANGE_SECRETS.items():
        config[key] = os.getenv(env_name)

    for env_name, key in NOTIFICATION_SECRETS.items():
        config[key] = os.getenv(env_name)

    required = {}

    if require_secrets:
        required.update(EXCHANGE_SECRETS)

    if require_notifications:
        required.update(NOTIFICATION_SECRETS)

    missing = [
        name
        for name in required
        if not os.getenv(name)
    ]

    if missing:
        raise EnvironmentError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    fallback_configured = all(
        bool(os.getenv(name))
        for name in FALLBACK_EXCHANGE_SECRETS
    )

    configured_fallback_enabled = bool(
        config["trading"].get(
            "fallback_exchange_enabled",
            False,
        )
    )

    if configured_fallback_enabled and not fallback_configured:
        raise EnvironmentError(
            "trading.fallback_exchange_enabled is true, but XT_API_KEY, "
            "XT_API_SECRET, and XT_PASSPHRASE are not all configured"
        )

    config["trading"]["fallback_exchange_enabled"] = (
        configured_fallback_enabled
        and fallback_configured
    )

    if not config["trading"].get("symbols"):
        raise ValueError(
            "trading.symbols must contain at least one CCXT symbol"
        )

    for timeframe_key in ("daily", "h4", "h1", "m15"):
        if timeframe_key not in config["market_data"]["timeframes"]:
            raise ValueError(
                f"market_data.timeframes.{timeframe_key} is required"
            )

    primary = config["trading"].get("exchange", "bitget").lower()
    fallback = config["trading"].get("fallback_exchange", "xt").lower()

    if primary != "bitget":
        raise ValueError(
            "Bitget must remain the primary exchange for the current credential configuration"
        )

    if fallback != "xt":
        raise ValueError(
            "The configured fallback exchange must be xt"
        )

    logger.info(
        "Configuration loaded: primary exchange=%s, fallback exchange=%s, "
        "fallback_enabled=%s, fallback_credentials=%s.",
        primary,
        fallback,
        config["trading"]["fallback_exchange_enabled"],
        fallback_configured,
    )

    return config
