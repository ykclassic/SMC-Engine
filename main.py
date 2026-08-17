from __future__ import annotations

import json
import logging
import sys
import uuid
from dataclasses import replace

from alerts.discord_bot import DiscordNotifier
from alerts.formatter import SignalFormatter
from core.engine import SMCEngine
from data.exchange_api import ExchangeInterface
from models.signal import TradingSignal
from utils.config_loader import load_all_configs
from utils.journal import SignalJournal


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("SMC-Main")
    config = load_all_configs()
    exchange = ExchangeInterface(config)
    engine = SMCEngine(config)
    notifier = DiscordNotifier(config)
    journal = SignalJournal(config["journal"]["path"])

    symbols = config["trading"]["symbols"]
    timeframes = config["market_data"]["timeframes"]
    limits = config["market_data"]["limits"]
    summary = {"scanned": 0, "candidates": 0, "signals_sent": 0, "rejected": 0, "data_errors": 0, "delivery_errors": 0}

    for symbol in symbols:
        scan_id = str(uuid.uuid4())
        summary["scanned"] += 1
        try:
            daily = exchange.fetch_ohlcv(symbol, timeframes["daily"], limits["daily"])
            h4 = exchange.fetch_ohlcv(symbol, timeframes["h4"], limits["h4"])
            h1 = exchange.fetch_ohlcv(symbol, timeframes["h1"], limits["h1"])
            m15 = exchange.fetch_ohlcv(symbol, timeframes["m15"], limits["m15"])
            minimums = config["market_data"]["minimum_rows"]
            for name, frame in (("daily", daily), ("h4", h4), ("h1", h1), ("m15", m15)):
                if len(frame) < int(minimums[name]):
                    raise ValueError(f"Insufficient {name} candles: {len(frame)}")

            signal, diagnostic, *_ = engine.process_market(daily, h4, h1, m15)
            diagnostic_payload = {**diagnostic, "symbol": symbol}
            if signal is None:
                summary["rejected"] += 1
                journal.record_scan(scan_id, TradingSignal.now_iso(), symbol, "REJECTED", diagnostic.get("reason", "UNKNOWN"), diagnostic_payload)
                logger.info("%s rejected: %s", symbol, diagnostic.get("reason"))
                continue

            summary["candidates"] += 1
            signal = replace(signal, symbol=symbol)
            journal.record_scan(scan_id, signal.timestamp, symbol, "SIGNAL", "VALIDATED", diagnostic_payload)
            embed = SignalFormatter.format_discord_embed(signal, config)
            delivered, status = notifier.send_signal(embed)
            journal.record_signal(scan_id, signal.to_dict(), status)
            if delivered:
                summary["signals_sent"] += 1
            else:
                summary["delivery_errors"] += 1
                logger.error("Signal %s was validated but Discord delivery failed: %s", signal.signal_id, status)
        except Exception as exc:
            summary["data_errors"] += 1
            journal.record_scan(scan_id, TradingSignal.now_iso(), symbol, "ERROR", str(exc), {"symbol": symbol})
            logger.error("Scan failed for %s: %s", symbol, exc, exc_info=True)

    logger.info("SCAN_SUMMARY %s", json.dumps(summary, sort_keys=True))
    # No-signal cycles are valid. Infrastructure/data/delivery failures are not.
    return 1 if summary["data_errors"] or summary["delivery_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
