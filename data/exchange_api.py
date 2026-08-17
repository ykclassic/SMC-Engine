from __future__ import annotations

import logging
from datetime import datetime, timezone

import ccxt
import pandas as pd


class ExchangeInterface:
    def __init__(self, config: dict) -> None:
        self.logger = logging.getLogger("SMC-Exchange")
        exchange_name = config["trading"].get("exchange", "bitget").lower()
        if exchange_name != "bitget":
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        self.exchange = ccxt.bitget(
            {
                "apiKey": config.get("api_key"),
                "secret": config.get("api_secret"),
                "password": config.get("passphrase"),
                "enableRateLimit": True,
                "options": {
                    "defaultType": config["trading"].get("market_type", "swap"),
                    "recvWindow": 5000,
                },
            }
        )
        self.exchange.load_markets()
        self.logger.info("Bitget interface initialized with %d markets.", len(self.exchange.markets))

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch closed OHLCV candles only and normalize timestamps to UTC."""
        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol {symbol} is not available on Bitget")
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not raw:
            raise ValueError(f"No OHLCV returned for {symbol} {timeframe}")

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        numeric = ["open", "high", "low", "close", "volume"]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=numeric).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        # The final CCXT candle can still be forming. Exclude it from all decisions.
        now = pd.Timestamp(datetime.now(timezone.utc))
        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
        if not df.empty and (now.value // 1_000_000) < (df.iloc[-1]["timestamp"].value // 1_000_000) + tf_ms:
            df = df.iloc[:-1].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"No closed candles available for {symbol} {timeframe}")
        return df

    def get_account_balance(self) -> float:
        balance = self.exchange.fetch_balance()
        return float(balance.get("total", {}).get("USDT", 0.0))
