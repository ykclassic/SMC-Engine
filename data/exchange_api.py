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

    def _normalize(self, raw: list) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        numeric = ["open", "high", "low", "close", "volume"]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        return df.dropna(subset=numeric).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    def _drop_open_candle(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        now = pd.Timestamp(datetime.now(timezone.utc))
        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
        if not df.empty and (now.value // 1_000_000) < (df.iloc[-1]["timestamp"].value // 1_000_000) + tf_ms:
            return df.iloc[:-1].reset_index(drop=True)
        return df

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol {symbol} is not available on Bitget")
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=min(limit, 1000))
        df = self._drop_open_candle(self._normalize(raw), timeframe)
        if df.empty:
            raise ValueError(f"No closed candles available for {symbol} {timeframe}")
        return df

    def fetch_ohlcv_history(self, symbol: str, timeframe: str, candles: int = 5000) -> pd.DataFrame:
        """Paginate public OHLCV until the requested historical depth is reached."""
        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol {symbol} is not available on Bitget")
        candles = max(1, int(candles))
        page_size = min(1000, candles)
        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
        all_rows: list[list] = []
        since = None
        while len(all_rows) < candles:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=page_size)
            if not batch:
                break
            all_rows.extend(batch)
            last_timestamp = batch[-1][0]
            next_since = last_timestamp + timeframe_ms
            if since is not None and next_since <= since:
                break
            since = next_since
            if len(batch) < page_size:
                break
        frame = self._drop_open_candle(self._normalize(all_rows), timeframe)
        if len(frame) > candles:
            frame = frame.iloc[-candles:].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No historical closed candles available for {symbol} {timeframe}")
        return frame

    def get_account_balance(self) -> float:
        balance = self.exchange.fetch_balance()
        return float(balance.get("total", {}).get("USDT", 0.0))
