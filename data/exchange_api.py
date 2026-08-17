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
        self.logger.info(
            "Bitget interface initialized with %d markets.",
            len(self.exchange.markets),
        )

    def _normalize(self, raw: list) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )

        df = pd.DataFrame(
            raw,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
        )
        numeric = ["open", "high", "low", "close", "volume"]
        df[numeric] = df[numeric].apply(
            pd.to_numeric,
            errors="coerce",
        )
        return (
            df.dropna(subset=numeric)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _drop_open_candle(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        now = pd.Timestamp(datetime.now(timezone.utc))
        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000

        if (
            not df.empty
            and (now.value // 1_000_000)
            < (df.iloc[-1]["timestamp"].value // 1_000_000) + tf_ms
        ):
            return df.iloc[:-1].reset_index(drop=True)

        return df

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        if symbol not in self.exchange.markets:
            raise ValueError(
                f"Symbol {symbol} is not available on Bitget"
            )

        raw = self.exchange.fetch_ohlcv(
            symbol,
            timeframe,
            limit=min(limit, 1000),
        )
        df = self._drop_open_candle(
            self._normalize(raw),
            timeframe,
        )

        if df.empty:
            raise ValueError(
                f"No closed candles available for {symbol} {timeframe}"
            )

        return df

    def fetch_ohlcv_history(
        self,
        symbol: str,
        timeframe: str,
        candles: int = 5000,
    ) -> pd.DataFrame:
        """Fetch closed OHLCV history using a backward-moving time cursor.

        Bitget's contract candle APIs have different recent/history limits and
        the history endpoint caps each response at 200 candles. Paging backward
        from the current time lets CCXT select the appropriate endpoint for
        each page and avoids constructing a forward request whose start/end
        range crosses an endpoint boundary incorrectly.

        A short page is not treated as end-of-history. Pagination stops only
        when the exchange returns no data, the requested candle count is met,
        or the cursor fails to move backward.
        """

        if symbol not in self.exchange.markets:
            raise ValueError(
                f"Symbol {symbol} is not available on Bitget"
            )

        candles = max(1, int(candles))
        page_size = min(1000, candles)
        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
        until = int(
            datetime.now(timezone.utc).timestamp() * 1000
        )

        all_rows: list[list] = []
        seen_cursors: set[int] = set()

        while len(all_rows) < candles:
            if until in seen_cursors:
                self.logger.warning(
                    "Bitget OHLCV pagination cursor repeated for %s %s at %s; stopping.",
                    symbol,
                    timeframe,
                    until,
                )
                break

            seen_cursors.add(until)

            batch = self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=page_size,
                params={"until": until},
            )

            if not batch:
                break

            all_rows.extend(batch)

            timestamps = [int(row[0]) for row in batch]
            oldest_timestamp = min(timestamps)
            next_until = oldest_timestamp - 1

            if next_until >= until:
                self.logger.warning(
                    "Bitget OHLCV pagination did not move backward for %s %s: until=%s next_until=%s; stopping.",
                    symbol,
                    timeframe,
                    until,
                    next_until,
                )
                break

            until = next_until

            self.logger.debug(
                "Fetched %d OHLCV rows for %s %s; accumulated=%d/%d; next_until=%d.",
                len(batch),
                symbol,
                timeframe,
                len(all_rows),
                candles,
                until,
            )

        frame = self._drop_open_candle(
            self._normalize(all_rows),
            timeframe,
        )

        if len(frame) > candles:
            frame = frame.iloc[-candles:].reset_index(drop=True)

        if frame.empty:
            raise ValueError(
                f"No historical closed candles available for {symbol} {timeframe}"
            )

        if len(frame) < candles:
            self.logger.warning(
                "Bitget returned %d closed candles for %s %s; requested %d.",
                len(frame),
                symbol,
                timeframe,
                candles,
            )

        return frame

    def get_account_balance(self) -> float:
        balance = self.exchange.fetch_balance()
        return float(
            balance.get("total", {}).get("USDT", 0.0)
        )
