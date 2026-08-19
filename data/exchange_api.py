from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import ccxt
import pandas as pd


class ExchangeInterface:
    """Market-data adapter with Bitget primary and XT.com failover.

    Bitget remains the authoritative exchange. XT.com is available only when
    the configured failover policy permits it. Account balance operations are
    intentionally Bitget-only because exchange accounts are independent.
    """

    def __init__(self, config: dict) -> None:
        self.logger = logging.getLogger("SMC-Exchange")
        trading = config["trading"]
        self.market_type = trading.get("market_type", "swap")
        self.primary_name = trading.get("exchange", "bitget").lower()
        self.fallback_name = trading.get(
            "fallback_exchange",
            "xt",
        ).lower()
        self.fallback_enabled = bool(
            trading.get("fallback_exchange_enabled", False)
        )

        if self.primary_name != "bitget":
            raise ValueError(
                "Bitget must remain the primary exchange"
            )

        if self.fallback_name != "xt":
            raise ValueError(
                "XT.com must be the configured fallback exchange"
            )

        self.primary = self._build_exchange(
            "bitget",
            api_key=config.get("api_key"),
            secret=config.get("api_secret"),
            password=config.get("passphrase"),
        )

        self.fallback = None

        if self.fallback_enabled:
            self.fallback = self._build_exchange(
                "xt",
                api_key=config.get("fallback_api_key"),
                secret=config.get("fallback_api_secret"),
                password=config.get("fallback_passphrase"),
            )

        self.exchange = self.primary
        self.active_name = "bitget"

        try:
            self.primary.load_markets()
            self.logger.info(
                "Bitget interface initialized with %d markets.",
                len(self.primary.markets),
            )
        except Exception as exc:
            if not self.fallback_enabled:
                raise RuntimeError(
                    "Bitget initialization failed and XT.com failover is disabled"
                ) from exc

            self.logger.error(
                "Bitget initialization failed: %s. Attempting XT.com fallback.",
                exc,
                exc_info=True,
            )
            self._activate_fallback(exc)

    def _build_exchange(
        self,
        name: str,
        *,
        api_key: str | None,
        secret: str | None,
        password: str | None,
    ) -> Any:
        exchange_class = getattr(ccxt, name)

        credentials = {
            "apiKey": api_key,
            "secret": secret,
            "password": password,
            "enableRateLimit": True,
            "options": {
                "defaultType": self.market_type,
                "recvWindow": 5000,
            },
        }

        return exchange_class(credentials)

    def _activate_fallback(self, reason: Exception) -> None:
        if not self.fallback_enabled or self.fallback is None:
            raise RuntimeError(
                "XT.com fallback is disabled"
            ) from reason

        try:
            self.fallback.load_markets()
        except Exception as fallback_exc:
            raise RuntimeError(
                "Bitget failed and XT.com fallback could not be initialized"
            ) from fallback_exc

        self.exchange = self.fallback
        self.active_name = "xt"

        self.logger.warning(
            "EXCHANGE_FAILOVER primary=bitget fallback=xt reason=%s markets=%d",
            reason,
            len(self.fallback.markets),
        )

    def _ensure_markets(self, exchange: Any) -> None:
        if not exchange.markets:
            exchange.load_markets()

    def _run_with_failover(self, operation: str, callback: Any) -> Any:
        try:
            return callback(self.exchange)
        except Exception as primary_exc:
            active_name = getattr(
                self,
                "active_name",
                getattr(self, "primary_name", "bitget"),
            )

            if active_name != "bitget":
                raise

            if not self.fallback_enabled:
                raise

            self.logger.error(
                "Bitget %s failed: %s. Failing over to XT.com.",
                operation,
                primary_exc,
            )

            self._activate_fallback(primary_exc)

            try:
                return callback(self.exchange)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Bitget {operation} failed and XT.com fallback also failed"
                ) from fallback_exc

    @staticmethod
    def _normalize(raw: list) -> pd.DataFrame:
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

        numeric = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

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
        exchange: Any | None = None,
    ) -> pd.DataFrame:
        exchange = exchange or self.exchange
        now = pd.Timestamp(datetime.now(timezone.utc))
        tf_ms = exchange.parse_timeframe(timeframe) * 1000

        if (
            not df.empty
            and (now.value // 1_000_000)
            < (
                df.iloc[-1]["timestamp"].value // 1_000_000
            )
            + tf_ms
        ):
            return df.iloc[:-1].reset_index(drop=True)

        return df

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        def fetch(exchange: Any) -> pd.DataFrame:
            self._ensure_markets(exchange)

            if symbol not in exchange.markets:
                raise ValueError(
                    f"Symbol {symbol} is not available on {exchange.id}"
                )

            raw = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=min(limit, 1000),
            )

            frame = self._drop_open_candle(
                self._normalize(raw),
                timeframe,
                exchange,
            )

            if frame.empty:
                raise ValueError(
                    f"No closed candles available for "
                    f"{symbol} {timeframe} on {exchange.id}"
                )

            return frame

        return self._run_with_failover(
            f"fetch_ohlcv({symbol}, {timeframe})",
            fetch,
        )

    def fetch_ohlcv_history(
        self,
        symbol: str,
        timeframe: str,
        candles: int = 5000,
    ) -> pd.DataFrame:
        """Fetch closed OHLCV history using backward pagination.

        Pagination deliberately continues after a short page. Some exchanges
        return fewer rows than requested even when older candles remain
        available. Each subsequent request moves the upper time boundary
        backwards from the oldest candle returned by the previous page.
        """

        candles = max(1, int(candles))

        def fetch_history(exchange: Any) -> pd.DataFrame:
            self._ensure_markets(exchange)

            if symbol not in exchange.markets:
                raise ValueError(
                    f"Symbol {symbol} is not available on {exchange.id}"
                )

            timeframe_ms = (
                exchange.parse_timeframe(timeframe) * 1000
            )
            page_size = min(1000, candles)
            until = int(
                datetime.now(timezone.utc).timestamp() * 1000
            )

            rows: list[list] = []
            seen_until: set[int] = set()

            while len(rows) < candles:
                if until in seen_until:
                    break

                seen_until.add(until)

                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    limit=page_size,
                    params={"until": until},
                )

                if not batch:
                    break

                rows.extend(batch)

                oldest = min(int(row[0]) for row in batch)
                next_until = oldest - timeframe_ms

                if next_until >= until:
                    break

                until = next_until

            frame = self._drop_open_candle(
                self._normalize(rows),
                timeframe,
                exchange,
            )

            if len(frame) > candles:
                frame = frame.iloc[-candles:].reset_index(
                    drop=True
                )

            if frame.empty:
                raise ValueError(
                    f"No historical closed candles available for "
                    f"{symbol} {timeframe} on {exchange.id}"
                )

            if len(frame) < candles:
                self.logger.warning(
                    "%s returned %d closed candles for %s %s; requested %d.",
                    exchange.id,
                    len(frame),
                    symbol,
                    timeframe,
                    candles,
                )

            return frame

        return self._run_with_failover(
            f"fetch_ohlcv_history({symbol}, {timeframe})",
            fetch_history,
        )

    def get_account_balance(self) -> float:
        """Return the Bitget account balance only.

        Account state is not failed over to XT.com because the exchanges hold
        separate accounts. Falling back to another account for balance or
        execution would create a materially unsafe trading state.
        """

        if self.active_name != "bitget":
            raise RuntimeError(
                "Bitget account is unavailable; refusing to read balance from XT.com"
            )

        balance = self.primary.fetch_balance()
        return float(
            balance.get("total", {}).get("USDT", 0.0)
        )
