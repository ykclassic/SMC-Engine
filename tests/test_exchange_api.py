import logging

import pandas as pd
import pytest

from data.exchange_api import ExchangeInterface


class FakeBitget:
    markets = {"BTC/USDT:USDT": {}}

    def __init__(self):
        self.calls = []

    @staticmethod
    def parse_timeframe(timeframe):
        assert timeframe == "15m"
        return 900

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        until = (params or {}).get("until")
        self.calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "since": since,
                "until": until,
                "limit": limit,
            }
        )

        if len(self.calls) == 1:
            return [
                [until - 1_800_000, 100.0, 101.0, 99.0, 100.5, 10.0],
                [until - 900_000, 100.5, 101.5, 100.0, 101.0, 11.0],
            ]

        if len(self.calls) == 2:
            return [
                [until - 4_500_000, 101.0, 102.0, 100.5, 101.5, 12.0],
                [until - 3_600_000, 101.5, 102.5, 101.0, 102.0, 13.0],
                [until - 2_700_000, 102.0, 103.0, 101.5, 102.5, 14.0],
            ]

        return []


def test_history_pagination_continues_after_short_page():
    interface = ExchangeInterface.__new__(ExchangeInterface)
    interface.logger = logging.getLogger("test")
    interface.exchange = FakeBitget()

    frame = interface.fetch_ohlcv_history(
        "BTC/USDT:USDT",
        "15m",
        candles=5,
    )

    assert len(frame) == 5
    assert frame["timestamp"].is_monotonic_increasing
    assert len(interface.exchange.calls) == 2
    assert interface.exchange.calls[1]["until"] < interface.exchange.calls[0]["until"]
    assert all(call["limit"] == 5 for call in interface.exchange.calls)
    assert list(frame["close"]) == [101.5, 102.0, 102.5, 100.5, 101.0]
    assert isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype)


class IncompleteBitget(FakeBitget):
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        self.calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "since": since,
                "until": (params or {}).get("until"),
                "limit": limit,
            }
        )
        timestamp = 1_700_000_000_000
        return [
            [timestamp, 100.0, 101.0, 99.0, 100.5, 10.0],
        ]


def test_history_fails_closed_when_exchange_cannot_supply_requested_history():
    interface = ExchangeInterface.__new__(ExchangeInterface)
    interface.logger = logging.getLogger("test")
    interface.exchange = IncompleteBitget()

    with pytest.raises(
        RuntimeError,
        match="Incomplete closed-candle history|no new candles",
    ):
        interface.fetch_ohlcv_history(
            "BTC/USDT:USDT",
            "15m",
            candles=5,
        )


def test_history_requests_are_capped_per_page_and_exactly_trimmed():
    class PagedBitget:
        markets = {"BTC/USDT:USDT": {}}

        def __init__(self):
            self.calls = []

        @staticmethod
        def parse_timeframe(timeframe):
            return 86_400

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
            until = int((params or {}).get("until"))
            self.calls.append((limit, until))
            page = limit
            start = until - page * 86_400_000
            return [
                [
                    start + index * 86_400_000,
                    100.0,
                    101.0,
                    99.0,
                    100.5,
                    10.0,
                ]
                for index in range(page)
            ]

    interface = ExchangeInterface.__new__(ExchangeInterface)
    interface.logger = logging.getLogger("test")
    interface.exchange = PagedBitget()

    frame = interface.fetch_ohlcv_history(
        "BTC/USDT:USDT",
        "1d",
        candles=250,
    )

    assert len(frame) == 250
    assert len(interface.exchange.calls) == 2
    assert all(limit == 200 for limit, _ in interface.exchange.calls[:1])
    assert interface.exchange.calls[1][0] == 200
    assert frame["timestamp"].is_monotonic_increasing


def test_history_does_not_use_fallback_when_disabled():
    interface = ExchangeInterface.__new__(ExchangeInterface)
    interface.logger = logging.getLogger("test")
    interface.exchange = IncompleteBitget()
    interface.active_name = "bitget"
    interface.primary_name = "bitget"
    interface.fallback_enabled = False
    interface.fallback = None

    with pytest.raises(RuntimeError):
        interface.fetch_ohlcv_history(
            "BTC/USDT:USDT",
            "15m",
            candles=5,
        )

    assert interface.active_name == "bitget"
