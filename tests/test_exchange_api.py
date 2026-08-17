import logging

import pandas as pd

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
                [until - 900_000, 100.0, 101.0, 99.0, 100.5, 10.0],
                [until, 100.5, 101.5, 100.0, 101.0, 11.0],
            ]

        if len(self.calls) == 2:
            return [
                [until - 2_700_000, 101.0, 102.0, 100.5, 101.5, 12.0],
                [until - 1_800_000, 101.5, 102.5, 101.0, 102.0, 13.0],
                [until - 900_000, 102.0, 103.0, 101.5, 102.5, 14.0],
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
