from datetime import datetime, timezone

from utils.event_dedup import has_signal_event
from utils.journal import SignalJournal


def _signal(signal_id: str, timestamp: str, trigger: str) -> dict:
    return {
        "signal_id": signal_id,
        "timestamp": timestamp,
        "symbol": "ETH/USDT:USDT",
        "side": "LONG",
        "m15_trigger": trigger,
    }


def test_exact_causal_event_is_suppressed(tmp_path) -> None:
    path = str(tmp_path / "journal.sqlite3")
    journal = SignalJournal(path)
    timestamp = datetime.now(timezone.utc).isoformat()
    signal = _signal("s1", timestamp, "BULLISH_SWEEP")

    journal.record_signal("scan-1", signal, "DELIVERED")

    assert has_signal_event(
        path,
        signal["symbol"],
        signal["side"],
        signal["m15_trigger"],
        signal["timestamp"],
    ) is True


def test_different_event_timestamp_is_not_suppressed(tmp_path) -> None:
    path = str(tmp_path / "journal.sqlite3")
    journal = SignalJournal(path)
    first_timestamp = datetime.now(timezone.utc).isoformat()
    signal = _signal("s1", first_timestamp, "BULLISH_SWEEP")

    journal.record_signal("scan-1", signal, "DELIVERED")

    second_timestamp = "2026-08-20T07:18:00+00:00"
    assert has_signal_event(
        path,
        signal["symbol"],
        signal["side"],
        signal["m15_trigger"],
        second_timestamp,
    ) is False


def test_different_trigger_is_not_suppressed(tmp_path) -> None:
    path = str(tmp_path / "journal.sqlite3")
    journal = SignalJournal(path)
    timestamp = datetime.now(timezone.utc).isoformat()
    signal = _signal("s1", timestamp, "BULLISH_SWEEP")

    journal.record_signal("scan-1", signal, "DELIVERED")

    assert has_signal_event(
        path,
        signal["symbol"],
        signal["side"],
        "BULLISH_CHOCH",
        timestamp,
    ) is False
