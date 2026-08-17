from datetime import datetime, timezone

from utils.journal import SignalJournal


def test_recent_signal_cooldown(tmp_path):
    journal = SignalJournal(str(tmp_path / "journal.sqlite3"))
    signal = {
        "signal_id": "s1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTC/USDT:USDT",
        "side": "LONG",
    }
    journal.record_scan("scan-1", signal["timestamp"], signal["symbol"], "SIGNAL", "VALIDATED", signal)
    journal.record_signal("scan-1", signal, "DELIVERED")
    assert journal.has_recent_signal(signal["symbol"], signal["side"], 30) is True
    assert journal.has_recent_signal(signal["symbol"], "SHORT", 30) is False
