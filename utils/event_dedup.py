from __future__ import annotations

import json
import sqlite3
from typing import Any


def has_signal_event(
    journal_path: str,
    symbol: str,
    side: str,
    trigger: str,
    event_timestamp: str,
) -> bool:
    """Return True when the exact causal SMC event was already delivered.

    Signal UUIDs are intentionally excluded from the identity because each
    scan creates a new UUID. The market event is identified by symbol, side,
    M15 trigger, and the timestamp of the closed signal candle.
    """
    with sqlite3.connect(journal_path, timeout=10) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM signals "
            "WHERE symbol = ? AND side = ? AND created_at = ?",
            (symbol, side, event_timestamp),
        ).fetchall()

    for (payload_json,) in rows:
        try:
            payload: dict[str, Any] = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if str(payload.get("m15_trigger", "")) == trigger:
            return True
    return False
