from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


class SignalJournal:
    def __init__(self, path: str = "data/signal_journal.sqlite3") -> None:
        self.path = path
        self.logger = logging.getLogger("SMC-Journal")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS scans (scan_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, symbol TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS signals (signal_id TEXT PRIMARY KEY, scan_id TEXT NOT NULL, created_at TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, delivery_status TEXT NOT NULL, payload_json TEXT NOT NULL, FOREIGN KEY(scan_id) REFERENCES scans(scan_id))")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_side_time ON signals(symbol, side, created_at)")

    def record_scan(self, scan_id: str, timestamp: str, symbol: str, decision: str, reason: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO scans VALUES (?, ?, ?, ?, ?, ?)", (scan_id, timestamp, symbol, decision, reason, json.dumps(payload, default=str)))

    def record_signal(self, scan_id: str, signal: dict[str, Any], delivery_status: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO signals VALUES (?, ?, ?, ?, ?, ?, ?)", (signal["signal_id"], scan_id, signal["timestamp"], signal["symbol"], signal["side"], delivery_status, json.dumps(signal, default=str)))

    def has_recent_signal(self, symbol: str, side: str, cooldown_minutes: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT created_at FROM signals WHERE symbol = ? AND side = ? ORDER BY created_at DESC LIMIT 1",
                (symbol, side),
            ).fetchone()
        if not row:
            return False
        try:
            created = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
        except ValueError:
            return False
        age_minutes = (datetime.now(timezone.utc) - created).total_seconds() / 60.0
        return age_minutes < cooldown_minutes
