"""BrzrkrDB — SQLite-backed persistence layer.

Replaces the scattered JSONL flat-files with a single SQLite database at
``data/brzrkr.db``.  All tables are created on first use (idempotent).

Public helpers
--------------
``get_db()``               — return a thread-local connection (WAL mode).
``init_schema()``          — create tables if they don't exist.
``append_track_record()``  — write one session row.
``read_track_record()``    — return all session rows as dicts.
``write_signal()``         — persist one signal dict.
``read_signals()``         — query recent signals.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/brzrkr.db")
_local = threading.local()

# Override this in tests to point at a temp DB: db._override_path = tmp_path/"test.db"
_override_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_db(path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a thread-local SQLite connection in WAL mode.

    Pass *path* explicitly, or set ``db._override_path`` (useful in tests).
    Falls back to ``data/brzrkr.db``.
    """
    resolved = path or _override_path or _DB_PATH
    # If the override changed, drop the cached connection so we reconnect.
    cached_path = getattr(_local, "path", None)
    conn = getattr(_local, "conn", None)
    if conn is None or cached_path != resolved:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(resolved), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
        _local.path = resolved
        init_schema(conn)
    return conn


def init_schema(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create all tables if they don't exist (idempotent)."""
    c = conn or get_db()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS postmortems (
            id              TEXT PRIMARY KEY,
            category        TEXT NOT NULL,
            title           TEXT NOT NULL,
            description     TEXT,
            symptom         TEXT,
            mitigation      TEXT,
            severity        INTEGER DEFAULT 3,
            source          TEXT DEFAULT 'seed',
            confirmed_count INTEGER DEFAULT 0,
            first_seen      TEXT,
            last_confirmed  TEXT,
            references_json TEXT,
            tags_json       TEXT
        );

        CREATE TABLE IF NOT EXISTS track_record (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            broker          TEXT,
            endpoint        TEXT,
            start_equity    REAL,
            end_equity      REAL,
            pnl             REAL,
            pnl_pct         REAL,
            trades_submitted INTEGER,
            ticks           INTEGER,
            breach_triggered INTEGER DEFAULT 0,
            notes           TEXT,
            lessons_fired_json TEXT
        );

        CREATE TABLE IF NOT EXISTS trade_journal (
            jid             TEXT PRIMARY KEY,
            status          TEXT DEFAULT 'open',
            symbol          TEXT,
            side            TEXT,
            entry_price     REAL,
            stop_price      REAL,
            tp_price        REAL,
            position_size_usd REAL,
            confidence      REAL,
            features_json   TEXT,
            signal_ts       TEXT,
            open_ts         TEXT,
            close_ts        TEXT,
            exit_price      REAL,
            pnl_usd         REAL,
            pnl_pct         REAL,
            r_multiple      REAL,
            outcome         TEXT
        );

        CREATE TABLE IF NOT EXISTS signals (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            asset                 TEXT NOT NULL,
            timestamp             TEXT NOT NULL,
            direction             TEXT,
            entry_price           REAL,
            stop_loss             REAL,
            take_profit           REAL,
            position_size_usd     REAL,
            expected_return_pct   REAL,
            iv_change_pct         REAL,
            confidence            REAL,
            risk_flags_json       TEXT,
            context_features_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trade_journal_symbol
            ON trade_journal (symbol, status);
        CREATE INDEX IF NOT EXISTS idx_signals_asset
            ON signals (asset, timestamp);
        CREATE INDEX IF NOT EXISTS idx_track_record_ts
            ON track_record (ts);
    """)
    c.commit()


# ---------------------------------------------------------------------------
# Track record
# ---------------------------------------------------------------------------

def append_track_record(record: Dict[str, Any], *, db_path: Optional[Path] = None) -> None:
    """Write one session summary row."""
    conn = get_db(db_path)
    conn.execute(
        """INSERT INTO track_record
               (ts, broker, endpoint, start_equity, end_equity, pnl, pnl_pct,
                trades_submitted, ticks, breach_triggered, notes, lessons_fired_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            record.get("ts", ""),
            record.get("broker", ""),
            record.get("endpoint", ""),
            record.get("start_equity"),
            record.get("end_equity"),
            record.get("pnl"),
            record.get("pnl_pct"),
            record.get("trades_submitted"),
            record.get("ticks"),
            int(bool(record.get("breach_triggered", False))),
            record.get("notes", ""),
            json.dumps(record.get("lessons_fired", [])),
        ),
    )
    conn.commit()


def read_track_record(*, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all track-record rows as plain dicts, ordered by ts."""
    conn = get_db(db_path)
    rows = conn.execute(
        "SELECT * FROM track_record ORDER BY ts"
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["lessons_fired"] = json.loads(d.pop("lessons_fired_json") or "[]")
        d["breach_triggered"] = bool(d["breach_triggered"])
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def write_signal(signal: Dict[str, Any]) -> int:
    """Persist a signal dict; returns the new row id."""
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO signals
               (asset, timestamp, direction, entry_price, stop_loss, take_profit,
                position_size_usd, expected_return_pct, iv_change_pct, confidence,
                risk_flags_json, context_features_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            signal.get("asset", ""),
            signal.get("timestamp", ""),
            signal.get("direction", ""),
            signal.get("entry_price"),
            signal.get("stop_loss"),
            signal.get("take_profit"),
            signal.get("position_size_usd"),
            signal.get("expected_return_pct"),
            signal.get("iv_change_pct"),
            signal.get("confidence"),
            json.dumps(signal.get("risk_flags", [])),
            json.dumps(signal.get("context_features", {})),
        ),
    )
    conn.commit()
    return cur.lastrowid


def read_signals(
    asset: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return recent signals, newest first."""
    conn = get_db()
    if asset:
        rows = conn.execute(
            "SELECT * FROM signals WHERE asset=? ORDER BY timestamp DESC LIMIT ?",
            (asset, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["risk_flags"] = json.loads(d.pop("risk_flags_json") or "[]")
        d["context_features"] = json.loads(d.pop("context_features_json") or "{}")
        out.append(d)
    return out
