"""migrate.py — one-time import of JSONL flat-files → SQLite.

Run once after upgrading:

    python -m src.data.migrate

Safe to run again: already-imported rows are skipped (INSERT OR IGNORE).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _migrate_postmortems(conn, path: Path) -> int:
    if not path.exists():
        logger.info("No postmortems.jsonl found — skipping.")
        return 0
    count = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO postmortems
                   (id, category, title, description, symptom, mitigation,
                    severity, source, confirmed_count, first_seen, last_confirmed,
                    references_json, tags_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d.get("id", ""),
                d.get("category", ""),
                d.get("title", ""),
                d.get("description", ""),
                d.get("symptom", ""),
                d.get("mitigation", ""),
                int(d.get("severity", 3)),
                d.get("source", "seed"),
                int(d.get("confirmed_count", 0)),
                d.get("first_seen", ""),
                d.get("last_confirmed", ""),
                json.dumps(d.get("references", [])),
                json.dumps(d.get("tags", [])),
            ),
        )
        count += 1
    conn.commit()
    return count


def _migrate_track_record(conn, path: Path) -> int:
    if not path.exists():
        logger.info("No track_record.jsonl found — skipping.")
        return 0
    count = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        conn.execute(
            """INSERT INTO track_record
                   (ts, broker, endpoint, start_equity, end_equity, pnl, pnl_pct,
                    trades_submitted, ticks, breach_triggered, notes, lessons_fired_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d.get("ts", ""),
                d.get("broker", ""),
                d.get("endpoint", ""),
                d.get("start_equity"),
                d.get("end_equity"),
                d.get("pnl"),
                d.get("pnl_pct"),
                d.get("trades_submitted"),
                d.get("ticks"),
                int(bool(d.get("breach_triggered", False))),
                d.get("notes", ""),
                json.dumps(d.get("lessons_fired", [])),
            ),
        )
        count += 1
    conn.commit()
    return count


def _migrate_trade_journal(conn, path: Path) -> int:
    if not path.exists():
        logger.info("No trade_journal.jsonl found — skipping.")
        return 0
    count = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO trade_journal
                   (jid, status, symbol, side, entry_price, stop_price, tp_price,
                    position_size_usd, confidence, features_json, signal_ts, open_ts,
                    close_ts, exit_price, pnl_usd, pnl_pct, r_multiple, outcome)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d.get("jid", ""),
                d.get("status", "open"),
                d.get("symbol", ""),
                d.get("side", ""),
                d.get("entry_price"),
                d.get("stop_price"),
                d.get("tp_price"),
                d.get("position_size_usd"),
                d.get("confidence"),
                json.dumps(d.get("features", {})),
                d.get("signal_ts"),
                d.get("open_ts"),
                d.get("close_ts"),
                d.get("exit_price"),
                d.get("pnl_usd"),
                d.get("pnl_pct"),
                d.get("r_multiple"),
                d.get("outcome"),
            ),
        )
        count += 1
    conn.commit()
    return count


def _migrate_signals(conn, sig_dir: Path) -> int:
    if not sig_dir.exists():
        logger.info("No signals/ directory — skipping.")
        return 0
    count = 0
    for f in sorted(sig_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        conn.execute(
            """INSERT INTO signals
                   (asset, timestamp, direction, entry_price, stop_loss, take_profit,
                    position_size_usd, expected_return_pct, iv_change_pct, confidence,
                    risk_flags_json, context_features_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d.get("asset", ""),
                d.get("timestamp", ""),
                d.get("direction", ""),
                d.get("entry_price"),
                d.get("stop_loss"),
                d.get("take_profit"),
                d.get("position_size_usd"),
                d.get("expected_return_pct"),
                d.get("iv_change_pct"),
                d.get("confidence"),
                json.dumps(d.get("risk_flags", [])),
                json.dumps(d.get("context_features", {})),
            ),
        )
        count += 1
    conn.commit()
    return count


def run_migration(data_dir: Path = Path("data")) -> None:
    from src.data.db import get_db, init_schema

    conn = get_db()
    init_schema(conn)

    pm  = _migrate_postmortems(conn,  data_dir / "postmortems.jsonl")
    tr  = _migrate_track_record(conn, data_dir / "track_record.jsonl")
    tj  = _migrate_trade_journal(conn, data_dir / "trade_journal.jsonl")
    sig = _migrate_signals(conn,      data_dir / "signals")

    print(f"Migration complete:")
    print(f"  postmortems   : {pm} rows")
    print(f"  track_record  : {tr} rows")
    print(f"  trade_journal : {tj} rows")
    print(f"  signals       : {sig} rows")
    print(f"\nDatabase at: {data_dir.parent / 'data' / 'brzrkr.db'}")
    print("Original JSONL files are untouched — delete them manually once verified.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_migration()
