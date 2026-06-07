"""TradeJournal — SQLite-backed log of every signal submitted and its outcome.

Every time the agent fires a bracket order the signal dict (including
entry, stop_loss, take_profit, confidence, features) is written here.
When a position closes the same record is updated with the exit price,
P&L, and R-multiple so the online learner can retrain on real outcomes.

Storage: ``data/brzrkr.db`` (trade_journal table).
Public API is identical to the old JSONL-backed version.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.db import get_db, init_schema

logger = logging.getLogger(__name__)

# kept for backward-compat; no longer used for IO
_JOURNAL_PATH = Path("data/trade_journal.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradeJournal:
    """Append-only ledger for every trade from signal to close.

    Thread-safe — SQLite WAL mode serialises concurrent writers.
    """

    def __init__(self, path: Path = _JOURNAL_PATH) -> None:
        self.path = Path(path)  # ignored; kept for API compat
        init_schema()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record_signal(self, signal: Dict[str, Any]) -> str:
        """Called immediately after a bracket order is submitted.

        Returns the journal entry's ``jid`` (journal ID) so callers can
        reference the record later.
        """
        jid = uuid.uuid4().hex[:12]
        conn = get_db()
        conn.execute(
            """INSERT INTO trade_journal
                   (jid, status, symbol, side, entry_price, stop_price, tp_price,
                    position_size_usd, confidence, features_json, signal_ts, open_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                jid,
                "open",
                signal.get("asset", ""),
                signal.get("direction", ""),
                float(signal.get("entry_price", 0)),
                float(signal.get("stop_loss", 0)),
                float(signal.get("take_profit", 0)),
                float(signal.get("position_size_usd", 0)),
                float(signal.get("confidence", 0)),
                json.dumps(signal.get("features", {})),
                signal.get("timestamp", _now_iso()),
                _now_iso(),
            ),
        )
        conn.commit()
        logger.info(
            "Journal: opened %s %s @ %.2f  SL=%.2f  TP=%.2f  (jid=%s)",
            signal.get("direction", ""), signal.get("asset", ""),
            float(signal.get("entry_price", 0)),
            float(signal.get("stop_loss", 0)),
            float(signal.get("take_profit", 0)),
            jid,
        )
        return jid

    def close_trade(
        self,
        symbol: str,
        exit_price: float,
        *,
        exit_time: Optional[str] = None,
        pnl_usd: Optional[float] = None,
    ) -> bool:
        """Mark the most recent open entry for *symbol* as closed."""
        conn = get_db()
        row = conn.execute(
            """SELECT * FROM trade_journal
               WHERE symbol=? AND status='open'
               ORDER BY open_ts DESC LIMIT 1""",
            (symbol,),
        ).fetchone()
        if row is None:
            return False

        rec = dict(row)
        entry = float(rec.get("entry_price") or 0)
        stop  = float(rec.get("stop_price")  or 0)
        side  = rec.get("side", "long")

        pnl_pct = _calc_pnl_pct(entry, exit_price, side)
        r_mult  = _calc_r(entry, exit_price, stop, side)

        if pnl_usd is None and rec.get("position_size_usd"):
            pnl_usd = rec["position_size_usd"] * pnl_pct / 100.0

        if pnl_pct > 0.1:
            outcome = "win"
        elif pnl_pct < -0.1:
            outcome = "loss"
        else:
            outcome = "scratch"

        conn.execute(
            """UPDATE trade_journal
               SET status='closed', close_ts=?, exit_price=?,
                   pnl_usd=?, pnl_pct=?, r_multiple=?, outcome=?
               WHERE jid=?""",
            (
                exit_time or _now_iso(),
                exit_price,
                round(pnl_usd, 2) if pnl_usd is not None else None,
                round(pnl_pct, 4),
                round(r_mult, 3),
                outcome,
                rec["jid"],
            ),
        )
        conn.commit()
        logger.info(
            "Journal: closed %s %s exit=%.2f pnl=%.2f%% R=%.2f (%s)",
            side, symbol, exit_price, pnl_pct, r_mult, outcome,
        )
        return True

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_open_entry(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the most recent open journal entry for *symbol*, or None."""
        row = get_db().execute(
            """SELECT * FROM trade_journal
               WHERE symbol=? AND status='open'
               ORDER BY open_ts DESC LIMIT 1""",
            (symbol,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_closed_trades(self, *, min_records: int = 0) -> List[Dict[str, Any]]:
        """All closed trades in chronological order."""
        rows = get_db().execute(
            "SELECT * FROM trade_journal WHERE status='closed' ORDER BY open_ts"
        ).fetchall()
        closed = [self._row_to_dict(r) for r in rows]
        if len(closed) < min_records:
            return []
        return closed

    def win_rate(self) -> float:
        rows = get_db().execute(
            """SELECT outcome FROM trade_journal
               WHERE status='closed'
               ORDER BY open_ts DESC LIMIT 50"""
        ).fetchall()
        if not rows:
            return 0.0
        wins = sum(1 for r in rows if r["outcome"] == "win")
        return wins / len(rows)

    def avg_r(self) -> float:
        rows = get_db().execute(
            """SELECT r_multiple FROM trade_journal
               WHERE status='closed' AND r_multiple IS NOT NULL
               ORDER BY open_ts DESC LIMIT 50"""
        ).fetchall()
        rs = [r["r_multiple"] for r in rows]
        return sum(rs) / len(rs) if rs else 0.0

    def stats(self) -> Dict[str, Any]:
        conn = get_db()
        closed = self.get_closed_trades()
        open_count = conn.execute(
            "SELECT COUNT(*) FROM trade_journal WHERE status='open'"
        ).fetchone()[0]
        wins   = [t for t in closed if t.get("outcome") == "win"]
        losses = [t for t in closed if t.get("outcome") == "loss"]
        rs     = [t["r_multiple"] for t in closed if t.get("r_multiple") is not None]
        return {
            "total_closed":  len(closed),
            "open":          open_count,
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      round(len(wins) / max(len(closed), 1), 3),
            "avg_r":         round(sum(rs) / max(len(rs), 1), 3),
            "total_pnl_usd": round(sum(t.get("pnl_usd") or 0 for t in closed), 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        d = dict(row)
        d["features"] = json.loads(d.pop("features_json") or "{}")
        return d


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _calc_pnl_pct(entry: float, exit_: float, side: str) -> float:
    if entry == 0:
        return 0.0
    if side == "long":
        return (exit_ - entry) / entry * 100.0
    else:
        return (entry - exit_) / entry * 100.0


def _calc_r(entry: float, exit_: float, stop: float, side: str) -> float:
    if side == "long":
        risk = entry - stop
    else:
        risk = stop - entry
    if risk <= 0:
        return 0.0
    if side == "long":
        return (exit_ - entry) / risk
    else:
        return (entry - exit_) / risk
