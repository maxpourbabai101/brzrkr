"""TradeJournal — append-only log of every signal submitted and its outcome.

Every time the agent fires a bracket order the signal dict (including
entry, stop_loss, take_profit, confidence, features) is written here.
When a position closes the same record is updated with the exit price,
P&L, and R-multiple so the online learner can retrain on real outcomes.

File: ``data/trade_journal.jsonl``  — one JSON object per line.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_JOURNAL_PATH = Path("data/trade_journal.jsonl")
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# TradeJournal
# ---------------------------------------------------------------------------

class TradeJournal:
    """Append-only ledger for every trade from signal to close.

    Thread-safe via a module-level lock (one journal per process).
    """

    def __init__(self, path: Path = _JOURNAL_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record_signal(self, signal: Dict[str, Any]) -> str:
        """Called immediately after a bracket order is submitted.

        Returns the journal entry's ``jid`` (journal ID) so callers can
        reference the record later.
        """
        jid = uuid.uuid4().hex[:12]
        entry = {
            "jid":          jid,
            "status":       "open",
            "symbol":       signal.get("asset", ""),
            "side":         signal.get("direction", ""),
            "entry_price":  float(signal.get("entry_price", 0)),
            "stop_price":   float(signal.get("stop_loss", 0)),
            "tp_price":     float(signal.get("take_profit", 0)),
            "position_size_usd": float(signal.get("position_size_usd", 0)),
            "confidence":   float(signal.get("confidence", 0)),
            "features":     signal.get("features", {}),
            "signal_ts":    signal.get("timestamp", _now_iso()),
            "open_ts":      _now_iso(),
            "close_ts":     None,
            "exit_price":   None,
            "pnl_usd":      None,
            "pnl_pct":      None,
            "r_multiple":   None,
            "outcome":      None,   # "win" | "loss" | "scratch"
        }
        self._append(entry)
        logger.info(
            "Journal: opened %s %s @ %.2f  SL=%.2f  TP=%.2f  (jid=%s)",
            entry["side"], entry["symbol"], entry["entry_price"],
            entry["stop_price"], entry["tp_price"], jid,
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
        """Mark the most recent open entry for *symbol* as closed.

        Called by the agent or observer when a position is fully exited.
        Returns True if a record was updated, False if nothing matched.
        """
        records = self._load_all()
        updated = False
        for rec in reversed(records):
            if rec.get("symbol") == symbol and rec.get("status") == "open":
                entry  = float(rec.get("entry_price", 0) or 0)
                stop   = float(rec.get("stop_price",  0) or 0)
                tp     = float(rec.get("tp_price",    0) or 0)
                side   = rec.get("side", "long")

                pnl_pct = _calc_pnl_pct(entry, exit_price, side)
                risk    = abs(entry - stop) if stop else entry * 0.01
                r_mult  = _calc_r(entry, exit_price, stop, side)

                if pnl_usd is None and rec.get("position_size_usd"):
                    pnl_usd = rec["position_size_usd"] * pnl_pct / 100.0

                if pnl_pct > 0.1:
                    outcome = "win"
                elif pnl_pct < -0.1:
                    outcome = "loss"
                else:
                    outcome = "scratch"

                rec.update({
                    "status":     "closed",
                    "close_ts":   exit_time or _now_iso(),
                    "exit_price": exit_price,
                    "pnl_usd":    round(pnl_usd, 2) if pnl_usd is not None else None,
                    "pnl_pct":    round(pnl_pct, 4),
                    "r_multiple": round(r_mult, 3),
                    "outcome":    outcome,
                })
                updated = True
                logger.info(
                    "Journal: closed %s %s exit=%.2f pnl=%.2f%% R=%.2f (%s)",
                    rec["side"], symbol, exit_price,
                    pnl_pct, r_mult, outcome,
                )
                break

        if updated:
            self._rewrite(records)
        return updated

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_open_entry(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the most recent open journal entry for *symbol*, or None."""
        for rec in reversed(self._load_all()):
            if rec.get("symbol") == symbol and rec.get("status") == "open":
                return rec
        return None

    def get_closed_trades(self, *, min_records: int = 0) -> List[Dict[str, Any]]:
        """All closed trades in chronological order."""
        closed = [r for r in self._load_all() if r.get("status") == "closed"]
        if len(closed) < min_records:
            return []
        return closed

    def win_rate(self) -> float:
        """Recent win rate (last 50 trades) — 0.0 if no closed trades."""
        closed = self.get_closed_trades()[-50:]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.get("outcome") == "win")
        return wins / len(closed)

    def avg_r(self) -> float:
        """Average R-multiple over last 50 trades."""
        closed = self.get_closed_trades()[-50:]
        rs = [t["r_multiple"] for t in closed
              if t.get("r_multiple") is not None]
        return sum(rs) / len(rs) if rs else 0.0

    def stats(self) -> Dict[str, Any]:
        closed = self.get_closed_trades()
        open_  = [r for r in self._load_all() if r.get("status") == "open"]
        wins   = [t for t in closed if t.get("outcome") == "win"]
        losses = [t for t in closed if t.get("outcome") == "loss"]
        rs     = [t["r_multiple"] for t in closed
                  if t.get("r_multiple") is not None]
        return {
            "total_closed": len(closed),
            "open":         len(open_),
            "wins":         len(wins),
            "losses":       len(losses),
            "win_rate":     round(len(wins) / max(len(closed), 1), 3),
            "avg_r":        round(sum(rs) / max(len(rs), 1), 3),
            "total_pnl_usd": round(
                sum(t.get("pnl_usd") or 0 for t in closed), 2),
        }

    # ------------------------------------------------------------------
    # Internal IO  (lock-protected)
    # ------------------------------------------------------------------

    def _append(self, record: Dict[str, Any]) -> None:
        with _lock:
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    def _load_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records: List[Dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return records

    def _rewrite(self, records: List[Dict[str, Any]]) -> None:
        with _lock:
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            tmp.replace(self.path)


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
    """Return the R-multiple: realised gain expressed as a multiple of initial risk."""
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
