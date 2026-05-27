"""Promotion gate — physically blocks live-money trading until a proven
paper-trading record exists.

Every agent / run.py session writes its outcome to
``data/track_record.jsonl``. Before the broker layer is allowed to
instantiate a live-money endpoint, this module reads the record and
verifies that the trailing N sessions meet **all** of the following:

* on paper (no prior live sessions)
* at least ``min_sessions`` sessions
* net positive total P&L
* per-session Sharpe-like score ≥ ``min_sharpe``
* max single-session drawdown ≤ ``max_session_dd``
* no session triggered the daily loss breaker

If any criterion fails, instantiation raises
:class:`PromotionBlockedError`. The user has to either (a) keep paper
trading, or (b) actively edit code to bypass — at which point they own
the decision.

This file is intentionally short and inspectable. Read it before you
trust it.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TRACK_FILE = Path("data/track_record.jsonl")


class PromotionBlockedError(RuntimeError):
    """Raised when live-money instantiation fails the gate."""


@dataclass
class PromotionCriteria:
    min_sessions: int = 20              # at least 20 paper sessions
    min_total_pnl_pct: float = 0.0      # net positive
    min_sharpe: float = 0.5             # daily-Sharpe-like floor
    max_session_dd: float = 0.05        # no single session lost > 5%
    max_breach_count: int = 0           # 0 daily-loss-breach sessions
    require_all_paper: bool = True      # no prior live sessions


@dataclass
class PromotionGate:
    track_file: Path = field(default_factory=lambda: DEFAULT_TRACK_FILE)
    criteria: PromotionCriteria = field(default_factory=PromotionCriteria)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_session(
        self,
        *,
        broker: str,
        endpoint: str,             # "paper" | "live"
        start_equity: float,
        end_equity: float,
        trades_submitted: int,
        ticks: int,
        breach_triggered: bool,
        notes: str = "",
        lessons_fired: Optional[list] = None,
    ) -> None:
        """Append a session record to the track file. Call from agent's
        :meth:`_log_session_summary`.

        ``lessons_fired`` is the list of postmortem lesson IDs the
        observer matched during this session — used downstream by
        :class:`src.learning.correlation_analyzer.CorrelationAnalyzer`
        to compute conditional P&L per lesson.
        """
        self.track_file.parent.mkdir(parents=True, exist_ok=True)
        rec: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "broker": broker,
            "endpoint": endpoint,
            "start_equity": float(start_equity),
            "end_equity": float(end_equity),
            "pnl": float(end_equity - start_equity),
            "pnl_pct": (
                float((end_equity - start_equity) / start_equity)
                if start_equity > 0 else 0.0
            ),
            "trades_submitted": int(trades_submitted),
            "ticks": int(ticks),
            "breach_triggered": bool(breach_triggered),
            "notes": notes,
            "lessons_fired": list(lessons_fired or []),
        }
        with self.track_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        logger.info("Promotion gate: recorded session %s", rec)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def load_history(self) -> List[Dict[str, Any]]:
        if not self.track_file.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.track_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed track-record line: %s", line[:120])
        return out

    # ------------------------------------------------------------------
    # Eligibility check
    # ------------------------------------------------------------------
    def evaluate(self) -> Dict[str, Any]:
        """Run all criteria. Returns a dict of (passed, reason, stats).

        ``passed=True`` means live-money trading is permitted.

        Sessions with zero submitted trades are ignored — those are
        startup/shutdown tests, not real trading data.
        """
        full_history = self.load_history()
        history = [s for s in full_history if s.get("trades_submitted", 0) > 0]
        n = len(history)
        cr = self.criteria

        stats: Dict[str, Any] = {
            "sessions": n,
            "raw_sessions_in_file": len(full_history),
            "ignored_zero_trade_sessions": len(full_history) - n,
            "all_paper": all(s.get("endpoint") == "paper" for s in history),
            "total_pnl_pct": float(sum(s.get("pnl_pct", 0.0) for s in history)),
            "breach_count": sum(1 for s in history if s.get("breach_triggered")),
        }

        # Daily-Sharpe-like estimate: mean(pnl_pct) / std(pnl_pct).
        if n >= 2:
            mean = sum(s["pnl_pct"] for s in history) / n
            var = sum((s["pnl_pct"] - mean) ** 2 for s in history) / (n - 1)
            std = math.sqrt(var) if var > 0 else 0.0
            stats["sharpe_like"] = (mean / std) if std > 0 else float("inf")
        else:
            stats["sharpe_like"] = 0.0

        # Worst single-session drawdown.
        worst = min((s["pnl_pct"] for s in history), default=0.0)
        stats["worst_session_pct"] = worst

        # Enforce in order, short-circuit with first failure.
        if n < cr.min_sessions:
            return self._fail(stats, f"only {n} session(s) recorded; need {cr.min_sessions}+")
        if cr.require_all_paper and not stats["all_paper"]:
            return self._fail(stats,
                              "history contains live-money sessions — gate only "
                              "promotes from a paper-only record")
        if stats["total_pnl_pct"] < cr.min_total_pnl_pct:
            return self._fail(stats,
                              f"total P&L {stats['total_pnl_pct']:+.2%} "
                              f"< required {cr.min_total_pnl_pct:+.2%}")
        if stats["sharpe_like"] < cr.min_sharpe:
            return self._fail(stats,
                              f"per-session Sharpe-like {stats['sharpe_like']:.2f} "
                              f"< required {cr.min_sharpe}")
        if abs(worst) > cr.max_session_dd:
            return self._fail(stats,
                              f"worst session {worst:+.2%} exceeded "
                              f"max allowed loss {-cr.max_session_dd:+.2%}")
        if stats["breach_count"] > cr.max_breach_count:
            return self._fail(stats,
                              f"{stats['breach_count']} session(s) triggered the daily-loss "
                              f"breaker; max allowed is {cr.max_breach_count}")

        return {"passed": True, "reason": "all criteria met", "stats": stats}

    @staticmethod
    def _fail(stats: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {"passed": False, "reason": reason, "stats": stats}

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------
    def require_eligibility(self, requested_live_money: bool) -> None:
        """Raise PromotionBlockedError unless the live-money request is
        backed by an eligible track record. Paper requests always pass.
        """
        if not requested_live_money:
            return
        if os.getenv("TRADING_ENHANCER_BYPASS_GATE") == "I_ACCEPT_FULL_RESPONSIBILITY":
            logger.warning(
                "Promotion gate BYPASSED via env var. You accepted full responsibility."
            )
            return
        verdict = self.evaluate()
        if verdict["passed"]:
            logger.info("Promotion gate PASSED: %s", verdict["stats"])
            return
        msg = (
            "Live-money trading blocked by promotion gate:\n"
            f"  reason : {verdict['reason']}\n"
            f"  stats  : {verdict['stats']}\n"
            "Run more paper sessions, fix the issues, and try again.\n"
            "To bypass (NOT recommended), set:\n"
            "  export TRADING_ENHANCER_BYPASS_GATE=I_ACCEPT_FULL_RESPONSIBILITY"
        )
        logger.error(msg)
        raise PromotionBlockedError(msg)
