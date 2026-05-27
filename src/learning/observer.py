"""SessionObserver — turns the agent's own behavior into knowledge.

After each agent session, the observer reads the latest `track_record`
entries plus the in-memory countermeasure stats, looks for known
patterns from the postmortem DB, and either:

* **Confirms** an existing lesson (increments its `confirmed_count`),
  e.g., "the circuit breaker fired" matches lesson `l_024`
  (revenge trading) — the system avoided the trap.

* **Adds** a new lesson if a novel pattern is observed
  (e.g., "11 consecutive losing days in a low-vol regime" doesn't
  match any seed lesson).

The result is a knowledge base that grows automatically with use, so
the next preflight check has more data to act on than the previous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.learning.postmortem_db import Lesson, PostmortemDB

logger = logging.getLogger(__name__)


@dataclass
class SessionObservation:
    """Snapshot of one session that the observer reasons over."""
    sod_equity: float
    end_equity: float
    trades_submitted: int
    ticks: int
    daily_breach: bool
    consecutive_losses_max: int = 0
    circuit_breaker_fired: bool = False
    cooldown_fired: bool = False
    countermeasure_blocks: int = 0
    vix_high_during_session: bool = False
    notes: str = ""

    @property
    def pnl_pct(self) -> float:
        if self.sod_equity <= 0:
            return 0.0
        return (self.end_equity - self.sod_equity) / self.sod_equity


@dataclass
class SessionObserver:
    db: Optional[PostmortemDB] = None

    def __post_init__(self) -> None:
        if self.db is None:
            self.db = PostmortemDB()
            self.db.bootstrap_if_empty()

    # ------------------------------------------------------------------
    # Main entry point — called by the agent at session end
    # ------------------------------------------------------------------
    def observe(self, snap: SessionObservation) -> Dict[str, List[str]]:
        """Inspect a session, mutate the DB, and return what fired.

        Returns ``{"confirmed": [ids], "added": [ids]}``.
        """
        confirmed: List[str] = []
        added: List[str] = []

        # ---- Confirmations of seed lessons ----
        if snap.daily_breach:
            self.db.confirm("l_009")          # full-Kelly type drawdown
            self.db.confirm("l_046")          # paper-period mandatory
            confirmed += ["l_009", "l_046"]

        if snap.circuit_breaker_fired:
            self.db.confirm("l_024")          # revenge trading prevented
            confirmed.append("l_024")

        if snap.cooldown_fired:
            self.db.confirm("l_024")
            confirmed.append("l_024")

        if snap.vix_high_during_session:
            self.db.confirm("l_028")          # vol spike
            confirmed.append("l_028")

        if snap.trades_submitted == 0 and snap.ticks > 5:
            # We thought about trading but didn't — that's the model
            # being honestly uncertain, which is good. Confirms l_036
            # (don't over-tune to make it "look like it's working").
            self.db.confirm("l_036")
            confirmed.append("l_036")

        # ---- Novel pattern detection ----
        # 1. Lost more than max_daily_loss without the breaker firing
        #    (means the breaker threshold may be too loose).
        if snap.pnl_pct < -0.02 and not snap.daily_breach:
            lid = "obs_loss_under_threshold"
            added += self._add_or_confirm(Lesson(
                id=lid,
                category="risk_sizing",
                title="Session lost > 2% without triggering daily breaker",
                description=("A session ended with material losses but the "
                             "agent's daily-loss breaker never fired, meaning "
                             "either the threshold is too loose or losses "
                             "compounded over multiple sessions."),
                symptom=f"Session pnl {snap.pnl_pct:.2%} but breach_triggered=False.",
                mitigation=("Lower max_daily_loss_pct (currently 3%); also "
                            "consider a rolling 5-day loss breaker."),
                severity=3,
                source="observer",
                tags=["sizing", "self_observed"],
            ))

        # 2. Many ticks, zero trades — the model is uniformly under
        #    threshold, which is fine, but if it persists across many
        #    sessions it's a sign of stale features or a too-high gate.
        if snap.ticks > 40 and snap.trades_submitted == 0:
            lid = "obs_zero_trade_session"
            added += self._add_or_confirm(Lesson(
                id=lid,
                category="overfitting",
                title="Session ran full duration with zero trades",
                description=("Model confidence stayed below the threshold "
                             "for the entire session. Either the threshold "
                             "is too high, the features are stale, or the "
                             "regime doesn't match training."),
                symptom=("ticks > 40 and trades_submitted == 0 across "
                         "consecutive sessions."),
                mitigation=("Inspect feature drift; consider lowering the "
                            "confidence threshold for paper sessions; "
                            "retrain if drift confirmed."),
                severity=2,
                source="observer",
                tags=["regime", "self_observed"],
            ))

        # 3. Many countermeasure blocks in one session
        if snap.countermeasure_blocks > 10:
            lid = "obs_excessive_blocking"
            added += self._add_or_confirm(Lesson(
                id=lid,
                category="risk_sizing",
                title="Countermeasures blocked > 10 entries in a single session",
                description=("Many entries blocked usually means either the "
                             "regime is hostile (vol/news event) or one "
                             "filter is too tight."),
                symptom=f"countermeasure_blocks={snap.countermeasure_blocks}",
                mitigation=("Inspect log for which guard fired most; loosen "
                            "if the guard is overcautious for current regime."),
                severity=2,
                source="observer",
                tags=["risk", "self_observed"],
            ))

        if confirmed or added:
            logger.info(
                "SessionObserver: confirmed=%s  added=%s",
                confirmed, added,
            )
        return {"confirmed": confirmed, "added": added}

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _add_or_confirm(self, lesson: Lesson) -> List[str]:
        existing = self.db.get(lesson.id)
        if existing is None:
            self.db.add_or_update(lesson)
            return [lesson.id]
        self.db.confirm(lesson.id)
        return []   # not strictly "added" — pre-existed
