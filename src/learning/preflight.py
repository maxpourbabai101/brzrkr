"""Preflight checks — consult the postmortem DB before starting an
agent session or kicking off a training run.

The checks aren't fortune-telling. They're a structured "have we
addressed each of these known traps?" review that runs every time
the system boots, surfaces high-severity lessons that are relevant
to the configuration, and warns if any unmitigated condition exists.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.learning.postmortem_db import Lesson, PostmortemDB

logger = logging.getLogger(__name__)


@dataclass
class PreflightReport:
    context: str                                  # "agent" | "train"
    relevant_lessons: List[Lesson] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.blockers

    def to_text(self) -> str:
        lines = [f"=== Preflight ({self.context}) ===",
                 f"Lessons consulted: {len(self.relevant_lessons)}",
                 f"Blockers:          {len(self.blockers)}",
                 f"Warnings:          {len(self.warnings)}",
                 ""]
        if self.blockers:
            lines.append("Blockers:")
            lines += [f"  ✗ {b}" for b in self.blockers]
            lines.append("")
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  ⚠ {w}" for w in self.warnings]
            lines.append("")
        if self.relevant_lessons:
            lines.append("Top relevant lessons (review before continuing):")
            for l in self.relevant_lessons[:5]:
                sev = "🔴" * l.severity
                lines.append(f"  {sev} [{l.id}] {l.title}")
                lines.append(f"      mitigation: {l.mitigation}")
            lines.append("")
        return "\n".join(lines)


def run_preflight(
    context: str,
    *,
    db: Optional[PostmortemDB] = None,
    live_money: bool = False,
    model_path: Optional[Path] = None,
    track_record_path: Optional[Path] = None,
) -> PreflightReport:
    """Run all preflight checks for the given context.

    ``context`` is ``"agent"`` (live/paper trading) or ``"train"``
    (training a new model).
    """
    db = db or PostmortemDB()
    db.bootstrap_if_empty()
    report = PreflightReport(context=context)

    # Always surface the top high-severity lessons relevant to the context.
    cat_filter = {
        "agent": ("execution", "risk_sizing", "behavioral", "regime_shift",
                   "liquidity", "counterparty", "deployment", "correlation"),
        "train": ("overfitting", "leakage", "regime_shift", "data_quality"),
    }.get(context, ())
    for l in db.find(min_severity=4):
        if not cat_filter or l.category in cat_filter:
            report.relevant_lessons.append(l)

    # ---- Context-specific checks ----
    if context == "agent":
        _agent_checks(report, live_money=live_money,
                       model_path=model_path,
                       track_record_path=track_record_path)
    elif context == "train":
        _train_checks(report)
    return report


# ---------------------------------------------------------------------------
# Agent-context checks
# ---------------------------------------------------------------------------
def _agent_checks(
    report: PreflightReport,
    *,
    live_money: bool,
    model_path: Optional[Path] = None,
    track_record_path: Optional[Path] = None,
) -> None:
    # l_012: Knight-style runaway. Verify we have a rate-cap (the
    # CountermeasureSet has session turnover; warn if config is missing).
    # l_032: API key permission scoping. We can only check env names.
    if os.getenv("ALPACA_API_KEY") and not os.getenv("ALPACA_API_KEY", "").startswith("PK"):
        report.warnings.append(
            "ALPACA_API_KEY does not start with 'PK' — verify it's a paper key "
            "if you intended paper trading. (Lesson l_032)"
        )

    # l_046: paper-period mandatory before live. Read track_record.
    if live_money:
        path = track_record_path or Path("data/track_record.jsonl")
        if not path.exists():
            report.blockers.append(
                "Live money requested but no track_record.jsonl exists. "
                "Run paper sessions first (l_046)."
            )
        else:
            n = sum(1 for ln in path.read_text().splitlines() if ln.strip())
            if n < 20:
                report.warnings.append(
                    f"Only {n} sessions in track record. Live money requires "
                    "20+ profitable paper sessions (gate-enforced, lesson l_046)."
                )

    # l_038: model not retrained. Warn if xgb model is older than 30 days.
    p = model_path or Path("models/xgb.json")
    if p.exists():
        from datetime import datetime, timezone, timedelta
        age_days = (datetime.now(timezone.utc).timestamp() - p.stat().st_mtime) / 86400.0
        if age_days > 30:
            report.warnings.append(
                f"Trained model is {age_days:.0f} days old. "
                "Consider retraining (l_038)."
            )

    # l_021 / l_052: data freshness — only checkable inside a tick, not preflight.


# ---------------------------------------------------------------------------
# Training-context checks
# ---------------------------------------------------------------------------
def _train_checks(report: PreflightReport) -> None:
    # l_001 + l_050: warn if model exists already (avoid endless retuning).
    if Path("models/xgb_report.json").exists():
        report.warnings.append(
            "Previous training report exists. Avoid tuning hyperparameters "
            "on the same data you'll validate on (l_036, l_050)."
        )

    # l_004 / l_022: survivorship and PIT — we can't auto-check this,
    # but we surface the lesson so the operator confirms data source.
    report.warnings.append(
        "Confirm your training data is point-in-time and survivorship-bias-free "
        "(l_004, l_022). Yahoo / scraped data fails both."
    )
