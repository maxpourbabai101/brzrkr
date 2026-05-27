"""CorrelationAnalyzer — empirical conditional P&L per lesson.

For every lesson the observer has matched against this system at
least N times, compute:

    μ_fired   = mean session P&L (%) when the lesson fired
    μ_unfired = mean session P&L (%) when it did NOT fire
    Δ         = μ_fired - μ_unfired
    p-value   = Welch's t-test (one-tailed, alt: μ_fired < μ_unfired)

Interpret:

* **confirmed_negative** — Δ < 0 AND p < α (default 0.10). The
  lesson genuinely correlates with losses in YOUR data. Promote
  severity.
* **countermeasure_working** — Δ > 0 AND p < α. Sessions where
  this lesson fired actually outperformed; the countermeasure or
  signal is doing real work. Optionally consider relaxing
  preconditions to let it fire more.
* **neutral** — |Δ| small or p ≥ α. The lesson doesn't appear
  to differentiate outcomes in your data. Consider whether the
  associated countermeasure is too aggressive.
* **insufficient_data** — fewer than `min_per_group` observations
  on either side.

`apply_recommendations()` mutates lesson severities based on the
report. Off by default (dry-run); pass `apply=True` to commit.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.learning.postmortem_db import PostmortemDB

logger = logging.getLogger(__name__)


@dataclass
class LessonCorrelation:
    lesson_id: str
    title: str
    fired_n: int
    unfired_n: int
    mean_pnl_fired: float
    mean_pnl_unfired: float
    effect_size: float          # mean_pnl_fired - mean_pnl_unfired
    p_value: float              # one-tailed, alt: fired < unfired
    interpretation: str
    recommendation: str
    current_severity: int
    suggested_severity: int


@dataclass
class CorrelationReport:
    n_sessions: int
    n_lessons_analyzed: int
    correlations: List[LessonCorrelation] = field(default_factory=list)
    alpha: float = 0.10

    def negative(self) -> List[LessonCorrelation]:
        return [c for c in self.correlations if c.interpretation == "confirmed_negative"]

    def positive(self) -> List[LessonCorrelation]:
        return [c for c in self.correlations if c.interpretation == "countermeasure_working"]

    def neutral(self) -> List[LessonCorrelation]:
        return [c for c in self.correlations if c.interpretation == "neutral"]

    def to_text(self) -> str:
        lines = [f"=== Correlation report ===",
                 f"Sessions analysed     : {self.n_sessions}",
                 f"Lessons with data     : {self.n_lessons_analyzed}",
                 f"Alpha (p-value cutoff): {self.alpha}",
                 ""]
        for group_name, items in (("CONFIRMED NEGATIVE (promote)", self.negative()),
                                   ("COUNTERMEASURE WORKING", self.positive()),
                                   ("NEUTRAL (consider loosening)", self.neutral())):
            if not items:
                continue
            lines.append(f"--- {group_name} ---")
            for c in items:
                lines.append(
                    f"  [{c.lesson_id}] {c.title[:54]}"
                )
                lines.append(
                    f"      fired n={c.fired_n:>3}  unfired n={c.unfired_n:>3}  "
                    f"Δ={c.effect_size:+.4f}  p={c.p_value:.3f}"
                )
                lines.append(
                    f"      severity {c.current_severity}→{c.suggested_severity}  "
                    f"reco: {c.recommendation}"
                )
            lines.append("")
        return "\n".join(lines)


class CorrelationAnalyzer:
    def __init__(
        self,
        db: Optional[PostmortemDB] = None,
        track_record_path: Path = Path("data/track_record.jsonl"),
    ) -> None:
        self.db = db or PostmortemDB()
        self.db.bootstrap_if_empty()
        self.track_record_path = track_record_path

    # ------------------------------------------------------------------
    def _load_sessions(self) -> List[dict]:
        if not self.track_record_path.exists():
            return []
        out = []
        for ln in self.track_record_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------
    def analyze(
        self,
        *,
        min_per_group: int = 5,
        alpha: float = 0.10,
        meaningful_effect: float = 0.001,    # 0.1% P&L delta to even consider
    ) -> CorrelationReport:
        sessions = self._load_sessions()
        report = CorrelationReport(
            n_sessions=len(sessions),
            n_lessons_analyzed=0,
            alpha=alpha,
        )

        # Find all unique lesson IDs that have ever fired in the record.
        all_lesson_ids: set[str] = set()
        for s in sessions:
            all_lesson_ids.update(s.get("lessons_fired", []) or [])

        for lid in sorted(all_lesson_ids):
            lesson = self.db.get(lid)
            if lesson is None:
                continue
            fired = [s["pnl_pct"] for s in sessions
                     if lid in (s.get("lessons_fired") or [])]
            unfired = [s["pnl_pct"] for s in sessions
                       if lid not in (s.get("lessons_fired") or [])]

            if len(fired) < min_per_group or len(unfired) < min_per_group:
                continue

            mu_fired = statistics.mean(fired)
            mu_unfired = statistics.mean(unfired)
            effect = mu_fired - mu_unfired
            p_value = _welch_one_tailed(fired, unfired)

            # Categorize.
            if abs(effect) < meaningful_effect:
                interp = "neutral"
                reco = ("No measurable difference in outcomes. Consider "
                        "loosening the corresponding countermeasure if it "
                        "blocks trades aggressively.")
                suggested = max(1, lesson.severity - 1) if lesson.severity > 2 else lesson.severity
            elif effect < 0 and p_value < alpha:
                interp = "confirmed_negative"
                reco = ("Lesson genuinely correlates with worse outcomes in "
                        "your data — promote severity and harden the "
                        "associated guardrail.")
                suggested = min(5, lesson.severity + 1)
            elif effect > 0 and (1.0 - p_value) < alpha:
                # Symmetric: one-tailed p in the OPPOSITE direction.
                interp = "countermeasure_working"
                reco = ("Sessions where this fired OUTPERFORMED, suggesting "
                        "the countermeasure caught real risk. Keep as-is.")
                suggested = lesson.severity
            else:
                interp = "neutral"
                reco = "Not statistically significant — keep watching."
                suggested = lesson.severity

            report.correlations.append(LessonCorrelation(
                lesson_id=lid,
                title=lesson.title,
                fired_n=len(fired),
                unfired_n=len(unfired),
                mean_pnl_fired=mu_fired,
                mean_pnl_unfired=mu_unfired,
                effect_size=effect,
                p_value=p_value,
                interpretation=interp,
                recommendation=reco,
                current_severity=lesson.severity,
                suggested_severity=suggested,
            ))

        report.n_lessons_analyzed = len(report.correlations)
        return report

    # ------------------------------------------------------------------
    def apply_recommendations(
        self,
        report: CorrelationReport,
        *,
        apply: bool = False,
    ) -> List[str]:
        """Promote / demote lesson severities based on report.

        Returns a list of human-readable change descriptions.
        When ``apply=False`` (default), only previews; no DB mutation.
        """
        changes: List[str] = []
        for c in report.correlations:
            if c.suggested_severity == c.current_severity:
                continue
            verb = "PROMOTE" if c.suggested_severity > c.current_severity else "DEMOTE"
            changes.append(
                f"{verb} [{c.lesson_id}] severity "
                f"{c.current_severity}→{c.suggested_severity}  "
                f"({c.interpretation})  «{c.title[:60]}»"
            )
            if apply:
                lesson = self.db.get(c.lesson_id)
                if lesson is not None:
                    lesson.severity = c.suggested_severity
                    # Stamp an analyzer reference so we can audit.
                    note = f"analyzer:{c.interpretation}"
                    if note not in lesson.tags:
                        lesson.tags.append(note)
                    self.db.add_or_update(lesson)
        return changes


# ---------------------------------------------------------------------------
# Welch's t-test helper (one-tailed) — no scipy required for the simple case
# ---------------------------------------------------------------------------
def _welch_one_tailed(a: List[float], b: List[float]) -> float:
    """One-tailed Welch t-test, alternative: mean(a) < mean(b).

    Returns a p-value in [0, 1]. We use a normal approximation for the
    tail (Welch-Satterthwaite df ≥ ~30 makes this fine; for small
    samples it's slightly conservative which is what we want).
    """
    if len(a) < 2 or len(b) < 2:
        return 1.0
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    if var_a == 0 and var_b == 0:
        # No within-group variation. If the means are equal, p=0.5
        # (no evidence either way). If they differ, that's a perfect
        # separation — treat as very strong evidence in the direction
        # of difference.
        if mean_a == mean_b:
            return 0.5
        return 0.0 if mean_a < mean_b else 1.0
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    if se == 0:
        return 0.5
    t = (mean_a - mean_b) / se
    # Normal CDF for one-tailed (left): P(Z ≤ t)
    return _phi(t)


def _phi(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
