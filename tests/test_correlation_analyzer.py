"""Tests for src.learning.correlation_analyzer."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.learning.correlation_analyzer import (
    CorrelationAnalyzer,
    _phi,
    _welch_one_tailed,
)
from src.learning.postmortem_db import PostmortemDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_sessions(path: Path, sessions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")


def _session(*, pnl_pct: float, lessons: list[str] | None = None) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "broker": "alpaca",
        "endpoint": "paper",
        "start_equity": 100_000.0,
        "end_equity": 100_000.0 * (1 + pnl_pct),
        "pnl": pnl_pct * 100_000.0,
        "pnl_pct": pnl_pct,
        "trades_submitted": 4,
        "ticks": 80,
        "breach_triggered": False,
        "lessons_fired": list(lessons or []),
    }


@pytest.fixture
def analyzer(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    return CorrelationAnalyzer(db=db, track_record_path=tmp_path / "tr.jsonl")


# ---------------------------------------------------------------------------
# Welch's t-test sanity
# ---------------------------------------------------------------------------
def test_welch_left_tail_when_a_clearly_smaller():
    a = [-0.05] * 20
    b = [0.05] * 20
    p = _welch_one_tailed(a, b)
    assert p < 0.001            # very strong evidence that mean(a) < mean(b)


def test_welch_returns_half_when_equal():
    a = [0.01] * 10
    b = [0.01] * 10
    p = _welch_one_tailed(a, b)
    assert p == pytest.approx(0.5, abs=0.01)


def test_phi_endpoints():
    assert _phi(-5) < 1e-6
    assert _phi(5) > 1 - 1e-6
    assert _phi(0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Analyzer behaviour
# ---------------------------------------------------------------------------
def test_analyzer_handles_empty_track_record(analyzer):
    report = analyzer.analyze()
    assert report.n_sessions == 0
    assert report.n_lessons_analyzed == 0


def test_analyzer_confirms_negative_correlation(analyzer, tmp_path):
    # Lesson l_009 always fires on losing sessions; never on winners.
    sessions = []
    for _ in range(15):
        sessions.append(_session(pnl_pct=-0.02, lessons=["l_009"]))
    for _ in range(15):
        sessions.append(_session(pnl_pct=0.01, lessons=[]))
    _write_sessions(tmp_path / "tr.jsonl", sessions)

    report = analyzer.analyze()
    assert report.n_sessions == 30
    neg = report.negative()
    assert len(neg) == 1
    assert neg[0].lesson_id == "l_009"
    assert neg[0].effect_size < 0
    assert neg[0].p_value < 0.10
    assert neg[0].suggested_severity > neg[0].current_severity


def test_analyzer_detects_countermeasure_working(analyzer, tmp_path):
    # Lesson l_012 fires on sessions that outperform (the guardrail saved them).
    sessions = []
    for _ in range(15):
        sessions.append(_session(pnl_pct=0.03, lessons=["l_012"]))
    for _ in range(15):
        sessions.append(_session(pnl_pct=-0.01, lessons=[]))
    _write_sessions(tmp_path / "tr.jsonl", sessions)

    report = analyzer.analyze()
    pos = report.positive()
    assert len(pos) == 1
    assert pos[0].lesson_id == "l_012"
    assert pos[0].effect_size > 0


def test_analyzer_marks_neutral_when_no_difference(analyzer, tmp_path):
    # Both groups make ~the same return.
    sessions = []
    for _ in range(12):
        sessions.append(_session(pnl_pct=0.005, lessons=["l_001"]))
    for _ in range(12):
        sessions.append(_session(pnl_pct=0.005, lessons=[]))
    _write_sessions(tmp_path / "tr.jsonl", sessions)

    report = analyzer.analyze(meaningful_effect=0.002)
    neut = report.neutral()
    assert any(c.lesson_id == "l_001" for c in neut)


def test_analyzer_respects_min_per_group(analyzer, tmp_path):
    # Only 3 sessions fired the lesson — below default min_per_group=5.
    sessions = []
    for _ in range(3):
        sessions.append(_session(pnl_pct=-0.05, lessons=["l_010"]))
    for _ in range(20):
        sessions.append(_session(pnl_pct=0.01, lessons=[]))
    _write_sessions(tmp_path / "tr.jsonl", sessions)

    report = analyzer.analyze(min_per_group=5)
    assert not any(c.lesson_id == "l_010" for c in report.correlations)


# ---------------------------------------------------------------------------
# Apply recommendations
# ---------------------------------------------------------------------------
def test_apply_dry_run_does_not_mutate(analyzer, tmp_path):
    sessions = []
    for _ in range(15):
        sessions.append(_session(pnl_pct=-0.02, lessons=["l_009"]))
    for _ in range(15):
        sessions.append(_session(pnl_pct=0.01, lessons=[]))
    _write_sessions(tmp_path / "tr.jsonl", sessions)

    sev_before = analyzer.db.get("l_009").severity
    report = analyzer.analyze()
    changes = analyzer.apply_recommendations(report, apply=False)
    assert changes  # at least one suggested change
    assert analyzer.db.get("l_009").severity == sev_before


def test_apply_commits_changes(analyzer, tmp_path):
    sessions = []
    for _ in range(15):
        sessions.append(_session(pnl_pct=-0.02, lessons=["l_009"]))
    for _ in range(15):
        sessions.append(_session(pnl_pct=0.01, lessons=[]))
    _write_sessions(tmp_path / "tr.jsonl", sessions)

    sev_before = analyzer.db.get("l_009").severity
    report = analyzer.analyze()
    analyzer.apply_recommendations(report, apply=True)
    sev_after = analyzer.db.get("l_009").severity
    assert sev_after > sev_before
    assert any(t.startswith("analyzer:") for t in analyzer.db.get("l_009").tags)


# ---------------------------------------------------------------------------
# Report text output sanity
# ---------------------------------------------------------------------------
def test_report_text_contains_all_groups(analyzer, tmp_path):
    sessions = []
    for _ in range(12):
        sessions.append(_session(pnl_pct=-0.02, lessons=["l_009"]))
    for _ in range(12):
        sessions.append(_session(pnl_pct=0.03, lessons=["l_012"]))
    for _ in range(12):
        sessions.append(_session(pnl_pct=0.005, lessons=["l_001"]))
    for _ in range(12):
        sessions.append(_session(pnl_pct=0.005, lessons=[]))
    _write_sessions(tmp_path / "tr.jsonl", sessions)

    report = analyzer.analyze(meaningful_effect=0.002)
    text = report.to_text()
    assert "Correlation report" in text
    assert "Sessions analysed" in text
