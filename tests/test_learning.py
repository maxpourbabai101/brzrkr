"""Tests for src.learning.* — PostmortemDB, SessionObserver, preflight."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.learning.postmortem_db import Lesson, PostmortemDB
from src.learning.observer import SessionObservation, SessionObserver
from src.learning.preflight import run_preflight


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def test_bootstrap_populates_seed(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    n = db.bootstrap_if_empty()
    assert n >= 50
    assert len(db.all_lessons) == n


def test_bootstrap_is_idempotent(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    n1 = db.bootstrap_if_empty()
    n2 = db.bootstrap_if_empty()
    assert n1 >= 50
    assert n2 == 0


def test_add_or_update_new_record(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    before = len(db.all_lessons)
    new = Lesson(id="test_l_x", category="test", title="t",
                  description="d", symptom="s", mitigation="m")
    is_new = db.add_or_update(new)
    assert is_new
    assert len(db.all_lessons) == before + 1
    # Reload from disk to confirm persistence.
    db2 = PostmortemDB(path=tmp_path / "pm.jsonl")
    assert db2.get("test_l_x") is not None


def test_confirm_increments_count(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    db.confirm("l_001")
    db.confirm("l_001")
    l = db.get("l_001")
    assert l.confirmed_count == 2
    assert l.last_confirmed != ""


def test_find_filters_correctly(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    overfitting = db.find(category="overfitting")
    assert all(l.category == "overfitting" for l in overfitting)
    assert len(overfitting) >= 3

    high_sev = db.find(min_severity=5)
    assert all(l.severity >= 5 for l in high_sev)
    assert len(high_sev) >= 5


def test_to_markdown_renders_no_crash(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    md = db.to_markdown(top=5)
    assert "Postmortem DB" in md
    assert "##" in md


def test_stats_returns_expected_keys(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    s = db.stats()
    assert "total" in s and s["total"] >= 50
    assert "by_category" in s
    assert "by_severity" in s
    assert "by_source" in s


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
@pytest.fixture
def obs_db(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    return db


def test_observer_confirms_on_breach(obs_db):
    o = SessionObserver(db=obs_db)
    snap = SessionObservation(
        sod_equity=100_000, end_equity=96_000, trades_submitted=4,
        ticks=100, daily_breach=True,
    )
    result = o.observe(snap)
    # Daily breach → confirms l_009 and l_046
    assert "l_009" in result["confirmed"]
    assert obs_db.get("l_009").confirmed_count >= 1


def test_observer_adds_novel_pattern(obs_db):
    o = SessionObserver(db=obs_db)
    snap = SessionObservation(
        sod_equity=100_000, end_equity=97_000, trades_submitted=3,
        ticks=100, daily_breach=False,
    )
    # PnL -3% without breach → adds obs_loss_under_threshold
    result = o.observe(snap)
    assert "obs_loss_under_threshold" in result["added"]
    assert obs_db.get("obs_loss_under_threshold") is not None


def test_observer_zero_trade_pattern(obs_db):
    o = SessionObserver(db=obs_db)
    snap = SessionObservation(
        sod_equity=100_000, end_equity=100_000, trades_submitted=0,
        ticks=60, daily_breach=False,
    )
    result = o.observe(snap)
    assert "obs_zero_trade_session" in result["added"]


def test_observer_confirms_excessive_blocking(obs_db):
    o = SessionObserver(db=obs_db)
    snap = SessionObservation(
        sod_equity=100_000, end_equity=100_000, trades_submitted=2,
        ticks=30, daily_breach=False, countermeasure_blocks=15,
    )
    result = o.observe(snap)
    assert "obs_excessive_blocking" in result["added"]


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def test_preflight_agent_paper_passes(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    report = run_preflight("agent", db=db, live_money=False)
    assert report.passed
    assert len(report.relevant_lessons) > 0


def test_preflight_agent_live_blocks_without_record(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    report = run_preflight("agent", db=db, live_money=True,
                            track_record_path=tmp_path / "tr.jsonl")
    assert not report.passed
    assert any("track_record" in b for b in report.blockers)


def test_preflight_train_warns_about_pit(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    report = run_preflight("train", db=db)
    assert any("point-in-time" in w for w in report.warnings)


def test_preflight_report_text_format(tmp_path):
    db = PostmortemDB(path=tmp_path / "pm.jsonl")
    db.bootstrap_if_empty()
    report = run_preflight("agent", db=db, live_money=False)
    text = report.to_text()
    assert "Preflight" in text
    assert "Lessons consulted" in text
