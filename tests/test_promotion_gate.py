"""Tests for src.execution.promotion_gate.PromotionGate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.execution.promotion_gate import (
    PromotionBlockedError,
    PromotionCriteria,
    PromotionGate,
)


def _gate(tmp_path: Path, **criteria_kw) -> PromotionGate:
    return PromotionGate(
        track_file=tmp_path / "track.jsonl",
        criteria=PromotionCriteria(**criteria_kw),
    )


def _record(gate: PromotionGate, *, pnl_pct: float, endpoint: str = "paper",
            breach: bool = False) -> None:
    gate.record_session(
        broker="alpaca",
        endpoint=endpoint,
        start_equity=100_000.0,
        end_equity=100_000.0 * (1 + pnl_pct),
        trades_submitted=4,
        ticks=80,
        breach_triggered=breach,
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def test_record_session_appends_jsonl(tmp_path):
    g = _gate(tmp_path)
    _record(g, pnl_pct=0.005)
    _record(g, pnl_pct=-0.002)
    history = g.load_history()
    assert len(history) == 2
    assert history[0]["pnl_pct"] == pytest.approx(0.005)
    assert history[1]["pnl_pct"] == pytest.approx(-0.002)


# ---------------------------------------------------------------------------
# Pass / fail criteria
# ---------------------------------------------------------------------------
def test_fewer_than_min_sessions_fails(tmp_path):
    g = _gate(tmp_path, min_sessions=5)
    for _ in range(3):
        _record(g, pnl_pct=0.01)
    v = g.evaluate()
    assert not v["passed"]
    assert "session" in v["reason"]


def test_live_in_history_fails(tmp_path):
    g = _gate(tmp_path, min_sessions=2)
    _record(g, pnl_pct=0.01, endpoint="live")
    _record(g, pnl_pct=0.01, endpoint="paper")
    v = g.evaluate()
    assert not v["passed"]
    assert "live-money" in v["reason"]


def test_negative_total_pnl_fails(tmp_path):
    g = _gate(tmp_path, min_sessions=3, min_sharpe=0.0)
    _record(g, pnl_pct=-0.01)
    _record(g, pnl_pct=-0.01)
    _record(g, pnl_pct=-0.01)
    v = g.evaluate()
    assert not v["passed"]
    assert "P&L" in v["reason"]


def test_breach_fails(tmp_path):
    g = _gate(tmp_path, min_sessions=2, min_sharpe=0.0, max_breach_count=0)
    _record(g, pnl_pct=0.01)
    _record(g, pnl_pct=0.01, breach=True)
    v = g.evaluate()
    assert not v["passed"]
    assert "breaker" in v["reason"].lower()


def test_drawdown_too_big_fails(tmp_path):
    # Permissive min_total_pnl_pct so the drawdown check fires first.
    g = _gate(tmp_path, min_sessions=2, min_sharpe=-99,
              min_total_pnl_pct=-1.0, max_session_dd=0.05)
    _record(g, pnl_pct=0.01)
    _record(g, pnl_pct=-0.10)  # bigger than 5%
    v = g.evaluate()
    assert not v["passed"]
    assert "max allowed loss" in v["reason"]


def test_full_pass(tmp_path):
    # 25 small positive sessions, no breaches, all paper.
    g = _gate(tmp_path, min_sessions=20, min_sharpe=0.0)
    for _ in range(25):
        _record(g, pnl_pct=0.002)
    v = g.evaluate()
    assert v["passed"]


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------
def test_require_eligibility_paper_always_passes(tmp_path):
    g = _gate(tmp_path)
    g.require_eligibility(requested_live_money=False)  # no exception


def test_require_eligibility_live_blocked(tmp_path):
    g = _gate(tmp_path)
    with pytest.raises(PromotionBlockedError):
        g.require_eligibility(requested_live_money=True)


def test_require_eligibility_live_passes_with_record(tmp_path):
    g = _gate(tmp_path, min_sessions=20, min_sharpe=0.0)
    for _ in range(22):
        _record(g, pnl_pct=0.003)
    g.require_eligibility(requested_live_money=True)  # no exception


def test_env_bypass_works(tmp_path, monkeypatch):
    g = _gate(tmp_path)
    monkeypatch.setenv("TRADING_ENHANCER_BYPASS_GATE",
                       "I_ACCEPT_FULL_RESPONSIBILITY")
    g.require_eligibility(requested_live_money=True)  # no exception
