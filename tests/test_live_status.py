"""Tests for src.backtest.live_status.LiveStatusWriter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.backtest.live_status import LiveStatusWriter


def test_initial_state_not_active(tmp_path):
    w = LiveStatusWriter(path=tmp_path / "_live.json")
    state = LiveStatusWriter.read(tmp_path / "_live.json")
    # File doesn't exist yet — no flush has happened.
    assert state is None


def test_start_persists_state(tmp_path):
    p = tmp_path / "_live.json"
    w = LiveStatusWriter(path=p, write_throttle_s=0)
    w.start(scenario="covid_crash_2020", symbol="SPY",
             category="crash", bars_total=50, initial_equity=100_000)
    state = LiveStatusWriter.read(p)
    assert state is not None
    assert state["active"] is True
    assert state["scenario"] == "covid_crash_2020"
    assert state["symbol"] == "SPY"
    assert state["bars_total"] == 50
    assert state["current_equity"] == 100_000.0
    assert state["equity_history"] == [100_000.0]


def test_update_appends_to_history(tmp_path):
    p = tmp_path / "_live.json"
    w = LiveStatusWriter(path=p, write_throttle_s=0)
    w.start(scenario="s", symbol="X", category="rally",
             bars_total=100, initial_equity=100_000)
    w.update(bars_processed=10, current_equity=100_500)
    w.update(bars_processed=20, current_equity=101_200)
    state = LiveStatusWriter.read(p)
    assert state["bars_processed"] == 20
    assert state["current_equity"] == 101_200
    assert len(state["equity_history"]) == 3      # initial + 2 updates


def test_finish_sets_inactive(tmp_path):
    p = tmp_path / "_live.json"
    w = LiveStatusWriter(path=p, write_throttle_s=0)
    w.start(scenario="s", symbol="X", category="r",
             bars_total=10, initial_equity=100_000)
    w.finish()
    state = LiveStatusWriter.read(p)
    assert state["active"] is False


def test_overall_progress_independent_of_scenario(tmp_path):
    p = tmp_path / "_live.json"
    w = LiveStatusWriter(path=p, write_throttle_s=0)
    w.set_overall_progress(3, 21)
    state = LiveStatusWriter.read(p)
    assert state["scenarios_done"] == 3
    assert state["scenarios_total"] == 21


def test_history_capped_at_history_max(tmp_path):
    p = tmp_path / "_live.json"
    w = LiveStatusWriter(path=p, write_throttle_s=0, history_max=10)
    w.start(scenario="s", symbol="X", category="r",
             bars_total=100, initial_equity=100_000)
    for i in range(50):
        w.update(bars_processed=i, current_equity=100_000 + i * 10)
    state = LiveStatusWriter.read(p)
    assert len(state["equity_history"]) == 10
    # Most-recent values retained
    assert state["equity_history"][-1] == 100_000 + 49 * 10


def test_open_trade_round_trips(tmp_path):
    p = tmp_path / "_live.json"
    w = LiveStatusWriter(path=p, write_throttle_s=0)
    w.start(scenario="s", symbol="X", category="r",
             bars_total=10, initial_equity=100_000)
    w.update(bars_processed=5, current_equity=100_000,
              open_trade={"direction": "long", "entry": 500.0,
                          "stop": 495.0, "qty": 10})
    state = LiveStatusWriter.read(p)
    assert state["open_trade"]["direction"] == "long"
    assert state["open_trade"]["entry"] == 500.0


def test_read_returns_none_when_missing(tmp_path):
    assert LiveStatusWriter.read(tmp_path / "nope.json") is None


def test_read_returns_none_on_corrupt_file(tmp_path):
    p = tmp_path / "_live.json"
    p.write_text("not json {")
    assert LiveStatusWriter.read(p) is None


def test_throttle_skips_rapid_writes(tmp_path):
    """With a non-zero throttle, rapid updates shouldn't all hit disk
    (only the first plus the forced ones)."""
    p = tmp_path / "_live.json"
    w = LiveStatusWriter(path=p, write_throttle_s=0.5)
    w.start(scenario="s", symbol="X", category="r",
             bars_total=100, initial_equity=100_000)
    # Sequential rapid updates within the throttle window
    for i in range(5):
        w.update(bars_processed=i, current_equity=100_000 + i)
    # Reading should still work, but the disk state may lag.
    state = LiveStatusWriter.read(p)
    assert state is not None
    # Force a write to ensure final state is on disk.
    w.finish()
    state = LiveStatusWriter.read(p)
    assert state["bars_processed"] == 100   # finish() sets it to bars_total
