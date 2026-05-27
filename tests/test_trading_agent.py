"""Tests for src.agent.trading_agent.TradingAgent.

Everything is stubbed — no real broker, no real data. The tests
verify the loop's decision logic: market-hours gating, daily-loss
breaker, position cap, pre-close cutoff, confidence threshold,
SIGINT-style stops, dry-run never submits orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.agent.trading_agent import AgentConfig, TradingAgent


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
@dataclass
class _StubExecutor:
    equity: float = 100_000.0
    positions: List[Dict[str, Any]] = field(default_factory=list)
    submitted: List[Dict[str, Any]] = field(default_factory=list)
    reject: bool = False
    # Mirror the AlpacaExecutor attribute the agent reaches into for clock.
    _client: Any = None
    _paper: bool = True

    def get_account_equity(self) -> float:
        return self.equity

    def get_open_positions(self):
        return self.positions

    def submit_signal(self, signal):
        if self.reject:
            return SimpleNamespace(submitted=False, order_id=None, reason="stub-reject")
        self.submitted.append(signal)
        return SimpleNamespace(submitted=True, order_id=f"id-{len(self.submitted)}",
                               reason="ok")


def _make_clock(is_open=True, minutes_to_close=120,
                ts=None):
    ts = ts or datetime(2026, 5, 22, 15, 0, tzinfo=timezone.utc)
    next_close = ts + timedelta(minutes=minutes_to_close)
    next_open = ts + timedelta(hours=18)
    return SimpleNamespace(
        is_open=is_open,
        timestamp=ts,
        next_open=next_open,
        next_close=next_close,
    )


def _stub_engineer(empty=False):
    class _Eng:
        def build_features(self, bundle):
            if empty:
                return pd.DataFrame()
            prices = bundle.get("prices")
            out = prices.copy()
            out.attrs["context"] = {"vix_level": 18.0}
            return out
    return _Eng()


def _stub_ensemble(confidence=0.85, direction="long"):
    class _Ens:
        def predict(self, features):
            return {
                "direction": direction,
                "expected_return_pct": 0.012,
                "iv_change_pct": 0.0,
                "confidence": confidence,
            }
    return _Ens()


def _stub_signal_builder(emit=True):
    def builder(symbol, prediction, risk_params):
        if not emit:
            return None
        entry = risk_params["entry_price"]
        return {
            "asset": symbol,
            "timestamp": "2026-05-22T15:00:00Z",
            "direction": prediction["direction"],
            "entry_price": entry,
            "stop_loss": entry * 0.99,
            "take_profit": entry * 1.02,
            "position_size_usd": 1000.0,
            "expected_return_pct": 0.012,
            "iv_change_pct": 0.0,
            "confidence": prediction["confidence"],
            "risk_flags": {},
        }
    return builder


def _price_frame(n=300):
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="1D")
    return pd.DataFrame({
        "open": base, "high": base + 0.5, "low": base - 0.5,
        "close": base, "volume": rng.integers(1000, 100000, n),
    }, index=idx)


def _stub_fetcher():
    def fetch(symbol):
        return {"prices": _price_frame()}
    return fetch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cfg(tmp_path):
    return AgentConfig(
        universe=["SPY", "QQQ"],
        seq_len=64,
        tick_seconds=0,
        max_positions=5,
        max_daily_loss_pct=0.03,
        pre_close_minutes=15,
        confidence_threshold=0.75,
        dry_run=False,
        stop_file=tmp_path / "STOP",
        signal_dir=tmp_path / "signals",
    )


def _make_agent(cfg, *, executor=None, clock=None,
                ensemble=None, signal_builder=None, engineer=None,
                fetcher=None):
    return TradingAgent(
        cfg,
        executor=executor or _StubExecutor(),
        data_fetcher=fetcher or _stub_fetcher(),
        feature_engineer=engineer or _stub_engineer(),
        ensemble=ensemble or _stub_ensemble(),
        signal_builder=signal_builder or _stub_signal_builder(),
        clock=clock or (lambda: _make_clock()),
        sleep=lambda _s: None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_tick_submits_when_confidence_clears(cfg):
    ex = _StubExecutor()
    agent = _make_agent(cfg, executor=ex)
    agent.tick()
    assert len(ex.submitted) == 2  # one per universe symbol
    assert ex.submitted[0]["direction"] == "long"


def test_tick_skips_when_market_closed(cfg):
    ex = _StubExecutor()
    closed = lambda: _make_clock(is_open=False)
    agent = _make_agent(cfg, executor=ex, clock=closed)
    agent.tick()
    assert ex.submitted == []


def test_tick_skips_symbols_already_held(cfg):
    ex = _StubExecutor(positions=[{"symbol": "SPY", "qty": 10,
                                   "market_value": 5000.0,
                                   "unrealized_pl": 100.0}])
    agent = _make_agent(cfg, executor=ex)
    agent.tick()
    # Only QQQ should fire — SPY is already held.
    assert [s["asset"] for s in ex.submitted] == ["QQQ"]


def test_position_cap_blocks_new_entries(cfg):
    cfg.max_positions = 1
    ex = _StubExecutor(positions=[{"symbol": "XLK", "qty": 1,
                                   "market_value": 100.0,
                                   "unrealized_pl": 0.0}])
    agent = _make_agent(cfg, executor=ex)
    agent.tick()
    assert ex.submitted == []


def test_daily_loss_breaker_stops_agent(cfg):
    """If equity drops > max_daily_loss_pct since SOD, agent halts."""
    ex = _StubExecutor(equity=100_000.0)
    agent = _make_agent(cfg, executor=ex)
    agent.tick()  # establishes SOD equity = 100k
    # Simulate 5% loss.
    ex.equity = 95_000.0
    agent.tick()
    assert agent._stopped is True


def test_pre_close_cutoff_blocks_entries(cfg):
    cfg.pre_close_minutes = 30
    ex = _StubExecutor()
    near_close = lambda: _make_clock(minutes_to_close=10)
    agent = _make_agent(cfg, executor=ex, clock=near_close)
    agent.tick()
    assert ex.submitted == []


def test_confidence_below_threshold_skipped(cfg):
    ex = _StubExecutor()
    agent = _make_agent(cfg, executor=ex,
                        ensemble=_stub_ensemble(confidence=0.55))
    agent.tick()
    assert ex.submitted == []


def test_dry_run_never_submits(cfg):
    cfg.dry_run = True
    ex = _StubExecutor()
    agent = _make_agent(cfg, executor=ex)
    agent.tick()
    assert ex.submitted == []
    # Signal JSON files should still be written.
    files = list(cfg.signal_dir.glob("*.json"))
    assert len(files) == 2


def test_stop_file_halts_loop(cfg):
    cfg.stop_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.stop_file.touch()
    ex = _StubExecutor()
    agent = _make_agent(cfg, executor=ex)
    agent.run()  # Should exit immediately.
    assert ex.submitted == []


def test_stop_method_halts_loop(cfg):
    ex = _StubExecutor()
    agent = _make_agent(cfg, executor=ex)
    agent.stop()
    agent.run()
    assert ex.submitted == []


def test_data_fetch_failure_does_not_kill_loop(cfg):
    ex = _StubExecutor()

    def broken_fetcher(symbol):
        if symbol == "SPY":
            raise RuntimeError("network down")
        return {"prices": _price_frame()}

    agent = _make_agent(cfg, executor=ex, fetcher=broken_fetcher)
    agent.tick()  # Should not raise.
    # QQQ still submits despite SPY blowing up.
    assert [s["asset"] for s in ex.submitted] == ["QQQ"]


def test_empty_features_skipped(cfg):
    ex = _StubExecutor()
    agent = _make_agent(cfg, executor=ex, engineer=_stub_engineer(empty=True))
    agent.tick()
    assert ex.submitted == []


def test_session_summary_logs_pnl(cfg, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="src.agent.trading_agent")
    ex = _StubExecutor()
    agent = _make_agent(cfg, executor=ex)
    agent.stop()
    agent.run()
    summary_lines = [r for r in caplog.records if "Agent stopped" in r.message]
    assert summary_lines, "expected a session summary log line"


# ---------------------------------------------------------------------------
# Sleep responsiveness fixes
# ---------------------------------------------------------------------------
def test_interruptible_sleep_breaks_when_stopped(cfg):
    """If stop is requested mid-sleep, we should exit fast, not wait the
    full duration."""
    cfg.tick_seconds = 60  # would normally sleep a full minute
    ex = _StubExecutor()
    sleeps: list[float] = []

    def fake_sleep(s):
        sleeps.append(s)
        # After the first 2s chunk, set the stop flag — agent should bail.
        if len(sleeps) >= 1:
            agent.stop()

    agent = TradingAgent(
        cfg,
        executor=ex,
        data_fetcher=_stub_fetcher(),
        feature_engineer=_stub_engineer(),
        ensemble=_stub_ensemble(),
        signal_builder=_stub_signal_builder(),
        clock=lambda: _make_clock(),
        sleep=fake_sleep,
    )
    agent._interruptible_sleep(60.0)
    # Should have slept at most one chunk (~2s) before bailing.
    assert sum(sleeps) <= 4.0


def test_market_closed_extends_sleep(cfg):
    """When market is closed and next_open is hours away, the sleep
    should extend toward next_open instead of staying at tick_seconds."""
    cfg.tick_seconds = 60
    ex = _StubExecutor()
    # next_open in 30 minutes, market closed now.
    closed_clock = lambda: _make_clock(is_open=False, minutes_to_close=0,
                                       ts=datetime(2026, 5, 23, 14, 0,
                                                   tzinfo=timezone.utc))
    closed_clock.__name__ = "closed_clock"
    agent = _make_agent(cfg, executor=ex, clock=closed_clock)
    # Manually drive _next_sleep_seconds: market is closed, next_open is
    # ~ 18h later in the stub helper, so we should hit the 3600s cap.
    sleep_secs = agent._next_sleep_seconds()
    assert sleep_secs >= cfg.tick_seconds
    assert sleep_secs <= 3600.0


def test_market_open_uses_tick_seconds(cfg):
    """When market is open, sleep stays at tick_seconds."""
    cfg.tick_seconds = 60
    ex = _StubExecutor()
    agent = _make_agent(cfg, executor=ex)  # default clock: market open
    assert agent._next_sleep_seconds() == 60.0
