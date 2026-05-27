"""Tests for src.execution.brokers (multi-broker registry)."""

from __future__ import annotations

import pytest

from src.execution.brokers import InMemoryBroker, get_broker


# ---------------------------------------------------------------------------
# InMemoryBroker simulator
# ---------------------------------------------------------------------------
def _signal(**kw):
    base = {
        "asset": "SPY",
        "timestamp": "2026-05-22T15:00:00Z",
        "direction": "long",
        "entry_price": 500.0,
        "stop_loss": 495.0,
        "take_profit": 510.0,
        "position_size_usd": 5000.0,
        "expected_return_pct": 0.02,
        "iv_change_pct": 0.0,
        "confidence": 0.9,
        "risk_flags": {},
    }
    base.update(kw)
    return base


def test_in_memory_buy_fills_immediately():
    b = InMemoryBroker(starting_equity=100_000)
    r = b.submit_signal(_signal())
    assert r.submitted
    positions = b.get_open_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "SPY"
    assert positions[0]["qty"] == 10


def test_in_memory_sub_share_rejected():
    b = InMemoryBroker()
    r = b.submit_signal(_signal(position_size_usd=10, entry_price=500))
    assert not r.submitted
    assert "sub-share" in r.reason


def test_in_memory_close_position():
    b = InMemoryBroker()
    b.submit_signal(_signal())
    assert b.get_open_positions()
    ok = b.close_position("SPY")
    assert ok
    assert b.get_open_positions() == []


def test_in_memory_cancel_all_orders():
    b = InMemoryBroker()
    b.submit_signal(_signal())
    # Filled orders aren't cancellable.
    assert b.cancel_all_orders() == 0


def test_in_memory_account_equity_reflects_market_value():
    b = InMemoryBroker(starting_equity=100_000)
    b.submit_signal(_signal())
    # We spent 5000 on 10 shares; market value is also 5000, so equity unchanged.
    assert b.get_account_equity() == pytest.approx(100_000)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_get_broker_paper_only():
    b = get_broker("paper_only")
    assert isinstance(b, InMemoryBroker)


def test_get_broker_unknown_raises():
    with pytest.raises(ValueError):
        get_broker("etrade")


def test_get_broker_ibkr_raises_without_ib_insync(monkeypatch):
    # Force the ib_insync import to fail.
    import sys
    monkeypatch.setitem(sys.modules, "ib_insync", None)
    with pytest.raises(ImportError):
        get_broker("ibkr")
