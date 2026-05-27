"""Tests for src.execution.broker.AlpacaExecutor.

The real alpaca-py SDK is patched out — these tests never reach the
network. They verify the signal → bracket‑order translation and the
paper/live safety logic.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake alpaca-py module wired into sys.modules before import.
# ---------------------------------------------------------------------------
def _install_fake_alpaca():
    trading_client_mod = types.ModuleType("alpaca.trading.client")
    requests_mod = types.ModuleType("alpaca.trading.requests")
    enums_mod = types.ModuleType("alpaca.trading.enums")
    trading_mod = types.ModuleType("alpaca.trading")
    root_mod = types.ModuleType("alpaca")

    class _FakeOrder:
        def __init__(self, **kw):
            self.id = "fake-order-123"
            self.kw = kw

    class _FakeClient:
        last_kwargs: dict | None = None

        def __init__(self, key, secret, paper=True):
            self.paper = paper

        def get_account(self):
            a = MagicMock()
            a.equity = "100000.00"
            return a

        def get_all_positions(self):
            return []

        def submit_order(self, order_data):
            _FakeClient.last_kwargs = order_data
            return _FakeOrder()

    trading_client_mod.TradingClient = _FakeClient

    class _Req:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    requests_mod.MarketOrderRequest = _Req
    requests_mod.StopLossRequest = _Req
    requests_mod.TakeProfitRequest = _Req

    enums_mod.OrderClass = types.SimpleNamespace(BRACKET="bracket")
    enums_mod.OrderSide = types.SimpleNamespace(BUY="buy", SELL="sell")
    enums_mod.TimeInForce = types.SimpleNamespace(DAY="day")

    sys.modules["alpaca"] = root_mod
    sys.modules["alpaca.trading"] = trading_mod
    sys.modules["alpaca.trading.client"] = trading_client_mod
    sys.modules["alpaca.trading.requests"] = requests_mod
    sys.modules["alpaca.trading.enums"] = enums_mod
    return _FakeClient


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    return _install_fake_alpaca()


def _signal(**overrides):
    base = {
        "asset": "SPY",
        "timestamp": "2026-05-22T13:35:12Z",
        "direction": "long",
        "entry_price": 500.0,
        "stop_loss": 495.0,
        "take_profit": 510.0,
        "position_size_usd": 5000.0,
        "expected_return_pct": 0.01,
        "iv_change_pct": 0.0,
        "confidence": 0.8,
        "risk_flags": {},
    }
    base.update(overrides)
    return base


def test_submit_signal_translates_to_bracket(fake_client):
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    result = ex.submit_signal(_signal())
    assert result.submitted
    sent = fake_client.last_kwargs
    assert sent.symbol == "SPY"
    assert sent.qty == 10  # 5000 / 500
    assert sent.side == "buy"
    assert sent.order_class == "bracket"
    assert sent.take_profit.limit_price == 510.0
    assert sent.stop_loss.stop_price == 495.0


def test_short_direction_uses_sell(fake_client):
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    ex.submit_signal(_signal(direction="short"))
    assert fake_client.last_kwargs.side == "sell"


def test_subshare_signal_is_skipped(fake_client):
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    result = ex.submit_signal(_signal(position_size_usd=10.0, entry_price=500.0))
    assert not result.submitted
    assert "sub‑share" in result.reason


def test_live_money_requires_env_flag(monkeypatch, fake_client):
    monkeypatch.setenv("ALPACA_LIVE", "false")
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor(live_money=True)
    # Should silently fall back to paper.
    assert ex._paper is True


def test_live_money_when_env_set(monkeypatch, fake_client):
    # The promotion gate would normally block this. Tests skip it via
    # the explicit constructor flag — production callers never get to
    # pass this argument.
    monkeypatch.setenv("ALPACA_LIVE", "true")
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor(live_money=True, skip_promotion_gate=True)
    assert ex._paper is False


def test_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    _install_fake_alpaca()
    from src.execution.broker import AlpacaExecutor
    with pytest.raises(EnvironmentError):
        AlpacaExecutor()


# ---------------------------------------------------------------------------
# Order management additions (used by dashboard.py and trade.py)
# ---------------------------------------------------------------------------
def _patch_fake_client_with_orders(fake_client, *, orders=None,
                                   cancel_count=3, close_ok=True):
    """Extend the fake client with order-management methods."""
    orders = orders or []

    def get_orders(filter=None):
        return orders

    def cancel_orders():
        return [object() for _ in range(cancel_count)]

    def cancel_order_by_id(_id):
        if not close_ok:
            raise RuntimeError("nope")

    def close_position(_symbol):
        if not close_ok:
            raise RuntimeError("no position")

    fake_client.get_orders = get_orders
    fake_client.cancel_orders = cancel_orders
    fake_client.cancel_order_by_id = cancel_order_by_id
    fake_client.close_position = close_position


def _mk_order(**kw):
    base = {
        "id": "abc-123", "symbol": "SPY", "side": "buy", "qty": "10",
        "filled_qty": "0", "order_type": "market", "status": "accepted",
        "limit_price": None, "stop_price": None,
        "submitted_at": "2026-05-22T15:00:00Z", "filled_at": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_get_orders_returns_normalised_dicts(fake_client):
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    _patch_fake_client_with_orders(
        ex._client,
        orders=[_mk_order(symbol="SPY"), _mk_order(symbol="QQQ", side="sell")],
    )
    rows = ex.get_orders()
    assert len(rows) == 2
    assert rows[0]["symbol"] == "SPY"
    assert {"id", "symbol", "side", "qty", "status"}.issubset(rows[0])


def test_get_orders_with_legacy_signature(fake_client):
    """Some older SDK builds don't accept the filter kwarg — agent should
    fall back gracefully."""
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    _patch_fake_client_with_orders(ex._client, orders=[_mk_order()])
    rows = ex.get_orders(status="open")
    assert len(rows) == 1


def test_cancel_order_returns_true_on_success(fake_client):
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    _patch_fake_client_with_orders(ex._client, close_ok=True)
    assert ex.cancel_order("abc-123") is True


def test_cancel_order_returns_false_on_failure(fake_client):
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    _patch_fake_client_with_orders(ex._client, close_ok=False)
    assert ex.cancel_order("abc-123") is False


def test_cancel_all_orders_returns_count(fake_client):
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    _patch_fake_client_with_orders(ex._client, cancel_count=5)
    assert ex.cancel_all_orders() == 5


def test_close_position_success(fake_client):
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    _patch_fake_client_with_orders(ex._client, close_ok=True)
    assert ex.close_position("SPY") is True


def test_close_position_failure_returns_false(fake_client):
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()
    _patch_fake_client_with_orders(ex._client, close_ok=False)
    assert ex.close_position("SPY") is False


def test_get_open_positions_extended_schema(fake_client):
    """Verify the dashboard-facing columns (avg_entry_price, side, pnl_pct)
    are populated."""
    from src.execution.broker import AlpacaExecutor
    ex = AlpacaExecutor()

    def fake_get_all_positions():
        return [SimpleNamespace(
            symbol="SPY", qty="10", avg_entry_price="500.0",
            current_price="510.0", market_value="5100.0",
            unrealized_pl="100.0", unrealized_plpc="0.02",
            side="long",
        )]
    ex._client.get_all_positions = fake_get_all_positions
    positions = ex.get_open_positions()
    assert len(positions) == 1
    p = positions[0]
    assert p["symbol"] == "SPY"
    assert p["avg_entry_price"] == 500.0
    assert p["current_price"] == 510.0
    assert p["unrealized_plpc"] == 0.02
    assert p["side"] == "long"
