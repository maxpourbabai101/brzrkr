"""Tests for NinjaTraderBroker (Fix 10).

All tests run without a real NinjaTrader instance by mocking the socket.
Run with:
    pytest tests/test_ninja_broker.py -v
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_broker(live_money=False, apex_tier="50k"):
    """Create a NinjaTraderBroker with a mocked ATI connection.

    The PromotionGate is bypassed for tests — we're testing broker
    mechanics, not the promotion gate (which has its own test file).
    """
    with patch("src.execution.ninja_broker._ATIConnection.connect"), \
         patch("src.execution.promotion_gate.PromotionGate.require_eligibility"):
        from src.execution.ninja_broker import NinjaTraderBroker
        broker = NinjaTraderBroker(
            account="Sim101",
            live_money=live_money,
            apex_tier=apex_tier,
        )
        # Replace connection with a mock
        broker._conn = MagicMock()
        return broker


def _signal(
    asset="MES",
    direction="long",
    confidence=0.85,
    entry=5200.0,
    stop=5185.0,
    tp=5230.0,
    notional=5000.0,
):
    return {
        "asset": asset,
        "direction": direction,
        "confidence": confidence,
        "entry_price": entry,
        "stop_loss": stop,
        "take_profit": tp,
        "position_size_usd": notional,
    }


# ── Import smoke test ────────────────────────────────────────────────────────

def test_import():
    from src.execution.ninja_broker import NinjaTraderBroker, APEX_RULES
    assert "50k" in APEX_RULES
    assert NinjaTraderBroker is not None


def test_get_broker_registry():
    """get_broker('ninja') returns a NinjaTraderBroker without connecting."""
    with patch("src.execution.ninja_broker._ATIConnection.connect"):
        from src.execution.brokers import get_broker
        broker = get_broker("ninja")
        from src.execution.ninja_broker import NinjaTraderBroker
        assert isinstance(broker, NinjaTraderBroker)


def test_get_broker_aliases():
    """'ninjatrader' and 'nt8' aliases also work."""
    with patch("src.execution.ninja_broker._ATIConnection.connect"):
        from src.execution.brokers import get_broker
        from src.execution.ninja_broker import NinjaTraderBroker
        for alias in ("ninja", "ninjatrader", "nt8"):
            b = get_broker(alias)
            assert isinstance(b, NinjaTraderBroker), f"alias {alias!r} failed"


# ── ATI protocol ─────────────────────────────────────────────────────────────

class TestATIProtocol:
    def test_submit_signal_sends_three_orders(self):
        """Entry + SL + TP = 3 ATI PLACE commands."""
        broker = _make_broker()
        broker._conn.send_command.return_value = []  # no error lines

        result = broker.submit_signal(_signal())

        assert result.submitted, f"Expected submitted=True, got reason: {result.reason}"
        assert broker._conn.send_command.call_count == 3
        calls = [str(c) for c in broker._conn.send_command.call_args_list]
        assert any("PLACE" in c for c in calls), "No PLACE commands found"
        assert any("STOP" in c for c in calls), "No STOP order found"
        assert any("LIMIT" in c for c in calls), "No LIMIT (TP) order found"

    def test_submit_signal_correct_action_for_long(self):
        broker = _make_broker()
        broker._conn.send_command.return_value = []
        broker.submit_signal(_signal(direction="long"))
        first_call = str(broker._conn.send_command.call_args_list[0])
        assert "BUY" in first_call

    def test_submit_signal_correct_action_for_short(self):
        broker = _make_broker()
        broker._conn.send_command.return_value = []
        broker.submit_signal(_signal(direction="short"))
        first_call = str(broker._conn.send_command.call_args_list[0])
        assert "SELL" in first_call

    def test_submit_signal_returns_false_on_error_response(self):
        broker = _make_broker()
        broker._conn.send_command.return_value = ["ERROR|Order rejected|Invalid account"]
        result = broker.submit_signal(_signal())
        assert not result.submitted

    def test_cancel_order_sends_cancel_command(self):
        broker = _make_broker()
        broker._conn.send_command.return_value = []
        broker.cancel_order("BRZRKR-abc123")
        cmd = str(broker._conn.send_command.call_args)
        assert "CANCEL" in cmd
        assert "BRZRKR-abc123" in cmd

    def test_close_position_sends_closeposition(self):
        broker = _make_broker()
        broker._conn.send_command.return_value = []
        broker.close_position("MES")
        cmd = str(broker._conn.send_command.call_args)
        assert "CLOSEPOSITION" in cmd
        assert "MES" in cmd

    def test_cancel_all_orders(self):
        broker = _make_broker()
        # Two open orders returned by get_orders
        broker._conn.send_command.return_value = [
            "ORD1|MES|buy|1|0|working|MARKET||||",
            "ORD2|NQ|sell|1|0|working|STOP|5000||",
        ]
        with patch.object(broker, "cancel_order", return_value=True) as mock_cancel:
            n = broker.cancel_all_orders()
            assert mock_cancel.call_count == 2


# ── Position/account parsing ─────────────────────────────────────────────────

class TestParsing:
    def test_parse_account_equity(self):
        from src.execution.ninja_broker import _parse_account
        lines = [
            "AccountName|Sim101",
            "CashValue|98500.00",
            "NetLiquidation|99250.50",
            "RealizedPnL|750.00",
        ]
        info = _parse_account(lines)
        # NetLiquidation should be returned as equity
        broker = _make_broker()
        broker._conn.send_command.return_value = lines
        eq = broker.get_account_equity()
        assert eq == pytest.approx(99250.50)

    def test_parse_positions(self):
        from src.execution.ninja_broker import _parse_positions
        lines = [
            "MES|2|5200.0|250.0|52000.0|long",
            "NQ|  -1|18500.0|-500.0|37000.0|short",
        ]
        positions = _parse_positions(lines)
        assert len(positions) == 2
        assert positions[0]["symbol"] == "MES"
        assert positions[0]["qty"] == 2.0
        assert positions[0]["side"] == "long"
        assert positions[1]["side"] == "short"

    def test_parse_orders(self):
        from src.execution.ninja_broker import _parse_orders
        lines = [
            "ORD001|ES|buy|2|2|filled|MARKET|||2026-06-09T14:30:00",
            "ORD002|MES|sell|1|0|working|STOP||5180.0|",
        ]
        orders = _parse_orders(lines)
        assert len(orders) == 2
        assert orders[0]["status"] == "filled"
        assert orders[1]["status"] == "working"


# ── Apex risk rules ───────────────────────────────────────────────────────────

class TestApexRiskRules:
    def test_halted_session_blocks_all_orders(self):
        broker = _make_broker(live_money=True)
        broker._halted = True
        broker._halt_reason = "daily loss exceeded"
        result = broker.submit_signal(_signal())
        assert not result.submitted
        assert "halted" in result.reason.lower()

    def test_daily_loss_limit_triggers_halt(self):
        broker = _make_broker(live_money=True, apex_tier="50k")
        # Rules: max_daily_loss=2500, halt_buffer_daily=200 → halt at -2300
        broker._daily_pnl = -2350.0
        broker._peak_equity = 100_000.0
        broker._conn.send_command.return_value = ["NetLiquidation|97650.00"]

        with patch.object(broker, "cancel_all_orders"), \
             patch.object(broker, "close_all_positions"):
            result = broker.submit_signal(_signal())

        assert not result.submitted
        assert broker._halted

    def test_trailing_drawdown_triggers_halt(self):
        broker = _make_broker(live_money=True, apex_tier="50k")
        # Peak was $100k, now equity $97,800 → drawdown $2,200 ≥ 2500-300=2200
        broker._peak_equity = 100_000.0
        broker._conn.send_command.return_value = ["NetLiquidation|97800.00"]

        with patch.object(broker, "cancel_all_orders"), \
             patch.object(broker, "close_all_positions"):
            result = broker.submit_signal(_signal())

        assert not result.submitted
        assert broker._halted

    def test_consistency_rule_blocks_over_30pct(self):
        broker = _make_broker(live_money=True, apex_tier="50k")
        # Session P&L = $3000, today already made $1000 > 30% ($900) → block
        broker._session_pnl = 3000.0
        broker._daily_pnl   = 1000.0
        broker._peak_equity = 103_000.0
        broker._conn.send_command.return_value = ["NetLiquidation|103000.00"]

        result = broker.submit_signal(_signal())
        assert not result.submitted
        assert "consistency" in result.reason.lower()

    def test_sim_mode_skips_apex_rules(self):
        """live_money=False should bypass all Apex checks."""
        broker = _make_broker(live_money=False)
        broker._daily_pnl   = -9999.0   # would halt in live mode
        broker._halted = False
        broker._conn.send_command.return_value = []
        result = broker.submit_signal(_signal())
        assert result.submitted

    def test_reset_daily_clears_halt(self):
        broker = _make_broker(live_money=True)
        broker._halted = True
        broker._halt_reason = "daily loss"
        broker._daily_pnl = -2500.0
        broker.reset_daily()
        assert not broker._halted
        assert broker._daily_pnl == 0.0

    def test_fomc_blackout_blocks_live_order(self):
        broker = _make_broker(live_money=True)
        broker._peak_equity = 100_000.0
        broker._conn.send_command.return_value = ["NetLiquidation|100000.00"]
        # FOMC June 18 2026 at 18:00 UTC — 30 min before
        fomc_time = datetime(2026, 6, 18, 17, 30, tzinfo=timezone.utc)
        with patch("src.filters.event_blackout._near_macro_event",
                   return_value=(True, "FOMC event within 30 min")):
            result = broker.submit_signal(_signal())
        assert not result.submitted
        assert "blackout" in result.reason.lower() or "FOMC" in result.reason

    def test_no_fomc_block_in_sim_mode(self):
        """Apex news blackout does not apply in sim/paper mode."""
        broker = _make_broker(live_money=False)
        broker._conn.send_command.return_value = []
        with patch("src.filters.event_blackout._near_macro_event",
                   return_value=(True, "FOMC event within 30 min")):
            result = broker.submit_signal(_signal())
        assert result.submitted


# ── apex_status snapshot ─────────────────────────────────────────────────────

class TestApexStatus:
    def test_status_keys_present(self):
        broker = _make_broker()
        broker._conn.send_command.return_value = ["NetLiquidation|100000.00"]
        status = broker.apex_status()
        required = {
            "account", "apex_tier", "equity", "peak_equity",
            "daily_pnl", "session_pnl", "trailing_dd_used",
            "trailing_dd_limit", "daily_loss_limit", "halted",
        }
        assert required.issubset(status.keys())

    def test_record_fill_updates_session_pnl(self):
        broker = _make_broker()
        broker._conn.send_command.return_value = ["NetLiquidation|100500.00"]
        broker.record_fill(500.0)
        assert broker._session_pnl == 500.0
        assert broker._daily_pnl   == 500.0

    def test_record_fill_updates_peak_equity(self):
        broker = _make_broker()
        broker._peak_equity = 100_000.0
        broker._conn.send_command.return_value = ["NetLiquidation|101000.00"]
        broker.record_fill(1000.0)
        assert broker._peak_equity == pytest.approx(101_000.0)
