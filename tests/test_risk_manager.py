"""Tests for src.risk.risk_manager."""

from __future__ import annotations

from datetime import datetime, time, timezone

import pytest

from src.risk.risk_manager import (
    BlackoutWindow,
    apply_blackout_time,
    apply_stop_loss,
    apply_take_profit,
    apply_volatility_filter,
    calculate_position_size,
    check_correlation,
    monitor_drawdown,
    MAX_POSITION_PCT,
)


def test_position_size_caps_at_max_pct():
    notional = calculate_position_size(
        account_equity=100_000,
        confidence=0.99,
        expected_return_pct=0.02,
        max_loss_pct=0.005,
    )
    assert notional <= 100_000 * MAX_POSITION_PCT + 1e-6


def test_position_size_zero_below_50pct_confidence():
    assert calculate_position_size(100_000, confidence=0.40) == 0.0


def test_stop_and_tp_long():
    stop = apply_stop_loss(100.0, "long", atr=1.0, atr_mult=2.0)
    tp = apply_take_profit(100.0, stop, "long", rr=2.0)
    assert stop == pytest.approx(98.0)
    assert tp == pytest.approx(104.0)


def test_stop_and_tp_short():
    stop = apply_stop_loss(100.0, "short", atr=1.0, atr_mult=2.0)
    tp = apply_take_profit(100.0, stop, "short", rr=2.0)
    assert stop == pytest.approx(102.0)
    assert tp == pytest.approx(96.0)


def test_correlation_filter_blocks_highly_correlated():
    matrix = {"SPY": {"QQQ": 0.95}}
    positions = [{"symbol": "QQQ", "qty": 100}]
    assert check_correlation("SPY", positions, matrix) is False


def test_correlation_filter_allows_uncorrelated():
    matrix = {"SPY": {"GLD": 0.1}}
    positions = [{"symbol": "GLD", "qty": 100}]
    assert check_correlation("SPY", positions, matrix) is True


def test_volatility_filter_blocks_high_vix():
    assert apply_volatility_filter(vix=40.0, realized_vol=0.01) is False


def test_volatility_filter_blocks_high_realized():
    assert apply_volatility_filter(vix=15.0, realized_vol=0.06) is False


def test_blackout_blocks_us_open():
    t = datetime(2026, 5, 21, 13, 32, tzinfo=timezone.utc)
    assert apply_blackout_time(t) is False


def test_blackout_allows_outside_windows():
    t = datetime(2026, 5, 21, 15, 0, tzinfo=timezone.utc)
    assert apply_blackout_time(t) is True


def test_custom_blackout_window():
    t = datetime(2026, 5, 21, 16, 0, tzinfo=timezone.utc)
    win = [BlackoutWindow(time(15, 55), time(16, 5), "lunch")]
    assert apply_blackout_time(t, windows=win) is False


def test_drawdown_breach():
    curve = [100, 110, 120, 100, 90]  # 25% drawdown
    assert monitor_drawdown(curve, threshold=0.20) is True


def test_drawdown_within_threshold():
    curve = [100, 105, 110, 108]
    assert monitor_drawdown(curve, threshold=0.10) is False
