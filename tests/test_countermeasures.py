"""Tests for src.risk.countermeasures."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pytest

from src.risk.countermeasures import (
    CountermeasureConfig,
    CountermeasureSet,
    time_stop_breached,
    update_trailing_stop,
)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
def test_circuit_breaker_blocks_after_n_consecutive_losses():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=3,
                                                  cooldown_minutes=0))
    for _ in range(3):
        cm.record_trade_outcome(-100)
    allowed, reason = cm.allow_new_entry(symbol="SPY")
    assert not allowed
    assert "circuit breaker" in reason


def test_winning_trade_resets_breaker():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=3,
                                                  cooldown_minutes=0))
    for _ in range(2):
        cm.record_trade_outcome(-100)
    cm.record_trade_outcome(50)  # winner resets
    for _ in range(2):
        cm.record_trade_outcome(-100)
    allowed, _ = cm.allow_new_entry(symbol="SPY")
    assert allowed


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------
def test_post_loss_cooldown_blocks_immediately():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=99,
                                                  cooldown_minutes=15))
    cm.record_trade_outcome(-50,
                              when=datetime(2026, 5, 22, 15, 0, tzinfo=timezone.utc))
    allowed, reason = cm.allow_new_entry(
        symbol="SPY",
        now=datetime(2026, 5, 22, 15, 5, tzinfo=timezone.utc),
    )
    assert not allowed
    assert "cooldown" in reason


def test_cooldown_clears_after_window():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=99,
                                                  cooldown_minutes=15))
    cm.record_trade_outcome(-50,
                              when=datetime(2026, 5, 22, 15, 0, tzinfo=timezone.utc))
    allowed, _ = cm.allow_new_entry(
        symbol="SPY",
        now=datetime(2026, 5, 22, 15, 30, tzinfo=timezone.utc),
    )
    assert allowed


# ---------------------------------------------------------------------------
# Sector cap
# ---------------------------------------------------------------------------
def test_sector_cap_blocks_when_full():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=99,
                                                  cooldown_minutes=0,
                                                  sector_position_limit=2))
    positions = [
        {"symbol": "AAPL", "sector": "Tech"},
        {"symbol": "MSFT", "sector": "Tech"},
    ]
    allowed, reason = cm.allow_new_entry(
        symbol="GOOG", sector="Tech", existing_positions=positions,
    )
    assert not allowed
    assert "sector cap" in reason


def test_sector_cap_allows_other_sectors():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=99,
                                                  cooldown_minutes=0,
                                                  sector_position_limit=2))
    positions = [
        {"symbol": "AAPL", "sector": "Tech"},
        {"symbol": "MSFT", "sector": "Tech"},
    ]
    allowed, _ = cm.allow_new_entry(
        symbol="JPM", sector="Financials", existing_positions=positions,
    )
    assert allowed


# ---------------------------------------------------------------------------
# Vol regime
# ---------------------------------------------------------------------------
def test_extreme_vix_blocks_new_entry():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=99,
                                                  cooldown_minutes=0,
                                                  vix_extreme_threshold=30.0))
    allowed, reason = cm.allow_new_entry(symbol="SPY", vix=35.0)
    assert not allowed
    assert "VIX" in reason


def test_high_vix_scales_notional_down():
    cm = CountermeasureSet(CountermeasureConfig(vix_high_threshold=22.0,
                                                  high_vix_size_multiplier=0.5))
    sized = cm.adjust_notional(10_000, vix=25.0)
    assert sized == 5_000


def test_extreme_vix_zeros_notional():
    cm = CountermeasureSet(CountermeasureConfig(vix_extreme_threshold=30.0,
                                                  extreme_vix_size_multiplier=0.0))
    sized = cm.adjust_notional(10_000, vix=35.0)
    assert sized == 0


# ---------------------------------------------------------------------------
# Spread + liquidity filters
# ---------------------------------------------------------------------------
def test_wide_spread_blocked():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=99,
                                                  cooldown_minutes=0,
                                                  max_spread_bps=25))
    allowed, reason = cm.allow_new_entry(symbol="X", bid=100.0, ask=101.0)
    assert not allowed
    assert "spread" in reason


def test_low_volume_blocked():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=99,
                                                  cooldown_minutes=0,
                                                  min_avg_volume=250_000))
    allowed, reason = cm.allow_new_entry(symbol="X", avg_volume=50_000)
    assert not allowed
    assert "avg volume" in reason


# ---------------------------------------------------------------------------
# Daily turnover cap
# ---------------------------------------------------------------------------
def test_session_turnover_cap_blocks():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=99,
                                                  cooldown_minutes=0,
                                                  max_trades_per_session=2))
    cm.record_trade_outcome(10)
    cm.record_trade_outcome(10)
    allowed, reason = cm.allow_new_entry(symbol="X")
    assert not allowed
    assert "turnover" in reason


# ---------------------------------------------------------------------------
# Blackout windows
# ---------------------------------------------------------------------------
def test_daily_blackout_window_blocks():
    cm = CountermeasureSet(CountermeasureConfig(
        consecutive_loss_limit=99, cooldown_minutes=0,
        daily_blackout_windows=[(time(13, 30), time(13, 35))],
    ))
    allowed, _ = cm.allow_new_entry(
        symbol="X",
        now=datetime(2026, 5, 22, 13, 32, tzinfo=timezone.utc),
    )
    assert not allowed


def test_event_blackout_blocks():
    start = datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc)
    end = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)
    cm = CountermeasureSet(CountermeasureConfig(
        consecutive_loss_limit=99, cooldown_minutes=0,
        event_blackouts=[(start, end)],
    ))
    allowed, _ = cm.allow_new_entry(
        symbol="X",
        now=datetime(2026, 6, 11, 18, 30, tzinfo=timezone.utc),
    )
    assert not allowed


# ---------------------------------------------------------------------------
# Slippage killer
# ---------------------------------------------------------------------------
def test_repeated_slippage_blocks_entries():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=99,
                                                  cooldown_minutes=0,
                                                  max_slippage_bps=20))
    for _ in range(3):
        cm.record_fill_slippage(40)
    allowed, reason = cm.allow_new_entry(symbol="X")
    assert not allowed
    assert "slipped" in reason


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
def test_reset_session_clears_state():
    cm = CountermeasureSet(CountermeasureConfig(consecutive_loss_limit=2,
                                                  cooldown_minutes=0))
    cm.record_trade_outcome(-10)
    cm.record_trade_outcome(-10)
    cm.reset_session()
    allowed, _ = cm.allow_new_entry(symbol="X")
    assert allowed


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------
def test_trailing_stop_ratchets_up_for_long():
    stop = update_trailing_stop(direction="long", entry_price=100,
                                 current_price=105, current_stop=98,
                                 trail_pct=0.01)
    # 1% trail off $105 = 103.95. Greater than 98, so stop moves up.
    assert stop == pytest.approx(103.95)


def test_trailing_stop_never_moves_down_for_long():
    stop = update_trailing_stop(direction="long", entry_price=100,
                                 current_price=99, current_stop=100,
                                 trail_pct=0.01)
    # 1% trail off $99 = 98.01. Less than current stop 100, so stop holds.
    assert stop == 100


def test_trailing_stop_ratchets_down_for_short():
    stop = update_trailing_stop(direction="short", entry_price=100,
                                 current_price=95, current_stop=102,
                                 trail_pct=0.01)
    # 1% trail off $95 = 95.95. Less than 102, so stop moves down.
    assert stop == pytest.approx(95.95)


# ---------------------------------------------------------------------------
# Time stop
# ---------------------------------------------------------------------------
def test_time_stop_breach():
    opened = datetime(2026, 5, 22, 14, 0, tzinfo=timezone.utc)
    now = datetime(2026, 5, 22, 19, 0, tzinfo=timezone.utc)  # 5h later
    assert time_stop_breached(opened_at=opened, max_holding_minutes=240, now=now)


def test_time_stop_not_breached():
    opened = datetime(2026, 5, 22, 14, 0, tzinfo=timezone.utc)
    now = datetime(2026, 5, 22, 15, 30, tzinfo=timezone.utc)  # 90 min later
    assert not time_stop_breached(opened_at=opened, max_holding_minutes=240, now=now)
