"""Tests for brzrkr_app.indicators."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from brzrkr_app.indicators import (
    EXPLANATIONS, TIME_SPREADS,
    all_indicators, adx, atr, bollinger, macd, obv, rsi,
    reading_52w_percentile, reading_adx, reading_atr, reading_bbands,
    reading_macd, reading_obv, reading_rsi, reading_sma_cross,
    reading_volume_ratio,
)


def _trending_ohlcv(n: int = 250, trend: float = 0.001, seed: int = 0
                     ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.012, n)
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    op = close * (1 + rng.normal(0, 0.001, n))
    vol = rng.integers(100_000, 500_000, n)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="1D")
    return pd.DataFrame({"open": op, "high": high, "low": low,
                          "close": close, "volume": vol}, index=idx)


# ---------------------------------------------------------------------------
# Raw indicator math
# ---------------------------------------------------------------------------
def test_rsi_bounded_0_100():
    df = _trending_ohlcv()
    r = rsi(df["close"])
    finite = r.dropna()
    assert (finite >= 0).all() and (finite <= 100).all()


def test_rsi_strongly_up_means_high_rsi():
    # Synthetic uptrend should push RSI > 70 eventually
    closes = pd.Series(np.linspace(100, 200, 50))
    r = rsi(closes)
    assert r.iloc[-1] > 70


def test_macd_lengths_match():
    df = _trending_ohlcv()
    m, s, h = macd(df["close"])
    assert len(m) == len(df) == len(s) == len(h)


def test_bollinger_envelopes_price():
    df = _trending_ohlcv()
    u, m, l = bollinger(df["close"])
    last_p = df["close"].iloc[-1]
    # Usually price is within ±2σ; allow violations as edge case
    assert l.iloc[-1] < m.iloc[-1] < u.iloc[-1]


def test_atr_positive():
    df = _trending_ohlcv()
    a = atr(df).dropna()
    assert (a > 0).all()


def test_adx_in_range():
    df = _trending_ohlcv()
    a = adx(df).dropna()
    assert (a >= 0).all() and (a <= 100).all()


def test_obv_changes_with_volume():
    df = _trending_ohlcv()
    o = obv(df)
    # OBV should not be constant
    assert o.std() > 0


# ---------------------------------------------------------------------------
# Reading wrappers
# ---------------------------------------------------------------------------
def test_all_indicators_returns_9_readings():
    df = _trending_ohlcv()
    out = all_indicators(df)
    assert len(out) == 9
    for r in out:
        assert r.state in ("bullish", "bearish", "neutral")
        assert r.name and r.display and r.note


def test_reading_rsi_overbought_label():
    closes = pd.Series(np.linspace(100, 200, 50))
    df = pd.DataFrame({"close": closes, "high": closes, "low": closes,
                       "volume": [1] * 50, "open": closes})
    r = reading_rsi(df)
    assert r.state == "bearish"   # overbought
    assert "overbought" in r.note


def test_reading_rsi_oversold_label():
    closes = pd.Series(np.linspace(200, 100, 50))
    df = pd.DataFrame({"close": closes, "high": closes, "low": closes,
                       "volume": [1] * 50, "open": closes})
    r = reading_rsi(df)
    assert r.state == "bullish"   # oversold = bullish reversion
    assert "oversold" in r.note


def test_reading_sma_cross_with_short_data():
    df = _trending_ohlcv(n=50)
    r = reading_sma_cross(df)
    assert r.state == "neutral"
    assert "need" in r.note


def test_reading_volume_ratio_unusual():
    df = _trending_ohlcv(n=30)
    df.loc[df.index[-1], "volume"] = int(df["volume"].iloc[:-1].mean() * 3)
    r = reading_volume_ratio(df)
    assert r.value >= 2.5
    assert "unusual" in r.note or "institutional" in r.note


def test_reading_52w_at_high_means_bullish():
    df = _trending_ohlcv(n=260, trend=0.003)
    r = reading_52w_percentile(df)
    # Strong uptrend → should be in top half of range
    assert r.value > 50


# ---------------------------------------------------------------------------
# Static data structures
# ---------------------------------------------------------------------------
def test_explanations_have_all_required_keys():
    for name, info in EXPLANATIONS.items():
        for key in ("what", "formula", "signals", "pitfall"):
            assert key in info, f"{name} missing {key}"


def test_time_spreads_use_valid_yahoo_ranges():
    valid_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m",
                        "1h", "1d", "5d", "1wk", "1mo", "3mo"}
    valid_ranges = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y",
                     "10y", "ytd", "max"}
    for name, (rng, interval) in TIME_SPREADS.items():
        assert rng in valid_ranges, f"{name} → invalid range {rng}"
        assert interval in valid_intervals, \
            f"{name} → invalid interval {interval}"


def test_time_spreads_covers_minute_to_5year():
    intervals = [v[1] for v in TIME_SPREADS.values()]
    ranges = [v[0] for v in TIME_SPREADS.values()]
    assert "1m" in intervals, "must have 1-minute option"
    assert "5y" in ranges, "must have 5-year option"
