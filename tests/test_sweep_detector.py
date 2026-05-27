"""Tests for src.scanners.sweep_detector.SweepDetector."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scanners.sweep_detector import (
    DEFAULT_SMALL_CAP_WATCHLIST, SweepAlert, SweepDetector,
)


def _make_ohlcv(n=30, last_volume=None, last_return=None, base_vol=100_000,
                 base_close=10.0):
    """Build a synthetic OHLCV frame with controllable last bar."""
    np.random.seed(0)
    closes = base_close + np.cumsum(np.random.normal(0, 0.05, n))
    closes = np.clip(closes, 1, None)
    if last_return is not None:
        closes[-1] = closes[-2] * (1 + last_return)
    volumes = np.full(n, base_vol).astype(int)
    if last_volume is not None:
        volumes[-1] = int(last_volume)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="1D")
    return pd.DataFrame({
        "open":   closes,
        "high":   closes * 1.005,
        "low":    closes * 0.995,
        "close":  closes,
        "volume": volumes,
    }, index=idx)


@pytest.fixture
def detector(tmp_path):
    return SweepDetector(
        watchlist=["FAKE"],
        output_path=tmp_path / "sweeps.jsonl",
    )


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------
def test_no_sweep_on_quiet_day(detector):
    """Normal volume + small return → no alert."""
    df = _make_ohlcv(last_volume=100_000, last_return=0.002)
    detector.fetch_fn = lambda sym: df
    assert detector.scan_symbol("FAKE") is None


def test_detects_volume_plus_price_spike(detector):
    """3x volume + 6% price move → alert."""
    df = _make_ohlcv(last_volume=500_000, last_return=0.06)
    detector.fetch_fn = lambda sym: df
    alert = detector.scan_symbol("FAKE")
    assert alert is not None
    assert alert.direction == "up"
    assert alert.volume_ratio >= 3.0
    assert alert.price_change_pct >= 5.0


def test_detects_down_sweep(detector):
    df = _make_ohlcv(last_volume=500_000, last_return=-0.07)
    detector.fetch_fn = lambda sym: df
    alert = detector.scan_symbol("FAKE")
    assert alert is not None
    assert alert.direction == "down"
    assert alert.price_change_pct < 0


def test_skips_when_only_volume_spikes(detector):
    """High volume but tiny price move → no alert."""
    df = _make_ohlcv(last_volume=500_000, last_return=0.001)
    detector.fetch_fn = lambda sym: df
    assert detector.scan_symbol("FAKE") is None


def test_skips_when_only_price_spikes_without_volume(detector):
    df = _make_ohlcv(last_volume=100_000, last_return=0.07)
    detector.fetch_fn = lambda sym: df
    # Volume ratio is 1.0; volume_multiplier defaults to 3.0 → no alert
    assert detector.scan_symbol("FAKE") is None


def test_liquidity_floor_drops_microcaps(detector):
    """If 20d avg dollar volume below floor, no alert."""
    df = _make_ohlcv(last_volume=200_000, last_return=0.08,
                      base_vol=1_000, base_close=0.5)
    # 1000 vol * $0.5 = $500/day average — under default 100k floor
    detector.fetch_fn = lambda sym: df
    assert detector.scan_symbol("FAKE") is None


def test_max_price_filter(detector):
    """Max_price filter excludes higher-priced names."""
    df = _make_ohlcv(last_volume=500_000, last_return=0.07,
                      base_close=200)
    detector.max_price = 50  # penny scope only
    detector.fetch_fn = lambda sym: df
    assert detector.scan_symbol("FAKE") is None


# ---------------------------------------------------------------------------
# Multi-symbol scan + write/load
# ---------------------------------------------------------------------------
def test_scan_returns_sorted_by_score(detector):
    big_df = _make_ohlcv(last_volume=600_000, last_return=0.08)
    small_df = _make_ohlcv(last_volume=400_000, last_return=0.06)

    def fetch(sym):
        return big_df if sym == "BIG" else small_df

    detector.fetch_fn = fetch
    detector.watchlist = ["SMALL", "BIG"]
    alerts = detector.scan()
    assert len(alerts) == 2
    assert alerts[0].symbol == "BIG"  # bigger spike → higher score
    assert alerts[0].score >= alerts[1].score


def test_write_and_load_round_trip(detector, tmp_path):
    df = _make_ohlcv(last_volume=500_000, last_return=0.07)
    detector.fetch_fn = lambda sym: df
    alerts = detector.scan()
    assert alerts
    detector.write(alerts)
    assert detector.output_path.exists()

    loaded = SweepDetector.load_recent(detector.output_path, limit=10)
    assert len(loaded) == 1
    assert loaded[0]["symbol"] == "FAKE"


def test_load_handles_missing_file(tmp_path):
    out = SweepDetector.load_recent(tmp_path / "nope.jsonl")
    assert out == []


def test_load_handles_corrupt_lines(tmp_path):
    p = tmp_path / "sweeps.jsonl"
    p.write_text(
        '{"symbol":"OK","detected_at":"2026-01-01T00:00:00"}\n'
        'not json\n'
        '{"symbol":"GOOD","detected_at":"2026-01-02T00:00:00"}\n'
    )
    out = SweepDetector.load_recent(p)
    # Two valid records, sorted by detected_at desc
    assert len(out) == 2
    assert out[0]["symbol"] == "GOOD"


def test_default_watchlist_nonempty():
    assert len(DEFAULT_SMALL_CAP_WATCHLIST) > 20
    # All should be uppercase tickers
    for s in DEFAULT_SMALL_CAP_WATCHLIST:
        assert s.upper() == s
