"""Tests for src.features.feature_engineer.FeatureEngineer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineer import FeatureEngineer, _rsi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fe():
    return FeatureEngineer(window=64)


def _price_frame(n=300, start=100.0):
    rng = np.random.default_rng(0)
    base = start + np.cumsum(rng.normal(0.05, 1.0, size=n))
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="1D")
    return pd.DataFrame({
        "open":   base + rng.normal(0, 0.2, n),
        "high":   base + 0.5,
        "low":    base - 0.5,
        "close":  base,
        "volume": rng.integers(10_000, 200_000, n),
    }, index=idx)


# ---------------------------------------------------------------------------
# Technical features
# ---------------------------------------------------------------------------
def test_build_features_adds_technical_columns(fe):
    bundle = {"prices": _price_frame()}
    out = fe.build_features(bundle)
    assert {"ret_1", "ret_5", "ret_20", "sma_20", "sma_50",
            "sma_spread", "realized_vol", "rsi"}.issubset(out.columns)
    assert len(out) == fe.window  # tail-window applied


def test_build_features_attaches_context(fe):
    bundle = {"prices": _price_frame()}
    out = fe.build_features(bundle)
    ctx = out.attrs["context"]
    # Every text source should at least have a count key.
    for k in ("news_count", "reddit_count", "stocktwits_count",
              "hacker_news_count", "google_news_count"):
        assert k in ctx
    # Smart-money keys should always be present.
    for k in ("congress_buys_60d", "congress_sells_60d",
              "congress_net_60d", "insider_filings_60d"):
        assert k in ctx


def test_build_features_empty_prices_returns_empty(fe):
    assert fe.build_features({"prices": pd.DataFrame()}).empty
    assert fe.build_features({}).empty


def test_rsi_within_bounds():
    series = pd.Series(np.linspace(100, 120, 50))  # monotonic up
    rsi = _rsi(series, period=14)
    # Last RSI should be > 70 (strong uptrend).
    assert 70 < rsi.iloc[-1] <= 100


# ---------------------------------------------------------------------------
# Sentiment aggregation
# ---------------------------------------------------------------------------
def test_sentiment_features_aggregates_scores(fe):
    news = pd.DataFrame({
        "title": ["a", "b"],
        "sentiment_score": [0.5, -0.3],
    })
    bundle = {"prices": _price_frame(), "news": news}
    ctx = fe.build_features(bundle).attrs["context"]
    assert ctx["news_count"] == 2.0
    assert ctx["news_sent_mean"] == pytest.approx(0.1, abs=1e-6)
    assert ctx["news_sent_salience"] == pytest.approx(0.5, abs=1e-6)


def test_sentiment_reads_av_overall_score(fe):
    av = pd.DataFrame({
        "title": ["x"],
        "overall_sentiment_score": [0.42],
    })
    bundle = {"prices": _price_frame(), "av_news": av}
    ctx = fe.build_features(bundle).attrs["context"]
    assert ctx["av_news_sent_mean"] == pytest.approx(0.42, abs=1e-6)


# ---------------------------------------------------------------------------
# Insider / congress
# ---------------------------------------------------------------------------
def test_insider_features_counts_recent_filings(fe):
    idx = pd.to_datetime([
        datetime.now(timezone.utc) - timedelta(days=5),
        datetime.now(timezone.utc) - timedelta(days=10),
        datetime.now(timezone.utc) - timedelta(days=400),  # outside 60d window
    ], utc=True)
    insider = pd.DataFrame({"title": ["t1", "t2", "t3"]}, index=idx)
    insider.index.name = "filed_at"
    bundle = {"prices": _price_frame(), "insider": insider}
    ctx = fe.build_features(bundle).attrs["context"]
    assert ctx["insider_filings_60d"] == 2.0


def test_congress_features_buys_minus_sells(fe):
    df = pd.DataFrame({
        "type": ["Purchase", "Purchase", "Sale", "Exchange"],
    })
    bundle = {"prices": _price_frame(), "congress": df}
    ctx = fe.build_features(bundle).attrs["context"]
    assert ctx["congress_buys_60d"] == 2.0
    assert ctx["congress_sells_60d"] == 1.0
    assert ctx["congress_net_60d"] == 1.0


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def test_options_features_pc_ratio_and_iv_atm(fe):
    opts = pd.DataFrame({
        "type": ["call", "call", "put", "put"],
        "strike": [100, 110, 90, 100],
        "volume": [1000, 500, 800, 1200],
        "iv": [0.20, 0.22, 0.25, 0.21],
    })
    bundle = {"prices": _price_frame(start=100), "options": opts}
    out = fe.build_features(bundle)
    ctx = out.attrs["context"]
    assert ctx["pc_ratio"] == pytest.approx((800 + 1200) / (1000 + 500))
    # IV ATM = nearest strike to spot. Spot is ~ last close in price frame.
    assert not np.isnan(ctx["iv_atm"])


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------
def test_macro_features_yield_curve(fe):
    macro = pd.DataFrame(
        {"DGS10": [4.3, 4.32], "DGS2": [4.8, 4.78], "VIXCLS": [16.5, 17.1]},
        index=pd.date_range("2026-05-19", periods=2, freq="D", tz="UTC"),
    )
    bundle = {"prices": _price_frame(), "macro": macro}
    ctx = fe.build_features(bundle).attrs["context"]
    assert ctx["y10y2_spread"] == pytest.approx(4.32 - 4.78, abs=1e-6)
    assert ctx["vix_level"] == pytest.approx(17.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Attention proxies
# ---------------------------------------------------------------------------
def test_wiki_views_z_score(fe):
    # 30 days of 1000 views, then a spike of 5000 today.
    n = 31
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="D")
    views = [1000] * (n - 1) + [5000]
    wiki = pd.DataFrame({"views": views}, index=idx)
    wiki.index.name = "timestamp"
    bundle = {"prices": _price_frame(), "wiki_views": wiki}
    ctx = fe.build_features(bundle).attrs["context"]
    # Z-score should be large positive (>> 5) since baseline std is 0.
    assert ctx["wiki_views_z"] > 5
