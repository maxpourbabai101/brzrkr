"""Tests for src.data_alternatives.

yfinance and Finnhub are stubbed — the tests never touch the network.
The Black-Scholes math is validated against textbook values.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src import data_alternatives as dalt


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------
def test_bs_greeks_atm_call_textbook():
    # S=100, K=100, T=1yr, r=5%, sigma=20% — classic textbook scenario.
    g = dalt._bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
    # ATM call delta ~ 0.6368 for these inputs.
    assert g["delta"] == pytest.approx(0.6368, abs=0.005)
    # Gamma is small but positive.
    assert g["gamma"] > 0
    # Theta should be negative for a long option.
    assert g["theta"] < 0
    # Vega positive.
    assert g["vega"] > 0


def test_bs_greeks_atm_put_textbook():
    g = dalt._bs_greeks(100, 100, 1.0, 0.05, 0.20, "put")
    # Put-call parity: delta_put = delta_call - 1 ≈ -0.3632
    assert g["delta"] == pytest.approx(-0.3632, abs=0.005)
    assert g["gamma"] > 0
    assert g["vega"] > 0


def test_bs_greeks_expired_returns_nans():
    g = dalt._bs_greeks(100, 100, 0.0, 0.05, 0.20, "call")
    for v in g.values():
        assert np.isnan(v)


def test_bs_greeks_rejects_unknown_type():
    with pytest.raises(ValueError):
        dalt._bs_greeks(100, 100, 1.0, 0.05, 0.20, "swap")


# ---------------------------------------------------------------------------
# Options replacement (yfinance stub)
# ---------------------------------------------------------------------------
def _make_yf_stub(spot=500.0):
    """Return a callable matching the yfinance.Ticker(symbol) interface."""
    calls = pd.DataFrame({
        "contractSymbol": ["SPY260620C00500000", "SPY260620C00510000"],
        "strike":         [500.0, 510.0],
        "bid":            [12.0, 7.0],
        "ask":            [12.3, 7.2],
        "lastPrice":      [12.1, 7.1],
        "volume":         [1500, 800],
        "openInterest":   [50_000, 22_000],
        "impliedVolatility": [0.18, 0.20],
    })
    puts = pd.DataFrame({
        "contractSymbol": ["SPY260620P00500000"],
        "strike":         [500.0],
        "bid":            [10.0],
        "ask":            [10.3],
        "lastPrice":      [10.1],
        "volume":         [900],
        "openInterest":   [33_000],
        "impliedVolatility": [0.19],
    })
    chain = SimpleNamespace(calls=calls, puts=puts)

    class _Ticker:
        def __init__(self, _symbol):
            future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
            self.options = [future]
            self.fast_info = {"last_price": spot}
        def option_chain(self, _expiry):
            return chain
        def history(self, period="1d"):
            return pd.DataFrame({"Close": [spot]})

    return _Ticker


def test_fetch_options_free_returns_normalised_schema():
    df = dalt.fetch_options_free("SPY", ticker_factory=_make_yf_stub())
    expected_cols = {"symbol", "strike", "type", "expiry", "bid", "ask",
                     "last", "volume", "open_interest", "iv",
                     "delta", "gamma", "theta", "vega", "rho"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 3  # 2 calls + 1 put
    # Type column must be exactly call/put.
    assert set(df["type"]).issubset({"call", "put"})
    # spot must be attached as a DataFrame attr.
    assert df.attrs["spot"] == 500.0


def test_fetch_options_free_greeks_match_signs():
    df = dalt.fetch_options_free("SPY", ticker_factory=_make_yf_stub())
    calls = df[df["type"] == "call"]
    puts = df[df["type"] == "put"]
    # Call deltas in [0, 1]; put deltas in [-1, 0].
    assert (calls["delta"] >= 0).all() and (calls["delta"] <= 1).all()
    assert (puts["delta"] <= 0).all() and (puts["delta"] >= -1).all()
    # Gamma always >= 0.
    assert (df["gamma"] >= 0).all()


def test_fetch_options_free_no_greeks_when_disabled():
    df = dalt.fetch_options_free(
        "SPY", ticker_factory=_make_yf_stub(), compute_greeks=False
    )
    for greek in ("delta", "gamma", "theta", "vega", "rho"):
        assert greek not in df.columns


def test_fetch_options_free_specific_expiry():
    future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
    df = dalt.fetch_options_free(
        "SPY", expiry=future, ticker_factory=_make_yf_stub()
    )
    assert not df.empty
    assert (df["expiry"].dt.strftime("%Y-%m-%d") == future).all()


# ---------------------------------------------------------------------------
# News + sentiment replacement (Finnhub + FinBERT stubs)
# ---------------------------------------------------------------------------
def _fake_finnhub_news(symbol, since=None, until=None):
    idx = pd.to_datetime([
        "2026-05-20T13:00:00Z", "2026-05-21T09:30:00Z",
    ], utc=True)
    return pd.DataFrame({
        "headline": ["Apple beats expectations", "Apple guidance soft"],
        "summary":  ["Strong iPhone demand", "FX headwinds"],
        "source":   ["Reuters", "WSJ"],
        "category": ["company", "company"],
        "url":      ["https://x", "https://y"],
        "related":  ["AAPL", "AAPL"],
    }, index=idx)


class _FakeEncoder:
    def score_news(self, text):
        # Bullish if "beat", bearish if "soft", neutral otherwise.
        if "beat" in text.lower():
            return SimpleNamespace(score=0.6, salience=True, label_probs={})
        if "soft" in text.lower():
            return SimpleNamespace(score=-0.4, salience=True, label_probs={})
        return SimpleNamespace(score=0.0, salience=False, label_probs={})


def test_fetch_news_with_sentiment_scores_each_headline():
    df = dalt.fetch_news_with_sentiment(
        "AAPL", encoder=_FakeEncoder(), news_fn=_fake_finnhub_news
    )
    assert len(df) == 2
    assert {"title", "summary", "source", "url",
            "overall_sentiment_score", "overall_sentiment_label"}.issubset(df.columns)
    # Bullish headline should outscore bearish one.
    assert df.iloc[0]["overall_sentiment_score"] > df.iloc[1]["overall_sentiment_score"]
    assert df.iloc[0]["overall_sentiment_label"] == "Bullish"
    assert df.iloc[1]["overall_sentiment_label"] == "Bearish"


def test_fetch_news_with_sentiment_skip_scoring():
    df = dalt.fetch_news_with_sentiment(
        "AAPL", score=False, news_fn=_fake_finnhub_news
    )
    assert "overall_sentiment_score" not in df.columns
    assert len(df) == 2


def test_fetch_news_with_sentiment_empty_news():
    def empty(symbol, since=None, until=None):
        return pd.DataFrame()
    df = dalt.fetch_news_with_sentiment(
        "AAPL", encoder=_FakeEncoder(), news_fn=empty
    )
    assert df.empty
    assert "overall_sentiment_score" in df.columns


def test_score_to_label_buckets():
    assert dalt._score_to_label(0.5)  == "Bullish"
    assert dalt._score_to_label(0.2)  == "Somewhat-Bullish"
    assert dalt._score_to_label(0.0)  == "Neutral"
    assert dalt._score_to_label(-0.2) == "Somewhat-Bearish"
    assert dalt._score_to_label(-0.5) == "Bearish"
