"""Tests for src.data_loader.

External HTTP is mocked — we never hit real APIs from the test suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src import data_loader


def _ok_alpaca_bars():
    return {
        "bars": [
            {"t": "2026-05-19T00:00:00Z", "o": 100.0, "h": 101.0,
             "l": 99.5, "c": 100.5, "v": 12345},
            {"t": "2026-05-20T00:00:00Z", "o": 100.5, "h": 101.5,
             "l": 100.0, "c": 101.0, "v": 14567},
        ]
    }


def test_fetch_futures_parses_response(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")

    with patch.object(data_loader, "_request_with_retry", return_value=_ok_alpaca_bars()):
        df = data_loader.fetch_futures(
            "ES",
            datetime.now(timezone.utc) - timedelta(days=5),
            datetime.now(timezone.utc),
        )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing


def test_fetch_futures_missing_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        data_loader.fetch_futures(
            "ES",
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc),
        )


def test_fetch_news_empty_response(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "k")
    with patch.object(data_loader, "_request_with_retry", return_value={"articles": []}):
        df = data_loader.fetch_news("anything")
    assert df.empty
    assert "title" in df.columns


def test_fetch_macro_combines_series(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    payloads = iter([
        {"observations": [
            {"date": "2026-05-19", "value": "4.30"},
            {"date": "2026-05-20", "value": "4.32"},
        ]},
        {"observations": [
            {"date": "2026-05-19", "value": "16.5"},
            {"date": "2026-05-20", "value": "17.1"},
        ]},
    ])
    with patch.object(data_loader, "_request_with_retry", side_effect=lambda *a, **k: next(payloads)):
        df = data_loader.fetch_macro(["DGS10", "VIXCLS"])
    assert {"DGS10", "VIXCLS"}.issubset(df.columns)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Finnhub
# ---------------------------------------------------------------------------
def test_fetch_finnhub_news_parses(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    payload = [
        {"datetime": 1716384912, "headline": "h1", "summary": "s1",
         "source": "Reuters", "category": "company", "url": "https://x",
         "related": "AAPL", "id": 1},
        {"datetime": 1716471312, "headline": "h2", "summary": "s2",
         "source": "WSJ", "category": "company", "url": "https://y",
         "related": "AAPL", "id": 2},
    ]
    with patch.object(data_loader, "_request_with_retry", return_value=payload):
        df = data_loader.fetch_finnhub_news("AAPL")
    assert len(df) == 2
    assert "headline" in df.columns
    assert df.index.is_monotonic_increasing


def test_fetch_finnhub_news_empty(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    with patch.object(data_loader, "_request_with_retry", return_value=[]):
        df = data_loader.fetch_finnhub_news("AAPL")
    assert df.empty


def test_fetch_finnhub_news_missing_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        data_loader.fetch_finnhub_news("AAPL")


def test_fetch_finnhub_insider_parses(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    payload = {"data": [
        {"name": "Jane CEO", "share": 1000, "change": -500,
         "transactionCode": "S", "transactionPrice": 200.0,
         "transactionDate": "2026-05-10"},
        {"name": "Bob CFO", "share": 2000, "change": 200,
         "transactionCode": "P", "transactionPrice": 195.0,
         "transactionDate": "2026-05-12"},
    ]}
    with patch.object(data_loader, "_request_with_retry", return_value=payload):
        df = data_loader.fetch_finnhub_insider("AAPL")
    assert len(df) == 2
    assert {"name", "change", "transactionCode"}.issubset(df.columns)


def test_fetch_finnhub_earnings_parses(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    payload = {"earningsCalendar": [
        {"date": "2026-05-24", "symbol": "NVDA", "epsActual": None,
         "epsEstimate": 5.20, "hour": "amc", "quarter": 1, "year": 2026},
        {"date": "2026-05-25", "symbol": "CRM", "epsActual": None,
         "epsEstimate": 2.10, "hour": "bmo", "quarter": 1, "year": 2026},
    ]}
    with patch.object(data_loader, "_request_with_retry", return_value=payload):
        df = data_loader.fetch_finnhub_earnings_calendar()
    assert len(df) == 2
    assert {"symbol", "epsEstimate", "date"}.issubset(df.columns)


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------
def test_fetch_alpha_vantage_news_parses(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "k")
    payload = {"feed": [
        {"time_published": "20260522T133512", "title": "Apple beats",
         "summary": "...", "source": "Reuters", "url": "https://x",
         "overall_sentiment_score": "0.35", "overall_sentiment_label": "Bullish",
         "ticker_sentiment": [{"ticker": "AAPL", "ticker_sentiment_score": "0.4"}]},
        {"time_published": "20260522T180000", "title": "Apple guidance soft",
         "summary": "...", "source": "WSJ", "url": "https://y",
         "overall_sentiment_score": "-0.12", "overall_sentiment_label": "Somewhat-Bearish",
         "ticker_sentiment": []},
    ]}
    with patch.object(data_loader, "_request_with_retry", return_value=payload):
        df = data_loader.fetch_alpha_vantage_news("AAPL")
    assert len(df) == 2
    assert df["overall_sentiment_score"].dtype.kind == "f"
    assert df.index.is_monotonic_increasing


def test_fetch_alpha_vantage_rate_limit(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "k")
    payload = {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."}
    with patch.object(data_loader, "_request_with_retry", return_value=payload):
        df = data_loader.fetch_alpha_vantage_news("AAPL")
    assert df.empty


# ---------------------------------------------------------------------------
# Polygon
# ---------------------------------------------------------------------------
def test_fetch_polygon_aggregates_parses(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    payload = {
        "ticker": "AAPL",
        "status": "OK",
        "results": [
            {"t": 1716163200000, "o": 189.0, "h": 191.5, "l": 188.2,
             "c": 190.4, "v": 12_345_000, "vw": 190.1, "n": 95_000},
            {"t": 1716249600000, "o": 190.4, "h": 192.0, "l": 189.5,
             "c": 191.6, "v": 13_456_000, "vw": 191.0, "n": 98_000},
        ],
    }
    with patch.object(data_loader, "_request_with_retry", return_value=payload):
        df = data_loader.fetch_polygon_aggregates(
            "AAPL",
            datetime.now(timezone.utc) - timedelta(days=5),
            datetime.now(timezone.utc),
        )
    assert len(df) == 2
    assert {"open", "high", "low", "close", "volume", "vwap", "trades"}.issubset(df.columns)
    assert df.index.is_monotonic_increasing


def test_fetch_polygon_empty(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    with patch.object(data_loader, "_request_with_retry", return_value={"results": []}):
        df = data_loader.fetch_polygon_aggregates(
            "AAPL",
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc),
        )
    assert df.empty
