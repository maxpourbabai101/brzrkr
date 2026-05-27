"""Tests for src.data_scraper.WebDataScraper.

All HTTP is mocked — the test suite never touches the network. The
session.get method is patched on a per-instance basis so each test
controls exactly what payload comes back.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_scraper import WebDataScraper, _score_to_label


# ---------------------------------------------------------------------------
# Fake HTTP helpers
# ---------------------------------------------------------------------------
def _fake_response(*, json_payload=None, text=None, status=200):
    r = MagicMock()
    r.status_code = status
    r.raise_for_status = MagicMock()
    if json_payload is not None:
        r.json.return_value = json_payload
    if text is not None:
        r.text = text
    return r


@pytest.fixture
def scraper():
    s = WebDataScraper(delay_s=0)  # disable polite delay in tests
    return s


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------
def test_scrape_ohlcv_parses_yahoo_chart(scraper):
    payload = {
        "chart": {
            "result": [{
                "timestamp": [1716163200, 1716249600],
                "indicators": {"quote": [{
                    "open":   [100.0, 101.0],
                    "high":   [102.0, 102.5],
                    "low":    [99.5, 100.5],
                    "close":  [101.5, 102.0],
                    "volume": [1_000_000, 1_200_000],
                }]},
            }]
        }
    }
    with patch.object(scraper.session, "get", return_value=_fake_response(json_payload=payload)):
        df = scraper.scrape_ohlcv("SPY")
    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing


def test_scrape_ohlcv_empty(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload={"chart": {"result": []}})):
        df = scraper.scrape_ohlcv("SPY")
    assert df.empty


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def test_scrape_options_parses_calls_and_puts(scraper):
    payload = {
        "optionChain": {
            "result": [{
                "quote": {"regularMarketPrice": 500.0},
                "options": [{
                    "expirationDate": 1719360000,
                    "calls": [{
                        "contractSymbol": "SPY260620C00500000",
                        "strike": 500.0, "bid": 12.0, "ask": 12.3,
                        "lastPrice": 12.1, "volume": 1500,
                        "openInterest": 50000, "impliedVolatility": 0.18,
                    }],
                    "puts": [{
                        "contractSymbol": "SPY260620P00500000",
                        "strike": 500.0, "bid": 10.0, "ask": 10.3,
                        "lastPrice": 10.1, "volume": 900,
                        "openInterest": 33000, "impliedVolatility": 0.19,
                    }],
                }],
            }]
        }
    }
    with patch.object(scraper.session, "get", return_value=_fake_response(json_payload=payload)):
        df = scraper.scrape_options("SPY")
    assert len(df) == 2
    assert set(df["type"]) == {"call", "put"}
    assert df.attrs["spot"] == 500.0
    assert {"strike", "iv", "open_interest", "expiry"}.issubset(df.columns)


# ---------------------------------------------------------------------------
# RSS news
# ---------------------------------------------------------------------------
RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Apple beats earnings</title>
      <description>Strong iPhone demand fuels Q1.</description>
      <link>https://example.com/a</link>
      <pubDate>Wed, 21 May 2026 13:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Apple guidance soft</title>
      <description>FX headwinds cited.</description>
      <link>https://example.com/b</link>
      <pubDate>Wed, 21 May 2026 18:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""


def test_scrape_news_rss_aggregates_feeds(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(text=RSS_SAMPLE)):
        df = scraper.scrape_news_rss("AAPL", feeds=["https://fake/{symbol}"])
    assert len(df) == 2
    assert {"title", "summary", "url", "source"}.issubset(df.columns)


def test_scrape_news_rss_handles_broken_xml(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(text="<not-xml>")):
        df = scraper.scrape_news_rss("AAPL", feeds=["https://fake"])
    assert df.empty


# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------
EDGAR_TICKER_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

EDGAR_ATOM_FORM4 = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <updated>2026-05-20T13:30:00Z</updated>
    <title>4 - Insider Trade</title>
    <link href="https://www.sec.gov/Archives/edgar/data/320193/000032019326000050/0000320193-26-000050-index.htm"/>
    <content>Accession Number: 0000320193-26-000050</content>
  </entry>
  <entry>
    <updated>2026-05-18T09:15:00Z</updated>
    <title>4 - Insider Trade</title>
    <link href="https://www.sec.gov/Archives/edgar/data/320193/000032019326000049/0000320193-26-000049-index.htm"/>
    <content>Accession Number: 0000320193-26-000049</content>
  </entry>
</feed>"""


def test_scrape_insider_form4_resolves_cik_and_parses_atom(scraper):
    # First call → ticker lookup; second call → atom feed.
    responses = iter([
        _fake_response(json_payload=EDGAR_TICKER_PAYLOAD),
        _fake_response(text=EDGAR_ATOM_FORM4),
    ])
    with patch.object(scraper.session, "get", side_effect=lambda *a, **k: next(responses)):
        df = scraper.scrape_insider_form4("AAPL")
    assert len(df) == 2
    assert {"title", "url", "accession"}.issubset(df.columns)
    assert df["accession"].iloc[0].startswith("0000320193")


def test_scrape_insider_unknown_ticker_returns_empty(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload=EDGAR_TICKER_PAYLOAD)):
        df = scraper.scrape_insider_form4("ZZZZ")
    assert df.empty


# ---------------------------------------------------------------------------
# CFTC COT
# ---------------------------------------------------------------------------
def test_scrape_cot_returns_raw_lines(scraper):
    text = "HEADER LINE\nDATA LINE 1\nDATA LINE 2\n"
    with patch.object(scraper.session, "get", return_value=_fake_response(text=text)):
        df = scraper.scrape_cot()
    assert len(df) == 3
    assert "raw" in df.columns


# ---------------------------------------------------------------------------
# Sentiment scoring (mocked encoder)
# ---------------------------------------------------------------------------
class _FakeEncoder:
    def score_news(self, text):
        if "beat" in text.lower():
            return SimpleNamespace(score=0.6, salience=True, label_probs={})
        if "soft" in text.lower():
            return SimpleNamespace(score=-0.5, salience=True, label_probs={})
        return SimpleNamespace(score=0.0, salience=False, label_probs={})


def test_score_sentiment_appends_columns(scraper):
    news = pd.DataFrame({
        "title":   ["Apple beats", "Apple soft"],
        "summary": ["earnings", "guidance"],
    })
    out = scraper.score_sentiment(news, encoder=_FakeEncoder())
    assert {"sentiment_score", "sentiment_label"}.issubset(out.columns)
    assert out.iloc[0]["sentiment_label"] == "Bullish"
    assert out.iloc[1]["sentiment_label"] == "Bearish"


def test_score_sentiment_empty_input(scraper):
    out = scraper.score_sentiment(pd.DataFrame(), encoder=_FakeEncoder())
    assert out.empty


def test_score_label_buckets():
    assert _score_to_label(0.5) == "Bullish"
    assert _score_to_label(0.2) == "Somewhat-Bullish"
    assert _score_to_label(0.0) == "Neutral"
    assert _score_to_label(-0.2) == "Somewhat-Bearish"
    assert _score_to_label(-0.5) == "Bearish"


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def test_scrape_all_gracefully_degrades(scraper):
    # Every source raises; scrape_all should fill empty frames, never raise.
    with patch.object(scraper.session, "get", side_effect=Exception("boom")):
        out = scraper.scrape_all("SPY", extras=True)
    # StockTwits intentionally NOT in default extras (free API blocks IPs).
    expected = {"prices", "options", "news", "insider", "treasury", "cot",
                "short_vol", "reddit", "congress", "google_news",
                "hacker_news"}
    assert expected.issubset(set(out.keys()))
    for k, v in out.items():
        assert v.empty, f"{k} should be empty after failure"


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------
def test_scrape_reddit_mentions_parses(scraper):
    payload = {
        "data": {"children": [
            {"data": {
                "subreddit": "wallstreetbets",
                "title": "AAPL to the moon",
                "selftext": "Loaded up calls",
                "score": 1500,
                "num_comments": 230,
                "url": "https://reddit.com/r/wallstreetbets/comments/abc",
                "author": "diamondhands",
                "created_utc": 1716384912,
            }},
            {"data": {
                "subreddit": "wallstreetbets",
                "title": "AAPL guidance cut",
                "selftext": "Bear case",
                "score": 800,
                "num_comments": 140,
                "url": "https://reddit.com/r/wallstreetbets/comments/def",
                "author": "puts_only",
                "created_utc": 1716471312,
            }},
        ]}
    }
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload=payload)):
        df = scraper.scrape_reddit_mentions("AAPL", subreddits=["wallstreetbets"])
    assert len(df) == 2
    assert {"subreddit", "title", "score", "num_comments"}.issubset(df.columns)
    # Should be sorted desc by created_at.
    assert df.index.is_monotonic_decreasing


def test_scrape_reddit_handles_empty(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload={"data": {"children": []}})):
        df = scraper.scrape_reddit_mentions("ZZZZ", subreddits=["stocks"])
    assert df.empty


# ---------------------------------------------------------------------------
# Congressional trades
# ---------------------------------------------------------------------------
def test_scrape_congress_trades_combines_chambers(scraper):
    recent = datetime.now(timezone.utc) - pd.Timedelta(days=5)
    recent_str = recent.strftime("%Y-%m-%d")
    senate_payload = [
        {"senator": "Smith", "ticker": "AAPL", "type": "Purchase",
         "amount": "$15,001 - $50,000", "transaction_date": recent_str,
         "disclosure_date": recent_str},
        {"senator": "Jones", "ticker": "MSFT", "type": "Sale",
         "amount": "$50,001 - $100,000", "transaction_date": recent_str,
         "disclosure_date": recent_str},
    ]
    house_payload = [
        {"representative": "Brown", "ticker": "AAPL", "type": "Purchase",
         "amount": "$1,001 - $15,000", "transaction_date": recent_str,
         "disclosure_date": recent_str},
    ]
    responses = iter([
        _fake_response(json_payload=senate_payload),
        _fake_response(json_payload=house_payload),
    ])
    with patch.object(scraper.session, "get", side_effect=lambda *a, **k: next(responses)):
        df = scraper.scrape_congress_trades(symbol="AAPL")
    # Two AAPL trades — one Senate, one House.
    assert len(df) == 2
    assert set(df["chamber"]) == {"senate", "house"}
    assert (df["ticker"] == "AAPL").all()


def test_scrape_congress_trades_filters_by_lookback(scraper):
    old = datetime.now(timezone.utc) - pd.Timedelta(days=400)
    new = datetime.now(timezone.utc) - pd.Timedelta(days=10)
    senate_payload = [
        {"senator": "Old Smith", "ticker": "AAPL", "type": "Purchase",
         "amount": "$1,001 - $15,000",
         "transaction_date": old.strftime("%Y-%m-%d"),
         "disclosure_date": old.strftime("%Y-%m-%d")},
        {"senator": "New Smith", "ticker": "AAPL", "type": "Purchase",
         "amount": "$1,001 - $15,000",
         "transaction_date": new.strftime("%Y-%m-%d"),
         "disclosure_date": new.strftime("%Y-%m-%d")},
    ]
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload=senate_payload)):
        df = scraper.scrape_congress_trades(
            symbol="AAPL", lookback_days=60, chambers=["senate"]
        )
    assert len(df) == 1
    assert df.iloc[0]["representative"] == "New Smith"


# ---------------------------------------------------------------------------
# Google News
# ---------------------------------------------------------------------------
def test_scrape_google_news_parses_rss(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(text=RSS_SAMPLE)):
        df = scraper.scrape_google_news("AAPL")
    assert len(df) == 2
    assert {"title", "url"}.issubset(df.columns)


# ---------------------------------------------------------------------------
# StockTwits
# ---------------------------------------------------------------------------
def test_scrape_stocktwits_parses_messages(scraper):
    payload = {"messages": [
        {"id": 1, "user": {"username": "trader1"}, "body": "Bullish on AAPL",
         "entities": {"sentiment": {"basic": "Bullish"}},
         "likes": {"total": 5}, "created_at": "2026-05-22T13:00:00Z"},
        {"id": 2, "user": {"username": "trader2"}, "body": "Hedging here",
         "entities": {}, "likes": {"total": 0},
         "created_at": "2026-05-22T13:30:00Z"},
    ]}
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload=payload)):
        df = scraper.scrape_stocktwits("AAPL")
    assert len(df) == 2
    assert df.iloc[0]["user"] in {"trader1", "trader2"}
    assert "sentiment" in df.columns


def test_scrape_stocktwits_empty(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload={"messages": []})):
        df = scraper.scrape_stocktwits("AAPL")
    assert df.empty


# ---------------------------------------------------------------------------
# Hacker News
# ---------------------------------------------------------------------------
def test_scrape_hacker_news_parses_rss(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(text=RSS_SAMPLE)):
        df = scraper.scrape_hacker_news("Apple")
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Wikipedia pageviews
# ---------------------------------------------------------------------------
def test_scrape_wikipedia_pageviews_parses(scraper):
    payload = {"items": [
        {"timestamp": "2026052000", "views": 15234},
        {"timestamp": "2026052100", "views": 17890},
        {"timestamp": "2026052200", "views": 22100},
    ]}
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload=payload)):
        df = scraper.scrape_wikipedia_pageviews("Apple_Inc.")
    assert len(df) == 3
    assert "views" in df.columns
    assert df.index.is_monotonic_increasing


def test_scrape_wikipedia_pageviews_empty(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload={"items": []})):
        df = scraper.scrape_wikipedia_pageviews("Nonexistent_Topic_XYZ")
    assert df.empty


# ---------------------------------------------------------------------------
# scrape_all with extras enabled — structural test, no FinBERT load
# ---------------------------------------------------------------------------
def test_scrape_all_with_extras_returns_every_key(scraper):
    """All extras should produce keys even if their HTTP calls 404."""
    reddit_payload = {"data": {"children": [
        {"data": {"subreddit": "stocks", "title": "AAPL beats",
                  "selftext": "Earnings strong", "score": 100,
                  "num_comments": 10, "url": "u1", "author": "x",
                  "created_utc": 1716384912}},
    ]}}
    stocktwits_payload = {"messages": [
        {"id": 1, "user": {"username": "u"}, "body": "Long AAPL",
         "entities": {"sentiment": {"basic": "Bullish"}},
         "likes": {"total": 1}, "created_at": "2026-05-22T13:00:00Z"},
    ]}

    def route(url, *args, **kwargs):
        if "reddit.com" in url:
            return _fake_response(json_payload=reddit_payload)
        if "stocktwits.com" in url:
            return _fake_response(json_payload=stocktwits_payload)
        if "news.google.com" in url or "hnrss" in url or "yahoo" in url \
                or "marketwatch" in url or "reutersagency" in url or "cnbc" in url:
            return _fake_response(text=RSS_SAMPLE)
        # Everything else → empty payload.
        return _fake_response(json_payload={}, text="")

    with patch.object(scraper.session, "get", side_effect=route):
        out = scraper.scrape_all("AAPL", score=False, extras=True)
    for key in ("prices", "options", "news", "insider", "treasury", "cot",
                "short_vol", "reddit", "congress", "google_news",
                "hacker_news"):
        assert key in out, f"missing key {key}"
    assert "stocktwits" not in out  # not in default extras anymore


def test_scrape_all_extras_off_skips_crowd_sources(scraper):
    with patch.object(scraper.session, "get",
                      return_value=_fake_response(json_payload={}, text="")):
        out = scraper.scrape_all("AAPL", score=False, extras=False)
    assert "reddit" not in out
    assert "congress" not in out
    assert "stocktwits" not in out


def test_normalise_text_columns_maps_reddit():
    from src.data_scraper import _normalise_text_columns
    df = pd.DataFrame({"title": ["a"], "selftext": ["b"]})
    out = _normalise_text_columns(df, "reddit")
    assert out["title"].iloc[0] == "a"
    assert out["summary"].iloc[0] == "b"


def test_normalise_text_columns_maps_stocktwits():
    from src.data_scraper import _normalise_text_columns
    df = pd.DataFrame({"body": ["hello"]})
    out = _normalise_text_columns(df, "stocktwits")
    assert out["title"].iloc[0] == "hello"
    assert out["summary"].iloc[0] == ""
