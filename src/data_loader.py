"""Data ingestion layer for trading_enhancer.

Pulls OHLCV (futures + equities), options chains/Greeks, news/social
sentiment text, and macro series from a configurable set of vendors.
All credentials are read from environment variables — never hard‑code
keys here. Configure them in `config/secrets.yaml` or your shell.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vendor endpoint configuration
# ---------------------------------------------------------------------------
# Insert / override these via environment variables.
POLYGON_BASE = "https://api.polygon.io"
ALPACA_DATA_BASE = "https://data.alpaca.markets"
TRADIER_BASE = "https://api.tradier.com/v1"
UNUSUAL_WHALES_BASE = "https://api.unusualwhales.com/api"
NEWSAPI_BASE = "https://newsapi.org/v2"
FRED_BASE = "https://api.stlouisfed.org/fred"
FINNHUB_BASE = "https://finnhub.io/api/v1"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

DEFAULT_TIMEOUT = 15  # seconds
MAX_RETRIES = 3
BACKOFF_BASE = 1.6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
@dataclass
class _Creds:
    """Bundle of API credentials pulled from the environment."""

    polygon: Optional[str] = None
    alpaca_key: Optional[str] = None
    alpaca_secret: Optional[str] = None
    tradier: Optional[str] = None
    unusual_whales: Optional[str] = None
    newsapi: Optional[str] = None
    fred: Optional[str] = None
    finnhub: Optional[str] = None
    alpha_vantage: Optional[str] = None

    @classmethod
    def from_env(cls) -> "_Creds":
        return cls(
            polygon=os.getenv("POLYGON_API_KEY"),
            alpaca_key=os.getenv("ALPACA_API_KEY"),
            alpaca_secret=os.getenv("ALPACA_SECRET_KEY"),
            tradier=os.getenv("TRADIER_API_KEY"),
            unusual_whales=os.getenv("UNUSUAL_WHALES_API_KEY"),
            newsapi=os.getenv("NEWSAPI_KEY"),
            fred=os.getenv("FRED_API_KEY"),
            finnhub=os.getenv("FINNHUB_API_KEY"),
            alpha_vantage=os.getenv("ALPHA_VANTAGE_API_KEY"),
        )


def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """HTTP request with exponential‑backoff retry on transient failures."""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method, url, headers=headers, params=params, timeout=timeout
            )
            if resp.status_code == 429:
                # Rate‑limit — back off and retry.
                wait = BACKOFF_BASE ** attempt
                logger.warning("Rate‑limited on %s; sleeping %.1fs", url, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:  # network/HTTP error
            last_exc = exc
            wait = BACKOFF_BASE ** attempt
            logger.warning(
                "Request failed (%s) attempt %d/%d on %s — retrying in %.1fs",
                exc, attempt, MAX_RETRIES, url, wait,
            )
            time.sleep(wait)
    logger.error("All retries exhausted for %s: %s", url, last_exc)
    raise RuntimeError(f"Request failed after {MAX_RETRIES} retries: {url}") from last_exc


# ---------------------------------------------------------------------------
# Public fetchers
# ---------------------------------------------------------------------------
def fetch_futures(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    """Fetch futures OHLCV bars from Alpaca/Polygon.

    Returns a DataFrame indexed by UTC timestamp with columns
    ``[open, high, low, close, volume]``.
    """
    creds = _Creds.from_env()
    if not creds.alpaca_key or not creds.alpaca_secret:
        raise EnvironmentError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY must be set in the environment."
        )

    url = f"{ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars"
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "timeframe": timeframe,
        "adjustment": "raw",
        "feed": "iex",
    }
    headers = {
        "APCA-API-KEY-ID": creds.alpaca_key,
        "APCA-API-SECRET-KEY": creds.alpaca_secret,
    }
    payload = _request_with_retry("GET", url, headers=headers, params=params)
    bars: List[Dict[str, Any]] = payload.get("bars", []) or []
    if not bars:
        logger.warning("fetch_futures returned no rows for %s", symbol)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(bars)
    df = df.rename(
        columns={"t": "timestamp", "o": "open", "h": "high",
                 "l": "low", "c": "close", "v": "volume"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def fetch_options(
    underlying: str,
    expiry: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch options chain + Greeks for `underlying` via Tradier.

    If `expiry` is omitted, the nearest expiration is used. Returns a
    DataFrame keyed by ``(strike, type, expiry)`` containing bid/ask,
    open interest, volume, and Greeks (delta, gamma, theta, vega, IV).
    """
    creds = _Creds.from_env()
    if not creds.tradier:
        raise EnvironmentError("TRADIER_API_KEY must be set.")

    headers = {
        "Authorization": f"Bearer {creds.tradier}",
        "Accept": "application/json",
    }

    if expiry is None:
        exp_url = f"{TRADIER_BASE}/markets/options/expirations"
        exp_payload = _request_with_retry(
            "GET", exp_url, headers=headers,
            params={"symbol": underlying, "includeAllRoots": "true"},
        )
        dates = (exp_payload.get("expirations") or {}).get("date") or []
        if not dates:
            raise RuntimeError(f"No expirations available for {underlying}")
        expiry = dates[0] if isinstance(dates, list) else dates

    chain_url = f"{TRADIER_BASE}/markets/options/chains"
    payload = _request_with_retry(
        "GET", chain_url, headers=headers,
        params={"symbol": underlying, "expiration": expiry, "greeks": "true"},
    )
    raw = (payload.get("options") or {}).get("option") or []
    if not raw:
        return pd.DataFrame()

    rows = []
    for opt in raw:
        greeks = opt.get("greeks") or {}
        rows.append({
            "symbol": opt.get("symbol"),
            "strike": opt.get("strike"),
            "type": opt.get("option_type"),
            "expiry": opt.get("expiration_date"),
            "bid": opt.get("bid"),
            "ask": opt.get("ask"),
            "last": opt.get("last"),
            "volume": opt.get("volume"),
            "open_interest": opt.get("open_interest"),
            "iv": greeks.get("mid_iv"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
        })
    df = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    return df


def fetch_news(
    query: str,
    since: Optional[datetime] = None,
    page_size: int = 100,
) -> pd.DataFrame:
    """Fetch recent headlines from NewsAPI.

    Returns DataFrame with columns ``[published_at, title, description,
    source, url]`` indexed by ``published_at`` (UTC).

    The NewsAPI developer tier only serves articles from the last ~30
    days; the `since` argument is clamped accordingly. Paid tiers can
    override by setting NEWSAPI_MAX_LOOKBACK_DAYS.
    """
    creds = _Creds.from_env()
    if not creds.newsapi:
        raise EnvironmentError("NEWSAPI_KEY must be set.")

    max_days = int(os.getenv("NEWSAPI_MAX_LOOKBACK_DAYS", "28"))
    earliest = datetime.now(timezone.utc) - timedelta(days=max_days)
    since = since or (datetime.now(timezone.utc) - timedelta(days=2))
    if since < earliest:
        logger.info(
            "Clamping NewsAPI lookback from %s to %s (free-tier limit %d days)",
            since.date(), earliest.date(), max_days,
        )
        since = earliest

    url = f"{NEWSAPI_BASE}/everything"
    params = {
        "q": query,
        "from": since.isoformat(),
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "language": "en",
        "apiKey": creds.newsapi,
    }
    payload = _request_with_retry("GET", url, params=params)
    articles = payload.get("articles", [])
    if not articles:
        return pd.DataFrame(
            columns=["published_at", "title", "description", "source", "url"]
        )

    df = pd.DataFrame([
        {
            "published_at": a.get("publishedAt"),
            "title": a.get("title"),
            "description": a.get("description"),
            "source": (a.get("source") or {}).get("name"),
            "url": a.get("url"),
        }
        for a in articles
    ])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df.set_index("published_at").sort_index()


def fetch_macro(series_ids: Iterable[str]) -> pd.DataFrame:
    """Fetch macro time series from FRED (e.g., DGS10, VIXCLS, UNRATE).

    Returns a wide DataFrame indexed by date with one column per series.
    """
    creds = _Creds.from_env()
    if not creds.fred:
        raise EnvironmentError("FRED_API_KEY must be set.")

    frames: List[pd.DataFrame] = []
    for sid in series_ids:
        url = f"{FRED_BASE}/series/observations"
        params = {"series_id": sid, "api_key": creds.fred, "file_type": "json"}
        payload = _request_with_retry("GET", url, params=params)
        obs = payload.get("observations", [])
        if not obs:
            logger.warning("FRED series %s returned no observations", sid)
            continue
        df = pd.DataFrame(obs)[["date", "value"]].rename(columns={"value": sid})
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df[sid] = pd.to_numeric(df[sid], errors="coerce")
        frames.append(df.set_index("date"))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index().ffill()


# ---------------------------------------------------------------------------
# Finnhub
# ---------------------------------------------------------------------------
def fetch_finnhub_news(
    symbol: str,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> pd.DataFrame:
    """Company-specific news from Finnhub.

    Returns a DataFrame indexed by ``datetime`` with columns
    ``[headline, summary, source, category, url, related]``.
    """
    creds = _Creds.from_env()
    if not creds.finnhub:
        raise EnvironmentError("FINNHUB_API_KEY must be set.")
    until = until or datetime.now(timezone.utc)
    since = since or (until - timedelta(days=14))

    url = f"{FINNHUB_BASE}/company-news"
    params = {
        "symbol": symbol,
        "from": since.strftime("%Y-%m-%d"),
        "to": until.strftime("%Y-%m-%d"),
        "token": creds.finnhub,
    }
    payload = _request_with_retry("GET", url, params=params)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(
            columns=["headline", "summary", "source", "category", "url", "related"]
        )

    df = pd.DataFrame(payload)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
    return df.set_index("datetime").sort_index()[
        ["headline", "summary", "source", "category", "url", "related"]
    ]


def fetch_finnhub_insider(symbol: str, lookback_days: int = 180) -> pd.DataFrame:
    """Recent insider transactions (Form 4 etc.) from Finnhub."""
    creds = _Creds.from_env()
    if not creds.finnhub:
        raise EnvironmentError("FINNHUB_API_KEY must be set.")

    url = f"{FINNHUB_BASE}/stock/insider-transactions"
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=lookback_days)
    params = {
        "symbol": symbol,
        "from": since.strftime("%Y-%m-%d"),
        "to": until.strftime("%Y-%m-%d"),
        "token": creds.finnhub,
    }
    payload = _request_with_retry("GET", url, params=params)
    rows = (payload or {}).get("data") or []
    if not rows:
        return pd.DataFrame(
            columns=["name", "share", "change", "transactionCode",
                     "transactionPrice", "transactionDate"]
        )

    df = pd.DataFrame(rows)
    df["transactionDate"] = pd.to_datetime(df["transactionDate"], utc=True)
    keep = [c for c in [
        "name", "share", "change", "transactionCode",
        "transactionPrice", "transactionDate",
    ] if c in df.columns]
    return df[keep].sort_values("transactionDate")


def fetch_finnhub_earnings_calendar(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> pd.DataFrame:
    """Upcoming earnings releases (whole market) within a date window."""
    creds = _Creds.from_env()
    if not creds.finnhub:
        raise EnvironmentError("FINNHUB_API_KEY must be set.")
    since = since or datetime.now(timezone.utc)
    until = until or (since + timedelta(days=14))

    url = f"{FINNHUB_BASE}/calendar/earnings"
    params = {
        "from": since.strftime("%Y-%m-%d"),
        "to": until.strftime("%Y-%m-%d"),
        "token": creds.finnhub,
    }
    payload = _request_with_retry("GET", url, params=params)
    rows = (payload or {}).get("earningsCalendar") or []
    if not rows:
        return pd.DataFrame(
            columns=["date", "symbol", "epsActual", "epsEstimate", "hour"]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    keep = [c for c in [
        "date", "symbol", "epsActual", "epsEstimate",
        "revenueActual", "revenueEstimate", "hour", "quarter", "year",
    ] if c in df.columns]
    return df[keep].sort_values("date")


# ---------------------------------------------------------------------------
# Alpha Vantage (news + sentiment)
# ---------------------------------------------------------------------------
def fetch_alpha_vantage_news(
    tickers: Iterable[str] | str,
    *,
    topics: Optional[str] = None,
    limit: int = 50,
) -> pd.DataFrame:
    """News + per-article sentiment scores from Alpha Vantage.

    Returns a DataFrame indexed by ``time_published`` (UTC) with
    columns ``[title, summary, source, url, overall_sentiment_score,
    overall_sentiment_label, ticker_sentiment]``.
    """
    creds = _Creds.from_env()
    if not creds.alpha_vantage:
        raise EnvironmentError("ALPHA_VANTAGE_API_KEY must be set.")

    if not isinstance(tickers, str):
        tickers = ",".join(tickers)

    params: Dict[str, Any] = {
        "function": "NEWS_SENTIMENT",
        "tickers": tickers,
        "limit": min(limit, 1000),
        "apikey": creds.alpha_vantage,
    }
    if topics:
        params["topics"] = topics

    payload = _request_with_retry("GET", ALPHA_VANTAGE_BASE, params=params)
    feed = payload.get("feed") or []
    # Alpha Vantage signals rate-limit hits via a "Note" / "Information" key.
    if not feed and ("Note" in payload or "Information" in payload):
        logger.warning("Alpha Vantage rate-limited: %s", payload)
        return pd.DataFrame()

    if not feed:
        return pd.DataFrame(
            columns=["title", "summary", "source", "url",
                     "overall_sentiment_score", "overall_sentiment_label",
                     "ticker_sentiment"]
        )

    rows = []
    for a in feed:
        rows.append({
            "time_published": a.get("time_published"),
            "title": a.get("title"),
            "summary": a.get("summary"),
            "source": a.get("source"),
            "url": a.get("url"),
            "overall_sentiment_score": a.get("overall_sentiment_score"),
            "overall_sentiment_label": a.get("overall_sentiment_label"),
            "ticker_sentiment": a.get("ticker_sentiment"),
        })
    df = pd.DataFrame(rows)
    # AV time format: "20260522T133512"
    df["time_published"] = pd.to_datetime(
        df["time_published"], format="%Y%m%dT%H%M%S", utc=True, errors="coerce"
    )
    df["overall_sentiment_score"] = pd.to_numeric(
        df["overall_sentiment_score"], errors="coerce"
    )
    return df.dropna(subset=["time_published"]).set_index("time_published").sort_index()


# ---------------------------------------------------------------------------
# Polygon (backup OHLCV)
# ---------------------------------------------------------------------------
def fetch_polygon_aggregates(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    timespan: str = "day",
    multiplier: int = 1,
    adjusted: bool = True,
) -> pd.DataFrame:
    """Aggregate bars from Polygon — independent backup to fetch_futures()."""
    creds = _Creds.from_env()
    if not creds.polygon:
        raise EnvironmentError("POLYGON_API_KEY must be set.")

    url = (
        f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/range/"
        f"{multiplier}/{timespan}/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    )
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": 50_000,
        "apiKey": creds.polygon,
    }
    payload = _request_with_retry("GET", url, params=params)
    results = payload.get("results") or []
    if not results:
        logger.warning("Polygon returned no bars for %s", symbol)
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "vwap", "trades"]
        )

    df = pd.DataFrame(results)
    df = df.rename(columns={
        "t": "timestamp", "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap", "n": "trades",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    keep = [c for c in
            ["open", "high", "low", "close", "volume", "vwap", "trades"]
            if c in df.columns]
    return df.set_index("timestamp").sort_index()[keep]


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------
def load_data(
    symbol: str,
    *,
    lookback_days: int = 365,
    macro_series: Optional[Iterable[str]] = None,
    news_query: Optional[str] = None,
    include_options: bool = True,
    include_finnhub: bool = True,
    include_alpha_vantage: bool = True,
    include_polygon: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Bundle every available data source into one dict of frames.

    The price frame is the canonical datetime-indexed DataFrame; the
    other frames are aligned on a best-effort basis by the caller.
    Each optional source degrades gracefully when its credentials are
    missing — a missing key produces a warning and an empty frame, not
    a hard failure.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    out: Dict[str, pd.DataFrame] = {}
    out["prices"] = fetch_futures(symbol, start, end)

    if include_options:
        try:
            out["options"] = fetch_options(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Options fetch skipped: %s", exc)
            out["options"] = pd.DataFrame()

    # NewsAPI generic search.
    try:
        out["news"] = fetch_news(news_query or symbol, since=start)
    except Exception as exc:  # noqa: BLE001
        logger.warning("News fetch skipped: %s", exc)
        out["news"] = pd.DataFrame()

    # FRED macro series.
    try:
        out["macro"] = fetch_macro(macro_series or ("DGS10", "VIXCLS"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Macro fetch skipped: %s", exc)
        out["macro"] = pd.DataFrame()

    # Finnhub: company news, insider transactions, earnings calendar.
    if include_finnhub:
        try:
            out["finnhub_news"] = fetch_finnhub_news(symbol, since=start)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Finnhub news skipped: %s", exc)
            out["finnhub_news"] = pd.DataFrame()
        try:
            out["finnhub_insider"] = fetch_finnhub_insider(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Finnhub insider skipped: %s", exc)
            out["finnhub_insider"] = pd.DataFrame()
        try:
            out["finnhub_earnings"] = fetch_finnhub_earnings_calendar()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Finnhub earnings skipped: %s", exc)
            out["finnhub_earnings"] = pd.DataFrame()

    # Alpha Vantage news + sentiment scores.
    if include_alpha_vantage:
        try:
            out["av_news"] = fetch_alpha_vantage_news(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alpha Vantage news skipped: %s", exc)
            out["av_news"] = pd.DataFrame()

    # Polygon backup OHLCV (off by default to avoid rate-limit thrash).
    if include_polygon:
        try:
            out["polygon_prices"] = fetch_polygon_aggregates(symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Polygon backup OHLCV skipped: %s", exc)
            out["polygon_prices"] = pd.DataFrame()

    return out


if __name__ == "__main__":  # pragma: no cover — smoke run
    from pprint import pprint
    pprint({k: v.shape for k, v in load_data("SPY", lookback_days=30).items()})
