"""Pure-web data scraping agent — no secondary API keys required.

Hits public endpoints directly (Yahoo Finance JSON, SEC EDGAR, Treasury,
CFTC, news RSS) and synthesises the same bundle that ``load_data()``
would produce from paid vendors. Sentiment scoring runs locally via
FinBERT so even that doesn't require an API key.

Each method is a thin scraper that can be called independently; the
:meth:`WebDataScraper.scrape_all` convenience method wraps them into a
single dict and gracefully degrades on per-source failures.

What this replaces:
    Tradier       → :meth:`scrape_options`        (Yahoo JSON)
    Alpha Vantage → :meth:`scrape_news_rss` + local FinBERT
    NewsAPI       → :meth:`scrape_news_rss` + :meth:`scrape_google_news`
    FRED          → :meth:`scrape_treasury_yields` (subset)
    Finnhub       → :meth:`scrape_news_rss` + :meth:`scrape_insider_form4`

What this adds beyond the paid APIs:
    Reddit (WSB / stocks / options / investing) — :meth:`scrape_reddit_mentions`
    Congressional trades (House + Senate STOCK Act) — :meth:`scrape_congress_trades`
    StockTwits public stream — :meth:`scrape_stocktwits`
    Google News cross-publisher RSS — :meth:`scrape_google_news`
    Hacker News (tech/finance crossover) — :meth:`scrape_hacker_news`
    Wikipedia pageviews (retail interest proxy) — :meth:`scrape_wikipedia_pageviews`

What this does *not* replace:
    Alpaca        — execution still needs broker credentials.

⚠️  Caveats:
    * Scrapers break when source HTML/JSON changes. Expect 1–3 fix
      cycles per year per source.
    * Be polite. The defaults below respect typical site etiquette
      (real User-Agent, short delays between calls, no parallel hammering).
    * Some sources (Yahoo) lightly throttle aggressive scraping —
      keep total request rate under ~30/min per IP.
"""

from __future__ import annotations

import io
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default endpoints — all public, no auth required.
# ---------------------------------------------------------------------------
YAHOO_OPTIONS_URL = "https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"

# RSS feeds — public, no key required.
# Reuters' free RSS was retired in 2024; their content is reached via
# Google News (see scrape_google_news) instead.
DEFAULT_RSS_FEEDS = [
    # Yahoo Finance per-symbol RSS (use {symbol} placeholder).
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
    # MarketWatch top stories.
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    # CNBC top news.
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    # Investing.com headlines.
    "https://www.investing.com/rss/news.rss",
    # NASDAQ news.
    "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
]

# SEC EDGAR — REQUIRES a polite User-Agent identifying you.
EDGAR_SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_TICKER_LOOKUP = "https://www.sec.gov/files/company_tickers.json"

# Treasury Direct (yield curve, auctions).
TREASURY_YIELD_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/{year}/all"
)

# CFTC Commitments of Traders — weekly futures positioning.
CFTC_COT_FUTURES_URL = (
    "https://www.cftc.gov/dea/newcot/deafut.txt"
)

# FINRA short interest (bi-weekly CSV by date).
FINRA_SHORT_INTEREST_URL = (
    "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
)

# ---------------------------------------------------------------------------
# Crowd / political / cross-publisher sources
# ---------------------------------------------------------------------------
# Reddit public JSON (search a subreddit for a symbol).
REDDIT_SEARCH_URL = "https://www.reddit.com/r/{sub}/search.json"
DEFAULT_REDDIT_SUBS = ("wallstreetbets", "stocks", "options", "investing", "StockMarket")

# senate-stock-watcher / house-stock-watcher publish their data as
# public S3 JSON dumps (no auth). These are large files; cache them.
SENATE_TRADES_URL = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com"
    "/aggregate/all_transactions.json"
)
HOUSE_TRADES_URL = (
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com"
    "/data/all_transactions.json"
)

# Google News RSS — query-based, aggregates across publishers.
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

# StockTwits public stream (JSON; auth-free for public symbols).
STOCKTWITS_SYMBOL_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

# Hacker News search RSS via hnrss.org.
HN_SEARCH_RSS = "https://hnrss.org/newest"

# Wikipedia pageviews (Wikimedia REST API — public, no key).
WIKIPEDIA_PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{title}/daily/{start}/{end}"
)

DEFAULT_USER_AGENT = (
    "trading_enhancer/0.1 (research scraper; contact: you@example.com)"
)
DEFAULT_TIMEOUT = 10
INTER_REQUEST_DELAY_S = 0.5


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
@dataclass
class WebDataScraper:
    """One agent, many public sources, zero secondary API keys."""

    user_agent: str = DEFAULT_USER_AGENT
    timeout: int = DEFAULT_TIMEOUT
    delay_s: float = INTER_REQUEST_DELAY_S
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.session.headers.update({"User-Agent": self.user_agent})

    # ------------------------------------------------------------------
    # Low-level helper
    # ------------------------------------------------------------------
    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        time.sleep(self.delay_s)
        resp = self.session.get(url, timeout=self.timeout, **kwargs)
        if resp.status_code == 429:
            logger.warning("Throttled by %s — sleeping 5s then retrying once", url)
            time.sleep(5)
            resp = self.session.get(url, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp

    # ==================================================================
    # 1. OHLCV  (Yahoo chart endpoint — no key)
    # ==================================================================
    def scrape_ohlcv(
        self,
        symbol: str,
        *,
        range_: str = "2y",
        interval: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """OHLCV bars from Yahoo's public chart endpoint.

        ``range_`` accepts ``1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max``.
        ``interval`` accepts ``1m, 5m, 15m, 1h, 1d, 1wk, 1mo``.

        If ``start`` and ``end`` are provided (UTC datetimes), they
        override ``range_`` and pull a specific historical window —
        used by the scenario backtester for replaying named historical
        episodes (COVID crash, Volmageddon, GameStop squeeze, etc.).
        """
        url = YAHOO_CHART_URL.format(symbol=symbol)
        if start is not None and end is not None:
            params = {
                "period1": int(start.timestamp()),
                "period2": int(end.timestamp()),
                "interval": interval,
                "includePrePost": "false",
            }
        else:
            params = {"range": range_, "interval": interval,
                      "includePrePost": "false"}
        payload = self._get(url, params=params).json()
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        r = result[0]
        ts = r.get("timestamp") or []
        quote = (r.get("indicators") or {}).get("quote", [{}])[0]
        df = pd.DataFrame({
            "open":   quote.get("open"),
            "high":   quote.get("high"),
            "low":    quote.get("low"),
            "close":  quote.get("close"),
            "volume": quote.get("volume"),
        }, index=pd.to_datetime(ts, unit="s", utc=True))
        df.index.name = "timestamp"
        return df.dropna(how="all")

    # ==================================================================
    # 2. Options  (Yahoo options endpoint — no key)
    # ==================================================================
    def scrape_options(
        self,
        symbol: str,
        *,
        expiry: Optional[str] = None,
    ) -> pd.DataFrame:
        """Options chain via Yahoo Finance's public JSON endpoint, with
        an automatic yfinance fallback when Yahoo demands auth (which
        they started doing in 2024 on some IPs).
        """
        url = YAHOO_OPTIONS_URL.format(symbol=symbol)
        params: Dict[str, Any] = {}
        if expiry:
            params["date"] = int(pd.Timestamp(expiry, tz="UTC").timestamp())

        try:
            payload = self._get(url, params=params).json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                logger.info(
                    "Yahoo /v7/finance/options returned %s — falling back to yfinance",
                    exc.response.status_code,
                )
                return self._scrape_options_via_yfinance(symbol, expiry=expiry)
            raise

        result = (payload.get("optionChain") or {}).get("result") or []
        if not result:
            # Empty result is sometimes a soft block. Try yfinance once.
            return self._scrape_options_via_yfinance(symbol, expiry=expiry)

        node = result[0]
        spot = (node.get("quote") or {}).get("regularMarketPrice")
        rows: List[Dict[str, Any]] = []
        for chain in node.get("options") or []:
            exp_unix = chain.get("expirationDate")
            exp_dt = pd.to_datetime(exp_unix, unit="s", utc=True) if exp_unix else None
            for side in ("calls", "puts"):
                for o in chain.get(side) or []:
                    rows.append({
                        "symbol": o.get("contractSymbol"),
                        "type": side[:-1],   # 'call' / 'put'
                        "strike": o.get("strike"),
                        "expiry": exp_dt,
                        "bid": o.get("bid"),
                        "ask": o.get("ask"),
                        "last": o.get("lastPrice"),
                        "volume": o.get("volume"),
                        "open_interest": o.get("openInterest"),
                        "iv": o.get("impliedVolatility"),
                    })
        df = pd.DataFrame(rows)
        df.attrs["spot"] = spot
        return df

    @staticmethod
    def _scrape_options_via_yfinance(
        symbol: str,
        *,
        expiry: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fallback using the yfinance library, which handles Yahoo's
        cookie/crumb auth flow internally."""
        try:
            from src.data_alternatives import fetch_options_free
        except ImportError as exc:
            logger.warning("yfinance fallback unavailable: %s", exc)
            return pd.DataFrame()
        try:
            return fetch_options_free(symbol, expiry=expiry, compute_greeks=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance options fallback failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    # ==================================================================
    # 3. News  (multi-source RSS — no key)
    # ==================================================================
    def scrape_news_rss(
        self,
        symbol: str,
        *,
        feeds: Optional[Iterable[str]] = None,
        max_items: int = 200,
    ) -> pd.DataFrame:
        """Aggregate news headlines from public RSS feeds.

        Per-symbol feeds (Yahoo) get the symbol substituted into the
        URL; market-wide feeds (MarketWatch, Reuters, CNBC) are pulled
        as-is and filtered post-hoc for mentions of the symbol's
        ticker or company name.
        """
        feeds = list(feeds or DEFAULT_RSS_FEEDS)
        rows: List[Dict[str, Any]] = []

        for feed_tmpl in feeds:
            url = feed_tmpl.format(symbol=symbol) if "{symbol}" in feed_tmpl else feed_tmpl
            try:
                xml = self._get(url).text
            except Exception as exc:  # noqa: BLE001
                logger.warning("RSS fetch failed (%s): %s", url, exc)
                continue
            rows.extend(self._parse_rss(xml, source=url))

        if not rows:
            return pd.DataFrame(
                columns=["title", "summary", "source", "url", "published_at"]
            )

        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = (df.dropna(subset=["published_at"])
                .drop_duplicates(subset=["title"])
                .sort_values("published_at", ascending=False)
                .head(max_items))
        return df.set_index("published_at")

    @staticmethod
    def _parse_rss(xml_text: str, source: str) -> List[Dict[str, Any]]:
        """Permissive RSS/Atom parser (handles both common formats)."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        items: List[Dict[str, Any]] = []
        # RSS 2.0 — <channel><item>
        for item in root.iter("item"):
            items.append({
                "title": _txt(item, "title"),
                "summary": _txt(item, "description"),
                "url": _txt(item, "link"),
                "published_at": _txt(item, "pubDate"),
                "source": source,
            })
        # Atom — <entry>
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            link_el = entry.find("{http://www.w3.org/2005/Atom}link")
            items.append({
                "title": _txt(entry, "{http://www.w3.org/2005/Atom}title"),
                "summary": _txt(entry, "{http://www.w3.org/2005/Atom}summary"),
                "url": link_el.get("href") if link_el is not None else None,
                "published_at": _txt(entry, "{http://www.w3.org/2005/Atom}updated"),
                "source": source,
            })
        return items

    # ==================================================================
    # 4. Insider trading  (SEC EDGAR — no key, polite UA only)
    # ==================================================================
    def scrape_insider_form4(self, symbol: str, count: int = 40) -> pd.DataFrame:
        """Recent Form 4 insider filings for `symbol` via SEC EDGAR.

        Returns a DataFrame of filing metadata (filing date, form,
        accession number, URL). Parsing the XBRL payloads inside each
        Form 4 to extract exact share counts is left to a downstream
        step — the URL column gives you everything you need.
        """
        cik = self._lookup_cik(symbol)
        if not cik:
            return pd.DataFrame()
        params = {
            "action": "getcompany",
            "CIK": f"{int(cik):010d}",
            "type": "4",
            "dateb": "",
            "owner": "include",
            "count": count,
            "output": "atom",
        }
        url = f"{EDGAR_SEARCH_URL}?{urlencode(params)}"
        xml = self._get(url).text

        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return pd.DataFrame()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        rows = []
        for entry in root.findall("atom:entry", ns):
            updated = entry.find("atom:updated", ns)
            title = entry.find("atom:title", ns)
            link = entry.find("atom:link", ns)
            content = entry.find("atom:content", ns)
            # Extract accession number from the content blob.
            acc_match = re.search(r"Accession Number:\s*([\d-]+)",
                                  content.text if content is not None else "") \
                if content is not None else None
            rows.append({
                "filed_at": updated.text if updated is not None else None,
                "title": title.text if title is not None else None,
                "url": link.get("href") if link is not None else None,
                "accession": acc_match.group(1) if acc_match else None,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["filed_at"] = pd.to_datetime(df["filed_at"], utc=True, errors="coerce")
        return df.dropna(subset=["filed_at"]).set_index("filed_at").sort_index()

    def _lookup_cik(self, ticker: str) -> Optional[str]:
        """Map a ticker → SEC CIK using EDGAR's master ticker file."""
        try:
            payload = self._get(EDGAR_TICKER_LOOKUP).json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("EDGAR ticker lookup failed: %s", exc)
            return None
        target = ticker.upper()
        for entry in payload.values():
            if entry.get("ticker", "").upper() == target:
                return str(entry["cik_str"])
        return None

    # ==================================================================
    # 5. Treasury yield curve  (Treasury Direct CSV — no key)
    # ==================================================================
    def scrape_treasury_yields(self, year: Optional[int] = None) -> pd.DataFrame:
        """Daily Treasury yield curve rates (1mo–30y)."""
        year = year or datetime.now(timezone.utc).year
        url = TREASURY_YIELD_URL.format(year=year)
        # Treasury serves CSV with a Q4 calendar-year filter query.
        params = {"type": "daily_treasury_yield_curve",
                  "field_tdr_date_value": str(year)}
        text = self._get(url, params=params).text
        df = pd.read_csv(io.StringIO(text))
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
            df = df.set_index("Date").sort_index()
        return df

    # ==================================================================
    # 6. CFTC Commitments of Traders  (weekly text file — no key)
    # ==================================================================
    def scrape_cot(self) -> pd.DataFrame:
        """Latest CFTC futures CoT report (raw fixed-width text)."""
        text = self._get(CFTC_COT_FUTURES_URL).text
        # The file is fixed-width; we expose it as a single column DataFrame
        # so the user can post-process for whatever contracts they want.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return pd.DataFrame({"raw": lines})

    # ==================================================================
    # 7. FINRA short interest  (daily CSV — no key)
    # ==================================================================
    def scrape_short_volume(self, date: Optional[datetime] = None) -> pd.DataFrame:
        """FINRA daily short-sale volume for the consolidated NMS feed.

        `date` defaults to the previous business day.
        """
        d = date or (datetime.now(timezone.utc) - timedelta(days=1))
        url = FINRA_SHORT_INTEREST_URL.format(date=d.strftime("%Y%m%d"))
        try:
            text = self._get(url).text
        except requests.HTTPError as exc:
            logger.warning("FINRA short volume not yet posted for %s: %s",
                           d.date(), exc)
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(text), sep="|")
        return df

    # ==================================================================
    # 8. Reddit mentions  (public reddit.com JSON — polite UA only)
    # ==================================================================
    def scrape_reddit_mentions(
        self,
        symbol: str,
        *,
        subreddits: Optional[Iterable[str]] = None,
        time_filter: str = "week",
        limit: int = 50,
    ) -> pd.DataFrame:
        """Search retail subreddits for posts mentioning a ticker.

        ``time_filter`` accepts ``hour, day, week, month, year, all``.

        Returns columns ``[subreddit, title, selftext, score,
        num_comments, url, author]`` indexed by ``created_at`` (UTC).
        Useful as a crowd-sentiment / attention feature.
        """
        subs = list(subreddits or DEFAULT_REDDIT_SUBS)
        rows: List[Dict[str, Any]] = []
        for sub in subs:
            url = REDDIT_SEARCH_URL.format(sub=sub)
            params = {
                "q": symbol,
                "restrict_sr": "1",
                "sort": "new",
                "t": time_filter,
                "limit": min(limit, 100),
            }
            try:
                payload = self._get(url, params=params).json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reddit fetch failed (r/%s): %s", sub, exc)
                continue
            for child in (payload.get("data") or {}).get("children", []):
                d = child.get("data") or {}
                rows.append({
                    "subreddit": d.get("subreddit"),
                    "title": d.get("title"),
                    "selftext": d.get("selftext"),
                    "score": d.get("score"),
                    "num_comments": d.get("num_comments"),
                    "url": d.get("url"),
                    "author": d.get("author"),
                    "created_at": d.get("created_utc"),
                })

        if not rows:
            return pd.DataFrame(
                columns=["subreddit", "title", "selftext", "score",
                         "num_comments", "url", "author"]
            )
        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s", utc=True)
        return (df.dropna(subset=["created_at"])
                  .drop_duplicates(subset=["url"])
                  .sort_values("created_at", ascending=False)
                  .set_index("created_at"))

    # ==================================================================
    # 9. Congressional trades  (community S3 dumps — no key)
    # ==================================================================
    def scrape_congress_trades(
        self,
        symbol: Optional[str] = None,
        *,
        lookback_days: int = 60,
        chambers: Iterable[str] = ("senate", "house"),
    ) -> pd.DataFrame:
        """Combined House + Senate STOCK Act disclosures.

        Pulls the public S3 JSON dumps maintained by the
        senate-stock-watcher / house-stock-watcher communities.
        Filters to the last `lookback_days` and (if `symbol` is given)
        to a specific ticker.

        Returns columns ``[chamber, representative, ticker, type,
        amount, transaction_date, disclosure_date]``.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        rows: List[Dict[str, Any]] = []
        url_map = {
            "senate": SENATE_TRADES_URL,
            "house": HOUSE_TRADES_URL,
        }
        for chamber in chambers:
            url = url_map.get(chamber)
            if not url:
                continue
            try:
                payload = self._get(url).json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Congress trades fetch failed (%s): %s", chamber, exc)
                continue
            for tx in payload or []:
                rows.append({
                    "chamber": chamber,
                    "representative": tx.get("senator") or tx.get("representative"),
                    "ticker": (tx.get("ticker") or "").upper().strip("$ -"),
                    "type": tx.get("type"),
                    "amount": tx.get("amount"),
                    "transaction_date": tx.get("transaction_date"),
                    "disclosure_date": tx.get("disclosure_date"),
                    "asset_description": tx.get("asset_description"),
                })

        if not rows:
            return pd.DataFrame(
                columns=["chamber", "representative", "ticker", "type",
                         "amount", "transaction_date", "disclosure_date"]
            )

        df = pd.DataFrame(rows)
        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"], utc=True, errors="coerce"
        )
        df["disclosure_date"] = pd.to_datetime(
            df["disclosure_date"], utc=True, errors="coerce"
        )
        df = df.dropna(subset=["transaction_date"])
        df = df[df["transaction_date"] >= cutoff]
        if symbol:
            df = df[df["ticker"] == symbol.upper()]
        return df.sort_values("transaction_date", ascending=False).reset_index(drop=True)

    # ==================================================================
    # 10. Google News RSS  (cross-publisher aggregator — no key)
    # ==================================================================
    def scrape_google_news(
        self,
        query: str,
        *,
        max_items: int = 100,
        lang: str = "en-US",
        country: str = "US",
    ) -> pd.DataFrame:
        """News headlines via Google News RSS — aggregates Bloomberg,
        Reuters, WSJ, FT, etc. into one feed.

        Returns the same schema as :meth:`scrape_news_rss`.
        """
        params = {
            "q": query,
            "hl": lang,
            "gl": country,
            "ceid": f"{country}:{lang.split('-')[0]}",
        }
        try:
            xml = self._get(GOOGLE_NEWS_RSS, params=params).text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Google News fetch failed: %s", exc)
            return pd.DataFrame()

        rows = self._parse_rss(xml, source="google_news")
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        return (df.dropna(subset=["published_at"])
                  .drop_duplicates(subset=["title"])
                  .sort_values("published_at", ascending=False)
                  .head(max_items)
                  .set_index("published_at"))

    # ==================================================================
    # 11. StockTwits  (public symbol stream — no key)
    # ==================================================================
    def scrape_stocktwits(self, symbol: str, *, limit: int = 30) -> pd.DataFrame:
        """Most recent public StockTwits messages for a symbol.

        Returns ``[user, body, sentiment, likes, url]`` indexed by
        ``created_at`` (UTC). ``sentiment`` is StockTwits' user-tagged
        Bullish/Bearish field when present.
        """
        url = STOCKTWITS_SYMBOL_URL.format(symbol=symbol.upper())
        try:
            payload = self._get(url).json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("StockTwits fetch failed: %s", exc)
            return pd.DataFrame()

        messages = payload.get("messages") or []
        rows = []
        for m in messages[:limit]:
            entities = m.get("entities") or {}
            sentiment = (entities.get("sentiment") or {}).get("basic")
            rows.append({
                "user": (m.get("user") or {}).get("username"),
                "body": m.get("body"),
                "sentiment": sentiment,
                "likes": (m.get("likes") or {}).get("total"),
                "url": f"https://stocktwits.com/{(m.get('user') or {}).get('username')}/message/{m.get('id')}",
                "created_at": m.get("created_at"),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        return df.dropna(subset=["created_at"]).set_index("created_at").sort_index()

    # ==================================================================
    # 12. Hacker News  (search RSS via hnrss.org — no key)
    # ==================================================================
    def scrape_hacker_news(self, query: str, *, max_items: int = 50) -> pd.DataFrame:
        """HN posts/comments mentioning `query`.

        HN is heavily tech-tilted but is the best free signal for
        company-specific product/engineering chatter — useful for
        large-cap tech (NVDA, AAPL, GOOG, MSFT, TSLA, etc.).
        """
        params = {"q": query}
        try:
            xml = self._get(HN_SEARCH_RSS, params=params).text
        except Exception as exc:  # noqa: BLE001
            logger.warning("HN fetch failed: %s", exc)
            return pd.DataFrame()
        rows = self._parse_rss(xml, source="hacker_news")
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        return (df.dropna(subset=["published_at"])
                  .sort_values("published_at", ascending=False)
                  .head(max_items)
                  .set_index("published_at"))

    # ==================================================================
    # 13. Wikipedia pageviews  (Wikimedia REST API — no key)
    # ==================================================================
    def scrape_wikipedia_pageviews(
        self,
        title: str,
        *,
        days: int = 30,
    ) -> pd.DataFrame:
        """Daily Wikipedia pageviews — a clean retail-attention proxy.

        Pass the article title (e.g. ``"Apple_Inc."``, ``"GameStop"``)
        — not the ticker. Spaces become underscores; punctuation must
        match the actual article URL.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        url = WIKIPEDIA_PAGEVIEWS_URL.format(
            title=title.replace(" ", "_"),
            start=start.strftime("%Y%m%d"),
            end=end.strftime("%Y%m%d"),
        )
        try:
            payload = self._get(url).json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wikipedia pageviews fetch failed (%s): %s", title, exc)
            return pd.DataFrame()
        items = payload.get("items") or []
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)[["timestamp", "views"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d", utc=True)
        return df.set_index("timestamp").sort_index()

    # ==================================================================
    # 14. Sentiment scoring  (FinBERT — local, no key)
    # ==================================================================
    def score_sentiment(
        self,
        news: pd.DataFrame,
        *,
        encoder=None,
    ) -> pd.DataFrame:
        """Score a DataFrame of news with FinBERT, adding
        ``sentiment_score`` and ``sentiment_label`` columns.
        """
        if news.empty:
            return news
        if encoder is None:
            from src.model.sentiment_encoder import SentimentEncoder
            encoder = SentimentEncoder()

        texts = (news.get("title", "").fillna("") + ". " +
                 news.get("summary", "").fillna("")).tolist()
        scores, labels = [], []
        for t in texts:
            r = encoder.score_news(t)
            s = float(r.score)
            scores.append(s)
            labels.append(_score_to_label(s))
        out = news.copy()
        out["sentiment_score"] = scores
        out["sentiment_label"] = labels
        return out

    # ==================================================================
    # Unified entry point
    # ==================================================================
    def scrape_all(
        self,
        symbol: str,
        *,
        score: bool = False,
        extras: bool = True,
        wikipedia_title: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """One-shot pull of every source. Per-source failures are logged
        and replaced with empty frames so a single broken scraper never
        kills the whole run.

        Parameters
        ----------
        symbol:
            Stock / ETF ticker (passed to every per-symbol method).
        score:
            If True, run FinBERT over the news + reddit + stocktwits
            text and add sentiment columns.
        extras:
            If True (default), include the heavier sources: Reddit,
            congressional trades (multi-MB S3 dumps), Google News,
            StockTwits, Hacker News, Wikipedia pageviews.
        wikipedia_title:
            Article title for the Wikipedia pageviews lookup. If
            omitted, the lookup is skipped (ticker → title mapping is
            not 1:1, e.g. AAPL → ``Apple_Inc.``).
        """
        out: Dict[str, pd.DataFrame] = {}

        def _try(name: str, fn):
            try:
                out[name] = fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("scrape_all: %s failed: %s", name, exc)
                out[name] = pd.DataFrame()

        # Core sources.
        _try("prices",    lambda: self.scrape_ohlcv(symbol))
        _try("options",   lambda: self.scrape_options(symbol))
        _try("news",      lambda: self.scrape_news_rss(symbol))
        _try("insider",   lambda: self.scrape_insider_form4(symbol))
        _try("treasury",  lambda: self.scrape_treasury_yields())
        _try("cot",       lambda: self.scrape_cot())
        _try("short_vol", lambda: self.scrape_short_volume())

        # Heavier / crowd-and-political sources.
        # NOTE: StockTwits' free unauthenticated API now returns 403
        # for most IPs (changed mid-2024). Not included by default;
        # call scrape_stocktwits() directly if you have a workaround.
        if extras:
            _try("reddit",      lambda: self.scrape_reddit_mentions(symbol))
            _try("congress",    lambda: self.scrape_congress_trades(symbol=symbol))
            _try("google_news", lambda: self.scrape_google_news(symbol))
            _try("hacker_news", lambda: self.scrape_hacker_news(symbol))
            if wikipedia_title:
                _try("wiki_views", lambda: self.scrape_wikipedia_pageviews(wikipedia_title))

        if score:
            for key in ("news", "google_news", "reddit", "stocktwits", "hacker_news"):
                df = out.get(key)
                if df is None or df.empty:
                    continue
                try:
                    out[key] = self.score_sentiment(_normalise_text_columns(df, key))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Sentiment scoring failed for %s: %s", key, exc)

        return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _txt(elem, tag) -> Optional[str]:
    found = elem.find(tag)
    return found.text if found is not None else None


def _normalise_text_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map each source's native columns onto a uniform `title`/`summary`
    pair so :meth:`WebDataScraper.score_sentiment` can chew on any of
    them uniformly.
    """
    out = df.copy()
    if source == "reddit":
        out["title"] = out.get("title", "")
        out["summary"] = out.get("selftext", "")
    elif source == "stocktwits":
        out["title"] = out.get("body", "")
        out["summary"] = ""
    elif source == "hacker_news":
        out["title"] = out.get("title", "")
        out["summary"] = out.get("summary", "")
    elif source == "google_news":
        out["title"] = out.get("title", "")
        out["summary"] = out.get("summary", "")
    # "news" already has both columns.
    return out


def _score_to_label(s: float) -> str:
    if s >= 0.35: return "Bullish"
    if s >= 0.15: return "Somewhat-Bullish"
    if s <= -0.35: return "Bearish"
    if s <= -0.15: return "Somewhat-Bearish"
    return "Neutral"
