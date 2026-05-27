"""Free, key-free alternatives for the paid/limited data sources.

Two drop-in replacements:

* :func:`fetch_options_free` replaces :func:`src.data_loader.fetch_options`
  (Tradier). Pulls live option chains via ``yfinance`` and computes
  Greeks locally with Black-Scholes — no API key, no rate limit, no
  vendor sign-up.

* :func:`fetch_news_with_sentiment` replaces
  :func:`src.data_loader.fetch_alpha_vantage_news`. Pulls company news
  from Finnhub (already wired) and scores each headline locally with
  the FinBERT encoder in :mod:`src.model.sentiment_encoder`. Result has
  identical schema to the Alpha Vantage fetcher so downstream code can
  swap one for the other.

Usage::

    from src.data_alternatives import fetch_options_free, fetch_news_with_sentiment

    chain = fetch_options_free("SPY")                    # all expiries, with Greeks
    chain = fetch_options_free("SPY", expiry="2026-06-20")
    news  = fetch_news_with_sentiment("AAPL", score=True)

Both functions return ``pandas.DataFrame`` objects ready to feed
into the existing ensemble + risk pipeline.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------
def _bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> dict:
    """Closed-form Black-Scholes greeks.

    Returns ``{delta, gamma, theta, vega, rho}``. Theta is per *calendar
    day*; vega and rho are per 1-percentage-point change.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": np.nan, "gamma": np.nan,
                "theta": np.nan, "vega": np.nan, "rho": np.nan}

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * sqrtT)
                 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100.0
    elif option_type == "put":
        delta = norm.cdf(d1) - 1.0
        theta = (-S * pdf_d1 * sigma / (2 * sqrtT)
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    gamma = pdf_d1 / (S * sigma * sqrtT)
    vega = S * pdf_d1 * sqrtT / 100.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


# ---------------------------------------------------------------------------
# Options replacement — yfinance + Black-Scholes
# ---------------------------------------------------------------------------
def fetch_options_free(
    underlying: str,
    *,
    expiry: Optional[str] = None,
    risk_free_rate: float = 0.05,
    compute_greeks: bool = True,
    ticker_factory=None,
) -> pd.DataFrame:
    """Drop-in replacement for the Tradier options fetcher.

    Parameters
    ----------
    underlying:
        Stock / ETF symbol (e.g. ``"SPY"``).
    expiry:
        Specific expiration in ``YYYY-MM-DD`` form. If omitted, every
        listed expiry is pulled and concatenated.
    risk_free_rate:
        Annualized rate used for the Greeks. 5% is a reasonable default
        for current US conditions; pull from FRED ``DGS3MO`` for more
        precision.
    compute_greeks:
        Set False if you only need bid/ask/IV/volume and want to skip
        the Black-Scholes computation entirely.
    ticker_factory:
        Internal hook for tests — pass a callable that returns a
        ``yfinance.Ticker``-like object. Production callers should leave
        this at the default.

    Returns
    -------
    pandas.DataFrame
        Columns: ``symbol, strike, type, expiry, bid, ask, last,
        volume, open_interest, iv, delta, gamma, theta, vega, rho``.
        Indexed by an integer range.
    """
    if ticker_factory is None:
        import yfinance as yf  # lazy import — only need it if no test stub
        ticker_factory = yf.Ticker

    t = ticker_factory(underlying)

    # Spot price — used for Greeks and log-moneyness if downstream needs it.
    try:
        spot = float(t.fast_info["last_price"])
    except Exception:  # noqa: BLE001
        try:
            spot = float(t.history(period="1d")["Close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"yfinance: cannot fetch spot for {underlying}") from exc

    expiries = [expiry] if expiry else list(t.options or [])
    if not expiries:
        logger.warning("yfinance returned no expirations for %s", underlying)
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    frames = []
    for exp in expiries:
        try:
            chain = t.option_chain(exp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance: skipping expiry %s for %s (%s)", exp, underlying, exc)
            continue
        for side, df_side in (("call", chain.calls), ("put", chain.puts)):
            if df_side is None or df_side.empty:
                continue
            df = df_side.copy()
            df["type"] = side
            df["expiry"] = exp
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # Normalise to the Tradier-style schema.
    out = pd.DataFrame({
        "symbol": raw.get("contractSymbol"),
        "strike": pd.to_numeric(raw["strike"], errors="coerce"),
        "type": raw["type"],
        "expiry": pd.to_datetime(raw["expiry"], utc=True),
        "bid": pd.to_numeric(raw.get("bid"), errors="coerce"),
        "ask": pd.to_numeric(raw.get("ask"), errors="coerce"),
        "last": pd.to_numeric(raw.get("lastPrice"), errors="coerce"),
        "volume": pd.to_numeric(raw.get("volume"), errors="coerce"),
        "open_interest": pd.to_numeric(raw.get("openInterest"), errors="coerce"),
        "iv": pd.to_numeric(raw.get("impliedVolatility"), errors="coerce"),
    })

    if compute_greeks:
        deltas, gammas, thetas, vegas, rhos = [], [], [], [], []
        for _, row in out.iterrows():
            T_years = max((row["expiry"] - now).total_seconds() / (365.0 * 86400.0), 1e-6)
            sigma = float(row["iv"]) if not pd.isna(row["iv"]) else float("nan")
            try:
                g = _bs_greeks(spot, float(row["strike"]), T_years,
                               risk_free_rate, sigma, row["type"])
            except Exception:  # noqa: BLE001
                g = {"delta": np.nan, "gamma": np.nan, "theta": np.nan,
                     "vega": np.nan, "rho": np.nan}
            deltas.append(g["delta"])
            gammas.append(g["gamma"])
            thetas.append(g["theta"])
            vegas.append(g["vega"])
            rhos.append(g["rho"])
        out["delta"] = deltas
        out["gamma"] = gammas
        out["theta"] = thetas
        out["vega"] = vegas
        out["rho"] = rhos

    out.attrs["spot"] = spot
    return out


# ---------------------------------------------------------------------------
# News + sentiment replacement — Finnhub + FinBERT
# ---------------------------------------------------------------------------
_cached_encoder = None


def _get_encoder():
    """Lazy singleton — only load FinBERT when first asked."""
    global _cached_encoder
    if _cached_encoder is None:
        from src.model.sentiment_encoder import SentimentEncoder
        _cached_encoder = SentimentEncoder()
    return _cached_encoder


def fetch_news_with_sentiment(
    symbol: str,
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    score: bool = True,
    encoder=None,
    news_fn=None,
) -> pd.DataFrame:
    """Drop-in replacement for the Alpha Vantage news+sentiment fetcher.

    Parameters
    ----------
    symbol:
        Stock / ETF ticker.
    since, until:
        Optional UTC datetimes; default is the last 14 days.
    score:
        If True, run each headline+summary through FinBERT and add
        ``overall_sentiment_score`` / ``overall_sentiment_label``
        columns. If False, return raw news rows.
    encoder, news_fn:
        Injection hooks for tests; production callers leave these alone.

    Returns
    -------
    pandas.DataFrame
        Same schema as :func:`src.data_loader.fetch_alpha_vantage_news`
        when ``score=True``: indexed by ``time_published`` (UTC),
        columns ``[title, summary, source, url,
        overall_sentiment_score, overall_sentiment_label]``.
    """
    if news_fn is None:
        from src.data_loader import fetch_finnhub_news
        news_fn = fetch_finnhub_news

    until = until or datetime.now(timezone.utc)
    since = since or (until - timedelta(days=14))

    raw = news_fn(symbol, since=since, until=until)
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=["title", "summary", "source", "url",
                     "overall_sentiment_score", "overall_sentiment_label"]
        )

    out = pd.DataFrame({
        "title": raw["headline"] if "headline" in raw.columns else raw.get("title"),
        "summary": raw.get("summary"),
        "source": raw.get("source"),
        "url": raw.get("url"),
    }, index=raw.index)
    out.index.name = "time_published"

    if not score:
        return out

    enc = encoder or _get_encoder()
    texts = (out["title"].fillna("") + ". " + out["summary"].fillna("")).tolist()
    scores: list[float] = []
    labels: list[str] = []
    # Score in modest batches to keep memory bounded.
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        for text in batch:
            result = enc.score_news(text)
            s = float(result.score)
            scores.append(s)
            labels.append(_score_to_label(s))
    out["overall_sentiment_score"] = scores
    out["overall_sentiment_label"] = labels
    return out


def _score_to_label(score: float) -> str:
    """Bucketize a sentiment score in [-1, 1] into the AV-style labels."""
    if score >= 0.35:
        return "Bullish"
    if score >= 0.15:
        return "Somewhat-Bullish"
    if score <= -0.35:
        return "Bearish"
    if score <= -0.15:
        return "Somewhat-Bearish"
    return "Neutral"
