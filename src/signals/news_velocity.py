"""
NewsVelocity — Yahoo Finance news volume + keyword sentiment.

Measures:
  - Article count in last 24 h vs 7-day rolling average (velocity)
  - Keyword sentiment: positive vs negative financial keywords
  - Source credibility weighting (Reuters/Bloomberg score higher)

A velocity spike (>2×) + positive sentiment (>60%) = bullish signal.
A velocity spike + negative sentiment = bearish signal (risk-off).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple

import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "upgrade", "outperform", "buy", "beat", "record", "surge", "rally",
    "breakthrough", "partnership", "acquisition", "approval", "approved",
    "profit", "revenue", "growth", "expansion", "bull", "strong", "raise",
    "dividend", "buyback", "positive", "launch", "win", "award",
}

NEGATIVE_WORDS = {
    "downgrade", "underperform", "sell", "miss", "decline", "fall", "drop",
    "investigation", "lawsuit", "recall", "loss", "layoff", "cut", "short",
    "warning", "risk", "concern", "fail", "disappoint", "bearish", "fraud",
    "fine", "penalty", "suspension", "delay", "negative",
}

HIGH_CREDIBILITY = {"reuters", "bloomberg", "wsj", "financial times", "barrons", "cnbc", "marketwatch"}


class NewsVelocity:

    CACHE_TTL = 600   # 10 min

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, Optional[ScanResult]]] = {}
        self._lock = threading.RLock()

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < self.CACHE_TTL:
            return cached[1]

        result = self._do_scan(symbol, prices)
        with self._lock:
            self._cache[symbol] = (now, result)
        return result

    def _do_scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news   = ticker.news
            if not news:
                return None

            now_ts = datetime.now(timezone.utc).timestamp()
            cutoff_24h  = now_ts - 86400
            cutoff_7d   = now_ts - 86400 * 7

            articles_24h = []
            articles_7d  = []

            for article in news:
                pub_ts = article.get("providerPublishTime", 0)
                if pub_ts > cutoff_7d:
                    articles_7d.append(article)
                if pub_ts > cutoff_24h:
                    articles_24h.append(article)

            n_24h = len(articles_24h)
            n_7d  = len(articles_7d)
            avg_daily = n_7d / 7.0

            if avg_daily < 0.5 or n_24h < 2:
                return None

            velocity = n_24h / max(avg_daily, 0.1)

            if velocity < 1.5:
                return None   # no unusual activity

            # Sentiment scoring
            pos_score = 0.0
            neg_score = 0.0

            for article in articles_24h:
                title   = (article.get("title", "") or "").lower()
                summary = (article.get("summary", "") or "").lower()
                text    = title + " " + summary

                publisher = (article.get("publisher", "") or "").lower()
                cred_mult = 1.5 if any(s in publisher for s in HIGH_CREDIBILITY) else 1.0

                pos_hits = sum(1 for w in POSITIVE_WORDS if w in text)
                neg_hits = sum(1 for w in NEGATIVE_WORDS if w in text)

                pos_score += pos_hits * cred_mult
                neg_score += neg_hits * cred_mult

            total_score = pos_score + neg_score
            if total_score < 2:
                return None

            sentiment_ratio = pos_score / total_score

            if sentiment_ratio > 0.60:
                boost = round(min(0.04 + (velocity - 1.5) * 0.02 + (sentiment_ratio - 0.6) * 0.10, 0.12), 3)
                result = ScanResult(
                    scanner="news_velocity", symbol=symbol,
                    direction="long", signal_boost=boost,
                    reason=f"News surge {velocity:.1f}× avg ({n_24h} articles), sentiment {sentiment_ratio:.0%} positive",
                    metadata={"velocity": round(velocity, 2), "n_24h": n_24h, "sentiment": round(sentiment_ratio, 3)},
                )
            elif sentiment_ratio < 0.40:
                boost = round(max(-(0.04 + (velocity - 1.5) * 0.02 + (0.4 - sentiment_ratio) * 0.10), -0.12), 3)
                result = ScanResult(
                    scanner="news_velocity", symbol=symbol,
                    direction="short", signal_boost=boost,
                    reason=f"News surge {velocity:.1f}× avg ({n_24h} articles), sentiment {sentiment_ratio:.0%} negative",
                    metadata={"velocity": round(velocity, 2), "n_24h": n_24h, "sentiment": round(sentiment_ratio, 3)},
                )
            else:
                return None

            return result

        except Exception as exc:
            logger.debug("NewsVelocity scan failed for %s: %s", symbol, exc)
            return None
