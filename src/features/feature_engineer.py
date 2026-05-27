"""FeatureEngineer — convert a heterogeneous data bundle into a flat,
model-ready feature set.

A "bundle" is the dict returned by either
:func:`src.data_loader.load_data` or
:meth:`src.data_scraper.WebDataScraper.scrape_all`. Keys can be any
subset of::

    prices, options, news, finnhub_news, av_news, google_news,
    reddit, stocktwits, hacker_news, insider, finnhub_insider,
    congress, treasury, macro, cot, short_vol, wiki_views

This module is tolerant: every missing source contributes NaN to the
corresponding context feature instead of raising.

Output:
    A pandas.DataFrame indexed by the price timeseries (most recent
    256 bars by default) with the OHLCV columns plus engineered
    technical indicators. The point-in-time crowd/event features are
    attached as ``df.attrs["context"]`` — a flat dict the ensemble can
    read as scalars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineer:
    """Stateless pipeline. One instance per process is fine."""

    window: int = 256
    sma_short: int = 20
    sma_long: int = 50
    rsi_period: int = 14
    realized_vol_window: int = 20

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def build_features(self, bundle: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = bundle.get("prices")
        if prices is None or prices.empty:
            logger.warning("FeatureEngineer: empty/missing prices — returning empty frame")
            return pd.DataFrame()

        feat = self._technical_features(prices.copy())
        feat = feat.tail(self.window)

        context: Dict[str, Any] = {}
        context.update(self._sentiment_features(bundle))
        context.update(self._insider_features(bundle))
        context.update(self._congress_features(bundle))
        context.update(self._options_features(bundle, spot=float(feat["close"].iloc[-1])))
        context.update(self._macro_features(bundle))
        context.update(self._attention_features(bundle))

        feat.attrs["context"] = context
        return feat

    # ------------------------------------------------------------------
    # Per-source feature builders
    # ------------------------------------------------------------------
    def _technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns, SMA cross, RSI, momentum, realized vol."""
        df = df.copy()
        # Make sure column names are lowercase.
        # yfinance ≥ 0.2 returns MultiIndex tuples — pick the OHLCV field part.
        _ohlcv = frozenset(("open", "high", "low", "close", "volume"))
        df.columns = [next((p.lower() for p in c if p.lower() in _ohlcv), c[0].lower())
                      if isinstance(c, tuple) else c.lower() for c in df.columns]
        if "close" not in df.columns:
            raise ValueError("price frame missing 'close' column")

        df["ret_1"] = df["close"].pct_change()
        df["ret_5"] = df["close"].pct_change(5)
        df["ret_20"] = df["close"].pct_change(20)

        df[f"sma_{self.sma_short}"] = df["close"].rolling(self.sma_short).mean()
        df[f"sma_{self.sma_long}"] = df["close"].rolling(self.sma_long).mean()
        df["sma_spread"] = (
            df[f"sma_{self.sma_short}"] - df[f"sma_{self.sma_long}"]
        ) / df["close"]

        df["realized_vol"] = (
            df["ret_1"].rolling(self.realized_vol_window).std() * np.sqrt(252)
        )

        df["rsi"] = _rsi(df["close"], period=self.rsi_period)

        if "volume" in df.columns:
            df["vol_z"] = (
                df["volume"] - df["volume"].rolling(20).mean()
            ) / df["volume"].rolling(20).std()
        return df

    @staticmethod
    def _sentiment_features(bundle: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Aggregate sentiment scores across every text source that
        carries them. Each source contributes mean / std / count / max-abs.
        """
        out: Dict[str, float] = {}
        score_cols = ("sentiment_score", "overall_sentiment_score")
        for key in ("news", "google_news", "reddit", "stocktwits",
                    "hacker_news", "av_news", "finnhub_news"):
            df = bundle.get(key)
            if df is None or df.empty:
                out[f"{key}_count"] = 0.0
                out[f"{key}_sent_mean"] = np.nan
                out[f"{key}_sent_salience"] = np.nan
                continue
            scores = None
            for col in score_cols:
                if col in df.columns:
                    scores = pd.to_numeric(df[col], errors="coerce").dropna()
                    break
            out[f"{key}_count"] = float(len(df))
            if scores is not None and not scores.empty:
                out[f"{key}_sent_mean"] = float(scores.mean())
                out[f"{key}_sent_salience"] = float(scores.abs().max())
            else:
                out[f"{key}_sent_mean"] = np.nan
                out[f"{key}_sent_salience"] = np.nan
        return out

    @staticmethod
    def _insider_features(bundle: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Form 4 / Finnhub insider activity in the trailing 60 days."""
        out: Dict[str, float] = {"insider_filings_60d": 0.0,
                                 "insider_net_share_change": np.nan}
        cutoff = datetime.now(timezone.utc) - timedelta(days=60)

        df = bundle.get("finnhub_insider")
        if df is not None and not df.empty and "transactionDate" in df.columns:
            recent = df[df["transactionDate"] >= cutoff]
            out["insider_filings_60d"] = float(len(recent))
            if "change" in recent.columns:
                out["insider_net_share_change"] = float(
                    pd.to_numeric(recent["change"], errors="coerce").sum()
                )
            return out

        df = bundle.get("insider")
        if df is not None and not df.empty:
            # EDGAR scraper indexes by filed_at.
            try:
                recent = df.loc[df.index >= cutoff]
            except TypeError:
                recent = df
            out["insider_filings_60d"] = float(len(recent))
        return out

    @staticmethod
    def _congress_features(bundle: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """STOCK Act disclosures in the trailing 60 days."""
        out = {
            "congress_buys_60d": 0.0,
            "congress_sells_60d": 0.0,
            "congress_net_60d": 0.0,
        }
        df = bundle.get("congress")
        if df is None or df.empty:
            return out
        types = df.get("type", pd.Series(dtype=str)).astype(str).str.lower()
        out["congress_buys_60d"] = float(
            types.str.contains("purchase").sum()
        )
        out["congress_sells_60d"] = float(
            types.str.contains("sale").sum()
        )
        out["congress_net_60d"] = (
            out["congress_buys_60d"] - out["congress_sells_60d"]
        )
        return out

    @staticmethod
    def _options_features(
        bundle: Dict[str, pd.DataFrame],
        *,
        spot: float,
    ) -> Dict[str, float]:
        """Put/call ratio, IV at the money, IV skew (25-delta proxy)."""
        out: Dict[str, float] = {
            "pc_ratio": np.nan,
            "iv_atm": np.nan,
            "iv_skew": np.nan,
        }
        df = bundle.get("options")
        if df is None or df.empty or "type" not in df.columns:
            return out

        calls = df[df["type"] == "call"]
        puts = df[df["type"] == "put"]
        if "volume" in df.columns:
            cvol = pd.to_numeric(calls["volume"], errors="coerce").fillna(0).sum()
            pvol = pd.to_numeric(puts["volume"], errors="coerce").fillna(0).sum()
            if cvol > 0:
                out["pc_ratio"] = float(pvol / cvol)

        if "iv" in df.columns and spot > 0:
            # IV ATM = nearest-strike call IV.
            df = df.copy()
            df["dist"] = (pd.to_numeric(df["strike"], errors="coerce") - spot).abs()
            df = df.dropna(subset=["iv", "dist"])
            if not df.empty:
                near = df.sort_values("dist").iloc[0]
                out["iv_atm"] = float(near["iv"])

                # Crude skew: avg IV of OTM puts - avg IV of OTM calls.
                otm_puts = df[(df["type"] == "put") & (df["strike"] < spot * 0.95)]
                otm_calls = df[(df["type"] == "call") & (df["strike"] > spot * 1.05)]
                if not otm_puts.empty and not otm_calls.empty:
                    out["iv_skew"] = float(
                        otm_puts["iv"].mean() - otm_calls["iv"].mean()
                    )
        return out

    @staticmethod
    def _macro_features(bundle: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Yield curve + VIX (whichever source has them)."""
        out = {"y10y2_spread": np.nan, "vix_level": np.nan}
        for key in ("macro", "treasury"):
            df = bundle.get(key)
            if df is None or df.empty:
                continue
            # Take latest row.
            last = df.iloc[-1]
            if "DGS10" in df.columns and "DGS2" in df.columns:
                out["y10y2_spread"] = float(last["DGS10"]) - float(last["DGS2"])
            elif "10 Yr" in df.columns and "2 Yr" in df.columns:
                # Treasury Direct CSV column naming.
                out["y10y2_spread"] = float(last["10 Yr"]) - float(last["2 Yr"])
            if "VIXCLS" in df.columns:
                out["vix_level"] = float(last["VIXCLS"])
        return out

    @staticmethod
    def _attention_features(bundle: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Retail attention proxies: wiki pageviews z-score, reddit total score."""
        out = {"wiki_views_z": np.nan, "reddit_score_sum": np.nan}
        wiki = bundle.get("wiki_views")
        if wiki is not None and not wiki.empty and "views" in wiki.columns:
            v = pd.to_numeric(wiki["views"], errors="coerce").dropna()
            if len(v) >= 7:
                base = v.iloc[:-1]
                z = (v.iloc[-1] - base.mean()) / (base.std() + 1e-9)
                out["wiki_views_z"] = float(z)

        reddit = bundle.get("reddit")
        if reddit is not None and not reddit.empty and "score" in reddit.columns:
            out["reddit_score_sum"] = float(
                pd.to_numeric(reddit["score"], errors="coerce").fillna(0).sum()
            )
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Classic 14-period RSI on a price series."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))
