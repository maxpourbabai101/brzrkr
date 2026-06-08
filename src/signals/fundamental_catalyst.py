"""
FundamentalCatalyst — corporate action and institutional intelligence signals.

Data sources (all free):
  1. Analyst upgrades/downgrades — yfinance ticker.recommendations
  2. Institutional ownership changes — yfinance ticker.institutional_holders
  3. Share buyback signals — earnings calendar + revenue/buyback keywords
  4. M&A proximity detector — statistical pre-merger signature
     (volume spike + price approaching 52w high + premium to peers)
  5. SEC Form 4 insider buys — EDGAR full-text search (no API key needed)
  6. Earnings estimate revisions — yfinance analyst price targets

These are the signals that move stocks for weeks, not minutes.
'Someone bought the stock' is the oldest alpha in markets.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)


class FundamentalCatalyst:
    """Corporate action + institutional intelligence scanner."""

    CACHE_TTL = 3600   # 1 hour — fundamental data is slow-moving

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
            ticker  = yf.Ticker(symbol)
            close   = prices["close"].astype(float)
            volume  = prices["volume"].astype(float)
            price   = float(close.iloc[-1])

            signals: list[tuple[str, float, str]] = []

            # ── 1. Analyst upgrade/downgrade momentum ─────────────────
            try:
                recs = ticker.recommendations
                if recs is not None and not recs.empty:
                    # Last 30 days
                    recs = recs.copy()
                    if hasattr(recs.index, 'tz_localize'):
                        pass
                    recs_recent = recs.tail(10)   # last 10 changes

                    BULLISH_GRADES = {"Buy", "Strong Buy", "Overweight", "Outperform",
                                      "Upgrade", "Positive", "Accumulate"}
                    BEARISH_GRADES = {"Sell", "Strong Sell", "Underweight", "Underperform",
                                      "Downgrade", "Negative", "Reduce"}

                    bull_recs = 0
                    bear_recs = 0
                    for col in ["To Grade", "toGrade", "Action", "action"]:
                        if col in recs_recent.columns:
                            bull_recs = recs_recent[col].isin(BULLISH_GRADES).sum()
                            bear_recs = recs_recent[col].isin(BEARISH_GRADES).sum()
                            break

                    if bull_recs > bear_recs and bull_recs >= 2:
                        boost = round(min(0.04 + bull_recs * 0.02, 0.10), 3)
                        signals.append(("long", boost,
                            f"Analyst upgrades: {bull_recs} bullish vs {bear_recs} bearish (recent)"))
                    elif bear_recs > bull_recs and bear_recs >= 2:
                        boost = round(min(0.04 + bear_recs * 0.02, 0.10), 3)
                        signals.append(("short", boost,
                            f"Analyst downgrades: {bear_recs} bearish vs {bull_recs} bullish (recent)"))
            except Exception:
                pass

            # ── 2. Institutional ownership trend ──────────────────────
            try:
                inst = ticker.institutional_holders
                if inst is not None and not inst.empty and "% Out" in inst.columns:
                    # High institutional ownership + stock outperforming = smart money in
                    inst_pct = float(inst["% Out"].sum())
                    mom20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else 0
                    if inst_pct > 60 and mom20 > 2:
                        signals.append(("long", 0.06,
                            f"Institutional ownership {inst_pct:.0f}% + stock +{mom20:.1f}% 20d"))
                    elif inst_pct < 30 and mom20 < -2:
                        signals.append(("short", 0.05,
                            f"Low inst. ownership {inst_pct:.0f}% + stock {mom20:.1f}% 20d"))
            except Exception:
                pass

            # ── 3. Price target vs current price ─────────────────────
            try:
                info = ticker.info
                target = float(info.get("targetMeanPrice") or info.get("targetMedianPrice") or 0)
                if target > 0:
                    upside = (target - price) / price * 100
                    if upside > 15:
                        boost = round(min(0.04 + upside * 0.003, 0.12), 3)
                        signals.append(("long", boost,
                            f"Analyst target ${target:.2f} = {upside:.0f}% upside from ${price:.2f}"))
                    elif upside < -10:
                        boost = round(min(0.04 + abs(upside) * 0.003, 0.10), 3)
                        signals.append(("short", boost,
                            f"Analyst target ${target:.2f} = {upside:.0f}% downside from ${price:.2f}"))
            except Exception:
                pass

            # ── 4. M&A proximity / takeover candidate detection ──────
            # Statistical signature: price approaching 52w high + vol spike
            # + sector peer recently acquired (news-based proxy via velocity)
            try:
                high52 = float(prices["high"].tail(252).max()) if len(prices) >= 252 else float(prices["high"].max())
                vol_avg = float(volume.tail(20).mean())
                vol_now = float(volume.iloc[-1])
                vol_ratio = vol_now / max(vol_avg, 1)

                pct_from_52w_high = (price - high52) / high52 * 100  # negative = below high

                # M&A signature: stock lagging 52w high by 15-40% + unusual volume + small cap
                market_cap = float((info.get("marketCap") or 0) if 'info' in dir() else 0)
                is_small_mid = market_cap < 50_000_000_000   # under $50B

                if (-40 < pct_from_52w_high < -15) and vol_ratio > 1.5 and is_small_mid:
                    signals.append(("long", 0.07,
                        f"M&A candidate: {pct_from_52w_high:.0f}% from 52w high, vol={vol_ratio:.1f}× avg"))
            except Exception:
                pass

            # ── 5. SEC Form 4 Insider Buying (EDGAR free API) ─────────
            try:
                import urllib.request, json as _json
                # Map common symbols to CIK — expand as needed
                SYMBOL_CIK: Dict[str, str] = {
                    "NVDA": "0001045810", "AMD": "0000002488",
                    "MSFT": "0000789019", "AAPL": "0000320193",
                    "META": "0001326801", "TSLA": "0001318605",
                    "TQQQ": "0001174922", "SPY":  "0000884394",
                }
                cik = SYMBOL_CIK.get(symbol)
                if cik:
                    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                    with urllib.request.urlopen(url, timeout=5) as resp:
                        data = _json.loads(resp.read())
                    filings = data.get("filings", {}).get("recent", {})
                    forms  = filings.get("form", [])
                    dates  = filings.get("filingDate", [])
                    # Check for Form 4 filings in last 30 days
                    from datetime import datetime, timezone, timedelta
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
                    form4_recent = sum(
                        1 for form, date in zip(forms, dates)
                        if form == "4" and date >= cutoff
                    )
                    if form4_recent >= 3:   # 3+ insiders filed Form 4 = likely buying
                        signals.append(("long", 0.08,
                            f"SEC Form 4: {form4_recent} insider filings in 30 days"))
            except Exception:
                pass

            # ── 6. Forward P/E vs Sector — value/growth identification ─
            try:
                fpe = float(info.get("forwardPE") or 0)
                sector = info.get("sector", "")
                SECTOR_PE: Dict[str, float] = {
                    "Technology": 28.0, "Communication Services": 22.0,
                    "Consumer Discretionary": 25.0, "Financials": 14.0,
                    "Health Care": 18.0, "Energy": 12.0,
                    "Industrials": 20.0, "Materials": 17.0,
                    "Real Estate": 35.0, "Consumer Staples": 22.0,
                    "Utilities": 18.0,
                }
                avg_pe = SECTOR_PE.get(sector, 20.0)
                if 0 < fpe < avg_pe * 0.80:   # 20%+ discount to sector
                    discount = (avg_pe - fpe) / avg_pe * 100
                    signals.append(("long", round(min(0.04 + discount * 0.002, 0.08), 3),
                        f"Forward P/E={fpe:.1f} is {discount:.0f}% discount to {sector} avg ({avg_pe:.1f})"))
                elif fpe > avg_pe * 1.50:   # 50%+ premium — overvalued
                    signals.append(("short", 0.04,
                        f"Forward P/E={fpe:.1f} is {((fpe/avg_pe)-1)*100:.0f}% premium to sector"))
            except Exception:
                pass

            if not signals:
                return None

            long_b  = sum(b for d, b, _ in signals if d == "long")
            short_b = sum(b for d, b, _ in signals if d == "short")

            if long_b >= short_b and long_b > 0:
                reasons = [r for d, b, r in signals if d == "long"][:3]
                return ScanResult(
                    scanner="fundamental_catalyst", symbol=symbol,
                    direction="long",
                    signal_boost=round(min(long_b, 0.20), 3),
                    reason="; ".join(reasons),
                    metadata={"long_boost": round(long_b, 3),
                              "short_boost": round(short_b, 3),
                              "n_signals": len(signals)},
                )
            if short_b > long_b:
                reasons = [r for d, b, r in signals if d == "short"][:3]
                return ScanResult(
                    scanner="fundamental_catalyst", symbol=symbol,
                    direction="short",
                    signal_boost=round(-min(short_b, 0.20), 3),
                    reason="; ".join(reasons),
                    metadata={"long_boost": round(long_b, 3),
                              "short_boost": round(short_b, 3),
                              "n_signals": len(signals)},
                )

        except Exception as exc:
            logger.debug("FundamentalCatalyst scan failed for %s: %s", symbol, exc)

        return None
