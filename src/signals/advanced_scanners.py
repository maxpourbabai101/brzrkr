"""Six alpha-generating scanners.

1. SocialSentinel      — StockTwits real-time sentiment velocity
2. OptionsFlowScanner  — Unusual options activity (whale bets)
3. SqueezeRadar        — Short squeeze setup detector
4. CatalystHunter      — Pre-earnings / event-driven momentum
5. BreakoutEngine      — Consolidation + volume-confirmed breakouts
6. MacroFlowScanner    — Cross-asset macro regime signals (TLT, GLD, SPY)

Each scanner returns a ScanResult with a signal_boost in [-0.25, +0.25].
CompositeScanner aggregates all six and returns a single net boost + direction.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    scanner:       str
    symbol:        str
    direction:     str   = "neutral"   # "long" | "short" | "neutral"
    signal_boost:  float = 0.0
    reason:        str   = ""
    metadata:      Dict  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Social Sentinel — StockTwits bullish/bearish velocity
# ---------------------------------------------------------------------------
class SocialSentinel:
    """Polls StockTwits for real-time message count and sentiment split.
    A spike in bullish message volume ahead of price = early-mover signal.
    This is the 'someone found life-changing news on Twitter' detector.
    """

    CACHE_TTL = 300   # 5 min

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, Optional[ScanResult]]] = {}
        self._lock = threading.RLock()

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < self.CACHE_TTL:
            return cached[1]

        try:
            import urllib.request
            import json as _json
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = _json.loads(resp.read())
        except Exception as exc:
            logger.debug("StockTwits fetch failed for %s: %s", symbol, exc)
            with self._lock:
                self._cache[symbol] = (now, None)
            return None

        messages = data.get("messages", [])
        if not messages or len(messages) < 5:
            with self._lock:
                self._cache[symbol] = (now, None)
            return None

        bullish = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bullish"
        )
        bearish = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bearish"
        )
        total = len(messages)

        bull_ratio = bullish / total
        bear_ratio = bearish / total

        result: Optional[ScanResult] = None
        if bull_ratio > 0.65:
            boost = round(min(0.05 + (bull_ratio - 0.65) * 0.40, 0.10), 3)
            result = ScanResult(
                scanner="social_sentinel", symbol=symbol,
                direction="long", signal_boost=boost,
                reason=f"StockTwits {bullish}/{total} bullish ({bull_ratio:.0%})",
                metadata={"bullish": bullish, "bearish": bearish, "total": total},
            )
        elif bear_ratio > 0.65:
            boost = round(max(-(0.05 + (bear_ratio - 0.65) * 0.40), -0.10), 3)
            result = ScanResult(
                scanner="social_sentinel", symbol=symbol,
                direction="short", signal_boost=boost,
                reason=f"StockTwits {bearish}/{total} bearish ({bear_ratio:.0%})",
                metadata={"bullish": bullish, "bearish": bearish, "total": total},
            )

        with self._lock:
            self._cache[symbol] = (now, result)
        return result


# ---------------------------------------------------------------------------
# 2. Options Flow Scanner — unusual call/put volume (whale bets)
# ---------------------------------------------------------------------------
class OptionsFlowScanner:
    """Detects institutional whale options bets via volume/OI ratio.
    Smart money buys options before they move stocks — follow the flow.
    Threshold: vol/OI > 5× is notable; > 20× is a confirmed sweep.
    """

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            exps = ticker.options
            if not exps:
                return None

            current_price = float(prices["close"].iloc[-1])
            near_exps = exps[:2]   # nearest 2 expiries — most liquid

            best_call_ratio = 0.0
            best_put_ratio  = 0.0
            best_call_strike = 0.0
            best_put_strike  = 0.0

            for exp in near_exps:
                chain = ticker.option_chain(exp)

                # OTM calls
                calls = chain.calls
                otm_calls = calls[calls["strike"] > current_price].copy()
                if not otm_calls.empty:
                    otm_calls["ratio"] = (
                        otm_calls["volume"].fillna(0) /
                        otm_calls["openInterest"].replace(0, np.nan).fillna(1)
                    )
                    idx = otm_calls["ratio"].idxmax()
                    r = float(otm_calls.loc[idx, "ratio"])
                    if r > best_call_ratio:
                        best_call_ratio  = r
                        best_call_strike = float(otm_calls.loc[idx, "strike"])

                # OTM puts
                puts = chain.puts
                otm_puts = puts[puts["strike"] < current_price].copy()
                if not otm_puts.empty:
                    otm_puts["ratio"] = (
                        otm_puts["volume"].fillna(0) /
                        otm_puts["openInterest"].replace(0, np.nan).fillna(1)
                    )
                    idx = otm_puts["ratio"].idxmax()
                    r = float(otm_puts.loc[idx, "ratio"])
                    if r > best_put_ratio:
                        best_put_ratio  = r
                        best_put_strike = float(otm_puts.loc[idx, "strike"])

            THRESHOLD = 5.0
            if best_call_ratio >= THRESHOLD and best_call_ratio >= best_put_ratio:
                boost = round(min(0.04 + (best_call_ratio / 80) * 0.12, 0.12), 3)
                return ScanResult(
                    scanner="options_flow", symbol=symbol,
                    direction="long", signal_boost=boost,
                    reason=f"Call sweep vol/OI={best_call_ratio:.1f}× strike=${best_call_strike:.0f}",
                    metadata={"call_ratio": best_call_ratio, "call_strike": best_call_strike},
                )
            if best_put_ratio >= THRESHOLD and best_put_ratio > best_call_ratio:
                boost = round(max(-(0.04 + (best_put_ratio / 80) * 0.12), -0.12), 3)
                return ScanResult(
                    scanner="options_flow", symbol=symbol,
                    direction="short", signal_boost=boost,
                    reason=f"Put sweep vol/OI={best_put_ratio:.1f}× strike=${best_put_strike:.0f}",
                    metadata={"put_ratio": best_put_ratio, "put_strike": best_put_strike},
                )

        except Exception as exc:
            logger.debug("OptionsFlow scan failed for %s: %s", symbol, exc)

        return None


# ---------------------------------------------------------------------------
# 3. Squeeze Radar — short squeeze setup detector
# ---------------------------------------------------------------------------
class SqueezeRadar:
    """Identifies stocks primed for a short squeeze:
    high short interest + price breaking above SMA + volume surge + RSI room.
    GameStop, AMC, NVDA 2023 — all had this setup before their explosive moves.
    """

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            close  = prices["close"].astype(float)
            volume = prices["volume"].astype(float)

            if len(close) < 30:
                return None

            # RSI (14)
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

            # Volume surge
            avg_vol   = float(volume.tail(21).iloc[:-1].mean())
            curr_vol  = float(volume.iloc[-1])
            vol_ratio = curr_vol / max(avg_vol, 1)

            # MA breakout
            price  = float(close.iloc[-1])
            sma20  = float(close.rolling(20).mean().iloc[-1])
            sma50  = float(close.rolling(min(50, len(close))).mean().iloc[-1])

            above_sma20 = price > sma20
            above_sma50 = price > sma50
            vol_surge   = vol_ratio >= 2.0
            rsi_room    = 35 <= rsi <= 65

            score = sum([above_sma20, above_sma50, vol_surge, rsi_room])
            if score < 3 or not vol_surge:
                return None

            # Optional: short interest from yfinance
            short_pct = 0.0
            try:
                import yfinance as yf
                info = yf.Ticker(symbol).info
                short_pct = float(info.get("shortPercentOfFloat") or 0)
            except Exception:
                pass

            squeeze_score = score + (2 if short_pct > 0.10 else 0)
            boost = round(min(0.04 * squeeze_score, 0.15), 3)

            return ScanResult(
                scanner="squeeze_radar", symbol=symbol,
                direction="long", signal_boost=boost,
                reason=(f"Squeeze setup: vol={vol_ratio:.1f}× avg, RSI={rsi:.0f}, "
                        f"above SMA20={above_sma20}, short%={short_pct:.0%}"),
                metadata={
                    "vol_ratio": round(vol_ratio, 2),
                    "rsi": round(rsi, 1),
                    "short_pct": round(short_pct, 3),
                    "squeeze_score": squeeze_score,
                },
            )

        except Exception as exc:
            logger.debug("SqueezeRadar scan failed for %s: %s", symbol, exc)

        return None


# ---------------------------------------------------------------------------
# 4. Catalyst Hunter — pre-earnings drift + event-driven momentum
# ---------------------------------------------------------------------------
class CatalystHunter:
    """Stocks drift in the 3–7 days before earnings as smart money positions.
    We enter WITH the drift and exit BEFORE the print (earnings itself = coin flip).
    Sweet spot: 3–7 trading days out, with price trending in one direction.
    """

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar

            if cal is None or (hasattr(cal, "empty") and cal.empty):
                return None

            today = datetime.now(timezone.utc).date()

            # Handle both DataFrame and dict calendar formats
            if hasattr(cal, "columns"):
                earnings_cols = [c for c in cal.columns if "Earnings" in str(c)]
                if not earnings_cols:
                    return None
                earnings_dates = cal[earnings_cols[0]].dropna()
                if earnings_dates.empty:
                    return None
                next_earnings = pd.Timestamp(earnings_dates.iloc[0]).date()
            elif isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if not ed:
                    return None
                next_earnings = pd.Timestamp(ed[0] if isinstance(ed, list) else ed).date()
            else:
                return None

            days_to_earnings = (next_earnings - today).days
            if not (3 <= days_to_earnings <= 7):
                return None

            # Pre-earnings drift conditions
            close = prices["close"].astype(float)
            if len(close) < 10:
                return None

            momentum_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)

            if momentum_5d > 1.0:
                boost = round(min(0.05 + momentum_5d * 0.01, 0.12), 3)
                return ScanResult(
                    scanner="catalyst_hunter", symbol=symbol,
                    direction="long", signal_boost=boost,
                    reason=f"Pre-earnings drift: {days_to_earnings}d to report, +{momentum_5d:.1f}% 5d",
                    metadata={"days_to_earnings": days_to_earnings,
                              "momentum_5d": round(momentum_5d, 2),
                              "earnings_date": str(next_earnings)},
                )
            if momentum_5d < -1.0:
                boost = round(max(-(0.05 + abs(momentum_5d) * 0.01), -0.12), 3)
                return ScanResult(
                    scanner="catalyst_hunter", symbol=symbol,
                    direction="short", signal_boost=boost,
                    reason=f"Pre-earnings breakdown: {days_to_earnings}d to report, {momentum_5d:.1f}% 5d",
                    metadata={"days_to_earnings": days_to_earnings,
                              "momentum_5d": round(momentum_5d, 2)},
                )

        except Exception as exc:
            logger.debug("CatalystHunter scan failed for %s: %s", symbol, exc)

        return None


# ---------------------------------------------------------------------------
# 5. Breakout Engine — consolidation + volume-confirmed breakouts
# ---------------------------------------------------------------------------
class BreakoutEngine:
    """The highest-conviction technical setup:
    1. Stock consolidates in a tight range (BB width in bottom 30% of 90-day history)
    2. Price explodes through the top on volume ≥ 2.5× average
    3. 52-week high breakout adds extra confirmation (+0.04)
    Breakouts from tight coils produce the biggest sustained moves.
    """

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            close  = prices["close"].astype(float)
            high   = prices["high"].astype(float)
            volume = prices["volume"].astype(float)

            if len(close) < 60:
                return None

            # Bollinger bandwidth (normalised)
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_width = (4 * std20) / sma20.replace(0, np.nan)

            bb_hist = bb_width.dropna().tail(90)
            if bb_hist.empty:
                return None
            curr_bw = float(bb_hist.iloc[-1])
            bb_pct  = float((bb_hist < curr_bw).mean())   # 0=tightest, 1=widest

            # Volume ratio
            avg_vol   = float(volume.tail(21).iloc[:-1].mean())
            curr_vol  = float(volume.iloc[-1])
            vol_ratio = curr_vol / max(avg_vol, 1)

            # Breakout conditions
            upper_band  = (sma20 + 2 * std20).iloc[-1]
            curr_price  = float(close.iloc[-1])
            above_upper = curr_price > float(upper_band)

            high_52w    = float(high.tail(252).max())
            at_52w_high = curr_price >= high_52w * 0.998

            tight_coil = bb_pct < 0.30
            vol_surge  = vol_ratio >= 2.5
            breakout   = above_upper or at_52w_high

            if tight_coil and vol_surge and breakout:
                boost = 0.08
                if at_52w_high:
                    boost += 0.04
                if vol_ratio >= 4.0:
                    boost += 0.03
                boost = round(min(boost, 0.18), 3)

                return ScanResult(
                    scanner="breakout_engine", symbol=symbol,
                    direction="long", signal_boost=boost,
                    reason=(f"Breakout: BB_pct={bb_pct:.0%} tight, "
                            f"vol={vol_ratio:.1f}×, 52wHigh={at_52w_high}"),
                    metadata={
                        "bb_percentile": round(bb_pct, 3),
                        "vol_ratio":     round(vol_ratio, 2),
                        "at_52w_high":   at_52w_high,
                        "above_upper_band": above_upper,
                    },
                )

        except Exception as exc:
            logger.debug("BreakoutEngine scan failed for %s: %s", symbol, exc)

        return None


# ---------------------------------------------------------------------------
# 6. Macro Flow Scanner — cross-asset macro regime signals
# ---------------------------------------------------------------------------
class MacroFlowScanner:
    """Reads macro instrument momentum to generate forward-looking signals:
    - TLT rising  → yields falling  → growth stocks outperform (QQQ, TQQQ, NVDA)
    - GLD surging → dollar weakness → commodity ETFs (GLD, SLV, USO)
    - SPY momentum → broad market trend confirmation
    Cross-asset flows lead individual stock prices by minutes to days.
    """

    TECH    = {"QQQ", "TQQQ", "NVDA", "AMD", "MSFT", "META", "AAPL"}
    COMMODS = {"GLD", "SLV", "USO"}
    BROAD   = {"SPY", "QQQ", "DIA", "IWM"}

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            import yfinance as yf

            def _mom(ticker_sym: str, days: int) -> Optional[float]:
                raw = yf.download(ticker_sym, period="2mo", interval="1d",
                                  progress=False, auto_adjust=True)
                if raw.empty or len(raw) < days + 1:
                    return None
                if isinstance(raw.columns[0], tuple):
                    raw.columns = [c[0].lower() for c in raw.columns]
                else:
                    raw.columns = [c.lower() for c in raw.columns]
                return float((raw["close"].iloc[-1] / raw["close"].iloc[-days - 1] - 1) * 100)

            signals = []

            if symbol in self.TECH:
                tlt_mom = _mom("TLT", 5)
                if tlt_mom is not None and tlt_mom > 1.0:
                    signals.append(("long", 0.06,
                        f"TLT +{tlt_mom:.1f}% 5d → yields falling → growth tailwind"))

            if symbol in self.COMMODS:
                gld_mom = _mom("GLD", 5)
                if gld_mom is not None and gld_mom > 1.5:
                    signals.append(("long", 0.07,
                        f"GLD +{gld_mom:.1f}% 5d → commodity/dollar-weakness surge"))

            if symbol in self.BROAD:
                spy_mom = _mom("SPY", 10)
                if spy_mom is not None and spy_mom > 2.0:
                    signals.append(("long", 0.05,
                        f"SPY +{spy_mom:.1f}% 10d → broad market bull"))
                elif spy_mom is not None and spy_mom < -2.0 and symbol == "SQQQ":
                    signals.append(("long", 0.06,
                        f"SPY {spy_mom:.1f}% 10d → inverse ETF SQQQ tailwind"))

            if not signals:
                return None

            signals.sort(key=lambda x: x[1], reverse=True)
            direction, boost, reason = signals[0]

            return ScanResult(
                scanner="macro_flow", symbol=symbol,
                direction=direction, signal_boost=boost, reason=reason,
                metadata={"macro_signals": [s[2] for s in signals]},
            )

        except Exception as exc:
            logger.debug("MacroFlow scan failed for %s: %s", symbol, exc)

        return None


# ---------------------------------------------------------------------------
# CompositeScanner — runs all 12, returns net boost + direction
# ---------------------------------------------------------------------------
class CompositeScanner:
    """Runs all 12 scanners for a symbol and aggregates:
    - total_boost: additive float in [-0.30, +0.30] for the confidence score
    - direction:   majority vote across scanners ("long"/"short"/"neutral")
    - results:     list of individual ScanResult objects that fired
    """

    def __init__(self) -> None:
        # Original 6 scanners
        from src.signals.advanced_scanners import (
            SocialSentinel, OptionsFlowScanner, SqueezeRadar,
            CatalystHunter, BreakoutEngine, MacroFlowScanner,
        )
        # Second wave — technical & pattern
        from src.signals.technical_confluence import TechnicalConfluence
        from src.signals.news_velocity        import NewsVelocity
        from src.signals.pattern_detector     import PricePatternDetector
        from src.signals.volume_profile       import VolumeProfileEngine
        from src.signals.sector_rotation      import SectorRotationTracker
        from src.signals.multi_timeframe      import MultiTimeframeConfluence
        # Third wave — statistical & fundamental
        from src.signals.statistical_edge     import StatisticalEdge
        from src.signals.fundamental_catalyst import FundamentalCatalyst
        # Fourth wave — expert panel additions
        from src.signals.iv_rank_scanner      import IVRankScanner
        from src.signals.market_regime_scanner import MarketRegimeScanner
        from src.signals.momentum_age_scanner  import MomentumAgeScanner

        self.scanners = [
            SocialSentinel(),
            OptionsFlowScanner(),
            SqueezeRadar(),
            CatalystHunter(),
            BreakoutEngine(),
            MacroFlowScanner(),
            TechnicalConfluence(),
            NewsVelocity(),
            PricePatternDetector(),
            VolumeProfileEngine(),
            SectorRotationTracker(),
            MultiTimeframeConfluence(),
            StatisticalEdge(),
            FundamentalCatalyst(),
            IVRankScanner(),
            MarketRegimeScanner(),
            MomentumAgeScanner(),
        ]

        # Scanner attributor for performance-weighted voting
        from src.learning.scanner_attribution import get_attributor
        self._attributor = get_attributor()

    def run(
        self, symbol: str, prices: pd.DataFrame
    ) -> Tuple[float, str, List[ScanResult]]:
        fired: List[ScanResult] = []
        for sc in self.scanners:
            try:
                r = sc.scan(symbol, prices)
                if r is not None:
                    fired.append(r)
            except Exception as exc:
                logger.debug("Scanner %s errored for %s: %s",
                             sc.__class__.__name__, symbol, exc)

        if not fired:
            return 0.0, "neutral", []

        # Get attribution-based weights for each scanner
        scanner_names = [sc.__class__.__name__ for sc in self.scanners]
        fired_names   = [r.scanner for r in fired]
        try:
            attr_weights = self._attributor.get_scanner_weights(scanner_names)
        except Exception:
            attr_weights = {n: 1.0 / len(scanner_names) for n in scanner_names}

        long_votes  = sum(1 for r in fired if r.direction == "long")
        short_votes = sum(1 for r in fired if r.direction == "short")

        if long_votes > short_votes:
            dominant = "long"
        elif short_votes > long_votes:
            dominant = "short"
        else:
            dominant = "neutral"

        # Attribution-weighted boost summation
        total_boost = 0.0
        for r in fired:
            # Look up this scanner's attribution weight (normalised → multiply by n_scanners to get relative weight)
            scanner_cls = next(
                (sc.__class__.__name__ for sc in self.scanners
                 if sc.__class__.__name__.lower().replace("scanner","").replace("tracker","")
                    in r.scanner.lower().replace("_","")),
                r.scanner
            )
            attr_w = attr_weights.get(scanner_cls, 1.0 / len(self.scanners)) * len(self.scanners)
            weighted_boost = r.signal_boost * attr_w

            if r.direction == dominant:
                total_boost += weighted_boost
            elif r.direction != "neutral":
                total_boost -= abs(weighted_boost) * 0.5

        # Agreement additive bonus (replaces procyclical multiplier).
        #
        # Old approach: multiply by 1.5× when 8+ scanners fire.
        # Problem: in strong momentum markets ALL 17 scanners align → 1.5× boost
        # creates overconfidence exactly when trades are most crowded.
        #
        # New approach: add a small fixed bonus per agreement tier.
        # More scanners = marginally more confidence, but total boost is still
        # clipped to ±0.35 so extreme agreement can't override the clip.
        agreement = max(long_votes, short_votes)
        if agreement >= 8:
            total_boost += 0.04 * np.sign(total_boost)   # +0.04 at 8+ (highest tier)
        elif agreement >= 6:
            total_boost += 0.03 * np.sign(total_boost)
        elif agreement >= 4:
            total_boost += 0.02 * np.sign(total_boost)
        elif agreement >= 3:
            total_boost += 0.01 * np.sign(total_boost)

        total_boost = float(np.clip(total_boost, -0.35, 0.35))

        if fired:
            reasons = " | ".join(r.reason for r in fired[:4])
            logger.info(
                "Scanners [%s] %d/%d fired: boost=%+.3f dir=%s — %s",
                symbol, len(fired), len(self.scanners), total_boost, dominant, reasons,
            )

        return total_boost, dominant, fired
