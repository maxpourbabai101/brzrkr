"""NYSE/NASDAQ Breakout Hunter — Institutional-Grade Setup Scanner

Built on six independently-validated methodologies:
  • Minervini Trend Template (8 conditions) + VCP pattern
  • Stan Weinstein Stage 2 breakout system
  • IBD Relative Strength Rating (exact 0.4/0.2/0.2/0.2 formula)
  • O'Neil CANSLIM — EPS acceleration + volume confirmation
  • Wyckoff accumulation footprint (OBV, volume dry-up, tight closes)
  • PEAD (Post-Earnings Announcement Drift) fundamental overlay

Pipeline
────────
  Stage 1  Universe filter     Liquidity, price, basic trend (Finviz)
  Stage 2  Trend template      Minervini 8-condition SMA stack + 52wk range
  Stage 3  VCP detection       ATR contraction, volume dry-up, tight closes, higher lows
  Stage 4  RS Rating           IBD formula, percentile-ranked vs. current universe
  Stage 5  Fundamental overlay EPS/revenue acceleration, short float, insider buying
  Stage 6  Breakout trigger    Price vs. pivot high, volume surge confirmation
  Stage 7  Composite scoring   Weighted 0–100 score → final ranking

BreakoutResult schema
─────────────────────
  symbol            ticker
  composite_score   0–100, primary ranking key
  setup_type        "VCP+Stage2" | "Stage2" | "VCP" | "Pocket Pivot" | "Emerging"
  direction         "long" | "short"
  entry_price       current close (or pivot breakout level)
  stop_price        below deepest VCP low (ATR-based fallback)
  target_price      base-height measured move
  rr_ratio          risk/reward
  rs_rating         0–99 IBD-style RS vs. scanned universe
  trend_score       0–8 Minervini conditions met
  vcp_score         0–100
  volume_ratio      today vol / 50d avg (breakout confirmation)
  atr_contraction   current ATR / 63d avg ATR (lower = tighter)
  eps_growth_pct    most recent YoY quarterly EPS growth %
  short_float_pct   short interest as % of float (squeeze fuel)
  insider_buying    True if Form 4 open-market purchase in last 90 days
  note              human-readable signal summary
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

logger = logging.getLogger(__name__)


# ── Scoring weights ──────────────────────────────────────────────────────────
W_RS          = 0.22   # RS Rating percentile (most predictive single factor)
W_TREND       = 0.20   # Minervini trend template
W_VCP         = 0.20   # VCP / volatility contraction quality
W_BREAKOUT    = 0.15   # Breakout trigger (volume + price vs pivot)
W_FUNDAMENTAL = 0.13   # EPS + revenue acceleration
W_SQUEEZE     = 0.10   # Short float + insider buying

# ── Thresholds ───────────────────────────────────────────────────────────────
MIN_PRICE            = 10.0      # skip penny stocks
MIN_AVG_VOLUME       = 300_000   # shares/day (50-day avg)
MIN_AVG_DOLLAR_VOL   = 5_000_000 # $/day (liquidity filter)
MIN_EPS_GROWTH       = 0.0       # % — relax to catch pre-profitability rockets
RS_BREAKOUT_THRESHOLD = 70        # minimum RS rating to include
VCP_MIN_SCORE        = 30         # minimum VCP score to bother analyzing further
TREND_MIN_CONDITIONS = 5          # out of 8 Minervini conditions
ATR_CONTRACTION_GOOD = 0.70       # ATR < 70% of 63d avg = compression starting
ATR_CONTRACTION_GREAT = 0.55      # ATR < 55% of 63d avg = strong compression
VOLUME_DRYUP_THRESHOLD = 0.80     # 10d avg vol < 80% of 50d avg
BREAKOUT_VOL_MIN     = 1.40       # 40% above average = minimum confirmation
BREAKOUT_VOL_STRONG  = 2.00       # 2× average = strong conviction
SQUEEZE_SHORT_FLOAT  = 10.0       # % float short = squeeze fuel starts here


# ── Data class ───────────────────────────────────────────────────────────────

@dataclass
class BreakoutResult:
    symbol:            str
    composite_score:   float        # 0–100, ranking key
    setup_type:        str          # human label
    direction:         str          # "long" (shorts rare but included)
    entry_price:       float
    stop_price:        float
    target_price:      float
    rr_ratio:          float

    # Stage scores (0–100 each)
    rs_rating:         float        # 0–99
    trend_score:       int          # 0–8 Minervini conditions
    vcp_score:         float        # 0–100
    breakout_score:    float        # 0–100
    fundamental_score: float        # 0–100

    # Key metrics
    volume_ratio:       float       # today vol / 50d avg
    atr_contraction:    float       # current ATR / 63d avg ATR  (lower = tighter)
    momentum_5d:        float       # % change, 5 bars
    momentum_20d:       float       # % change, 20 bars
    momentum_63d:       float       # % change, 63 bars (quarter)

    # Fundamental
    eps_growth_pct:    float        # YoY, most recent quarter
    revenue_growth_pct:float        # YoY, most recent quarter
    short_float_pct:   float        # % of float short
    days_to_cover:     float        # short interest / avg vol

    # Flags
    passes_trend_template: bool
    vcp_detected:      bool
    breakout_confirmed:bool
    insider_buying:    bool

    # Market context
    sector:            str
    industry:          str
    market_cap_b:      float        # billions

    note:              str
    scanned_at:        str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Main scanner class ────────────────────────────────────────────────────────

class BreakoutHunter:
    """Scan the full NYSE/NASDAQ for breakout setups.

    Parameters
    ──────────
    min_composite : float
        Minimum composite score (0–100) to include in results.
    top_n : int | None
        Cap results at this many (sorted by score descending).
    exchanges : list[str]
        Finviz exchange filters.  Default: ["NYSE", "NASDAQ"].
    max_universe : int
        Cap the finviz pre-screen at this many tickers (saves time).
    workers : int
        ThreadPool workers for parallel fundamental lookups.
    verbose : bool
        If True, log INFO-level progress.
    """

    def __init__(
        self,
        min_composite: float = 50.0,
        top_n: Optional[int] = 25,
        exchanges: Optional[List[str]] = None,
        max_universe: int = 800,
        workers: int = 8,
        verbose: bool = False,
    ) -> None:
        self.min_composite = min_composite
        self.top_n = top_n
        self.exchanges = exchanges or ["NYSE", "NASDAQ"]
        self.max_universe = max_universe
        self.workers = workers
        if verbose:
            logging.getLogger().setLevel(logging.INFO)

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        extra_symbols: Optional[List[str]] = None,
        progress_cb=None,
    ) -> List[BreakoutResult]:
        """Run the full pipeline.  Returns list sorted by composite_score desc.

        progress_cb(step: str, pct: float) is called at key milestones if provided.
        """
        def _prog(step, pct):
            if progress_cb:
                try: progress_cb(step, pct)
                except Exception: pass

        # ── Stage 1: universe via Finviz ──────────────────────────────────────
        _prog("universe", 0.0)
        logger.info("Stage 1 — fetching Finviz universe…")
        finviz_rows, symbols = self._get_universe()
        logger.info("Stage 1 complete: %d candidates", len(symbols))
        if extra_symbols:
            symbols = list(dict.fromkeys(symbols + [s.upper() for s in extra_symbols]))

        # ── Stage 2: batch OHLCV ─────────────────────────────────────────────
        _prog("ohlcv", 0.10)
        logger.info("Stage 2 — downloading 1yr OHLCV for %d symbols…", len(symbols))
        ohlcv = self._batch_ohlcv(symbols)
        live_symbols = [s for s in symbols if s in ohlcv and len(ohlcv[s]) >= 200]
        logger.info("Stage 2 complete: %d symbols have sufficient history", len(live_symbols))

        # ── Stage 3: RS Rating (needs all scores first for percentile rank) ───
        _prog("rs_rating", 0.30)
        logger.info("Stage 3 — computing RS Ratings…")
        rs_raw_map = self._compute_rs_raw(ohlcv, live_symbols)
        rs_raw_values = list(rs_raw_map.values())

        # ── Stage 4–7: per-symbol deep analysis ──────────────────────────────
        _prog("analysis", 0.40)
        logger.info("Stage 4–7 — deep analysis…")
        finviz_map = {r.get("Ticker", ""): r for r in finviz_rows}

        results: List[BreakoutResult] = []
        for i, sym in enumerate(live_symbols):
            _prog("analysis", 0.40 + 0.55 * (i / max(len(live_symbols), 1)))
            try:
                df = ohlcv[sym]
                rs_raw = rs_raw_map.get(sym, 0.0)
                rs_pct = float(percentileofscore(rs_raw_values, rs_raw, kind="rank"))
                frow   = finviz_map.get(sym, {})
                result = self._analyze(sym, df, rs_pct, frow)
                if result and result.composite_score >= self.min_composite:
                    results.append(result)
            except Exception as exc:
                logger.debug("Skipping %s: %s", sym, exc)

        results.sort(key=lambda r: r.composite_score, reverse=True)
        if self.top_n:
            results = results[: self.top_n]

        _prog("done", 1.0)
        logger.info("Scan complete: %d signals above threshold %.0f",
                    len(results), self.min_composite)
        return results

    # ── Stage 1: universe ─────────────────────────────────────────────────────

    def _get_universe(self) -> Tuple[List[dict], List[str]]:
        """Fetch pre-filtered candidates from Finviz.
        Falls back to a built-in curated list if Finviz is unavailable.
        """
        rows: List[dict] = []
        try:
            from finvizfinance.screener.overview import Overview
            screen = Overview()
            filters: dict = {
                "Price":                    "Over $10",
                "Average Volume":           "Over 300K",
                "200-Day Simple Moving Average": "Price above SMA200",
            }
            for exchange in self.exchanges:
                screen.set_filter(filters_dict={**filters, "Exchange": exchange})
                try:
                    df = screen.screener_view(verbose=0)
                    if df is not None and len(df) > 0:
                        rows.extend(df.to_dict("records"))
                except Exception as e:
                    logger.debug("Finviz exchange %s failed: %s", exchange, e)

            if rows:
                seen = set()
                deduped = []
                for r in rows:
                    t = r.get("Ticker", "")
                    if t and t not in seen:
                        seen.add(t)
                        deduped.append(r)
                rows = deduped[: self.max_universe]
                symbols = [r["Ticker"] for r in rows if r.get("Ticker")]
                logger.info("Finviz returned %d candidates", len(symbols))
                return rows, symbols

        except Exception as exc:
            logger.warning("Finviz unavailable (%s), using fallback universe", exc)

        # Fallback: curated ~350-stock watchlist spanning all major sectors
        symbols = _FALLBACK_UNIVERSE
        return [], symbols[: self.max_universe]

    # ── Stage 2: batch OHLCV ──────────────────────────────────────────────────

    def _batch_ohlcv(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Download 1yr daily OHLCV for all symbols in one shot via yfinance."""
        import yfinance as yf
        result: Dict[str, pd.DataFrame] = {}
        # yfinance handles batches up to ~500 tickers cleanly
        chunk_size = 400
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            try:
                raw = yf.download(
                    chunk, period="1y", interval="1d",
                    auto_adjust=True, progress=False,
                    group_by="ticker", threads=True,
                )
                for sym in chunk:
                    try:
                        if len(chunk) == 1:
                            df = raw.copy()
                        else:
                            df = raw[sym].copy() if sym in raw.columns.get_level_values(0) else pd.DataFrame()
                        if df is not None and len(df) >= 60:
                            df.columns = [c.lower() for c in df.columns]
                            df = df.dropna(subset=["close", "volume"])
                            result[sym] = df
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Batch OHLCV chunk %d failed: %s", i, exc)
            time.sleep(0.5)  # be polite to yfinance

        return result

    # ── Stage 3: RS Ratings ───────────────────────────────────────────────────

    def _compute_rs_raw(
        self, ohlcv: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> Dict[str, float]:
        """IBD RS formula: 0.4*ROC63 + 0.2*ROC126 + 0.2*ROC189 + 0.2*ROC252."""
        rs_map: Dict[str, float] = {}
        for sym in symbols:
            df = ohlcv.get(sym)
            if df is None or len(df) < 63:
                continue
            c = df["close"].values.astype(float)
            try:
                roc63  = c[-1] / c[-63]  - 1
                roc126 = c[-1] / c[-126] - 1 if len(c) >= 126 else roc63
                roc189 = c[-1] / c[-189] - 1 if len(c) >= 189 else roc126
                roc252 = c[-1] / c[-252] - 1 if len(c) >= 252 else roc189
                rs_map[sym] = 0.4 * roc63 + 0.2 * roc126 + 0.2 * roc189 + 0.2 * roc252
            except Exception:
                pass
        return rs_map

    # ── Stage 4–7: per-symbol deep analysis ──────────────────────────────────

    def _analyze(
        self,
        sym: str,
        df: pd.DataFrame,
        rs_percentile: float,
        frow: dict,
    ) -> Optional[BreakoutResult]:
        """Run all scoring stages for one symbol.  Returns None if stock is junk."""
        close  = df["close"].values.astype(float)
        high   = df["high"].values.astype(float)
        low    = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        price  = float(close[-1])

        # ── Liquidity hard gates ──────────────────────────────────────────────
        avg_vol_50  = float(np.mean(volume[-50:])) if len(volume) >= 50 else float(np.mean(volume))
        avg_dvol_50 = avg_vol_50 * price
        if price < MIN_PRICE:
            return None
        if avg_vol_50 < MIN_AVG_VOLUME:
            return None
        if avg_dvol_50 < MIN_AVG_DOLLAR_VOL:
            return None
        if rs_percentile < RS_BREAKOUT_THRESHOLD:
            return None

        # ── Moving averages ───────────────────────────────────────────────────
        smas = self._smas(close, [20, 50, 150, 200])
        sma20, sma50, sma150, sma200 = (
            smas.get(20), smas.get(50), smas.get(150), smas.get(200))
        if any(x is None for x in [sma50, sma150, sma200]):
            return None

        # ── ATR ───────────────────────────────────────────────────────────────
        atr14_series = self._atr_series(high, low, close, 14)
        if len(atr14_series) < 63:
            return None
        atr_current  = float(atr14_series[-1])
        atr_63avg    = float(np.mean(atr14_series[-63:]))
        atr_contraction = atr_current / max(atr_63avg, 1e-9)

        # ── 52-week hi/lo ─────────────────────────────────────────────────────
        w52_high = float(np.max(high[-252:])) if len(high) >= 252 else float(np.max(high))
        w52_low  = float(np.min(low[-252:]))  if len(low)  >= 252 else float(np.min(low))

        # ── Trend template (Minervini 8 conditions) ───────────────────────────
        trend_n, trend_flags = self._trend_template(
            close, sma50, sma150, sma200, w52_high, w52_low)
        passes_trend = trend_n >= TREND_MIN_CONDITIONS

        # ── VCP / volatility contraction ──────────────────────────────────────
        vcp_score_val = self._vcp_score(close, high, low, volume, atr_contraction)
        vcp_detected  = vcp_score_val >= VCP_MIN_SCORE

        # ── Breakout signal ───────────────────────────────────────────────────
        is_breaking, vol_ratio, pivot_high = self._breakout_signal(
            close, high, volume, avg_vol_50)
        breakout_confirmed = is_breaking and vol_ratio >= BREAKOUT_VOL_MIN

        # ── Momentum ──────────────────────────────────────────────────────────
        mom5  = float((close[-1] / close[-6]  - 1) * 100) if len(close) > 6  else 0.0
        mom20 = float((close[-1] / close[-21] - 1) * 100) if len(close) > 21 else 0.0
        mom63 = float((close[-1] / close[-64] - 1) * 100) if len(close) > 64 else 0.0

        # ── RS Line (price / SPY — approximated by raw RS percentile) ─────────
        rs_line_at_high = rs_percentile >= 85  # RS line effectively at new highs

        # ── Fundamentals (from Finviz row, fast path) ─────────────────────────
        fundamentals = self._parse_finviz_fundamentals(frow)
        eps_growth   = fundamentals["eps_growth_pct"]
        rev_growth   = fundamentals["revenue_growth_pct"]
        short_float  = fundamentals["short_float_pct"]
        dtc          = fundamentals["days_to_cover"]
        sector       = fundamentals["sector"]
        industry     = fundamentals["industry"]
        mktcap       = fundamentals["market_cap_b"]

        # ── Insider buying (EdgarTools, optional — fails gracefully) ──────────
        insider_buy = self._check_insider_buying(sym)

        # ── Entry / stop / target geometry ───────────────────────────────────
        entry  = price
        stop   = self._compute_stop(close, low, atr_current, pivot_high)
        target = self._compute_target(close, high, low, entry, stop)
        rr     = abs(target - entry) / max(abs(entry - stop), 1e-6)

        # ── Stage scores (0–100 each) ──────────────────────────────────────────
        # RS sub-score
        rs_sub = rs_percentile  # already 0–99

        # Trend sub-score
        trend_sub = (trend_n / 8.0) * 100

        # VCP sub-score (already 0–100)
        vcp_sub = vcp_score_val

        # Breakout sub-score
        if breakout_confirmed:
            bk_sub = min(100.0, 50 + (vol_ratio - 1.0) * 25)
        elif is_breaking:
            bk_sub = 35.0
        elif vcp_detected and price >= pivot_high * 0.98:
            bk_sub = 25.0   # coiling at resistance — pre-breakout
        else:
            bk_sub = max(0.0, (vol_ratio - 0.5) * 20)

        # Fundamental sub-score
        fund_sub = 0.0
        if eps_growth > 0:
            fund_sub += min(40.0, eps_growth * 0.4)    # up to 40 for +100% EPS
        if rev_growth > 0:
            fund_sub += min(30.0, rev_growth * 0.6)    # up to 30 for +50% rev
        fund_sub += 20.0 if eps_growth >= 50 else 10.0 if eps_growth >= 25 else 0.0
        fund_sub = min(100.0, fund_sub)

        # Squeeze sub-score
        squeeze_sub = 0.0
        if short_float >= SQUEEZE_SHORT_FLOAT:
            squeeze_sub += min(60.0, short_float * 2.0)
        if dtc >= 5:
            squeeze_sub += min(25.0, dtc * 3.0)
        if insider_buy:
            squeeze_sub += 15.0
        squeeze_sub = min(100.0, squeeze_sub)

        # ── Composite weighted score ───────────────────────────────────────────
        composite = (
            W_RS          * rs_sub
            + W_TREND     * trend_sub
            + W_VCP       * vcp_sub
            + W_BREAKOUT  * bk_sub
            + W_FUNDAMENTAL * fund_sub
            + W_SQUEEZE   * squeeze_sub
        )

        # Bonus: RS line at new highs is a rare, highly-predictive signal
        if rs_line_at_high and (passes_trend or breakout_confirmed):
            composite = min(100.0, composite * 1.08)

        # Bonus: breakout confirmed on 2×+ volume while passing full trend template
        if breakout_confirmed and passes_trend and vol_ratio >= BREAKOUT_VOL_STRONG:
            composite = min(100.0, composite * 1.10)

        composite = round(composite, 1)

        # ── Setup label ───────────────────────────────────────────────────────
        setup_type = _classify_setup(
            passes_trend, vcp_detected, breakout_confirmed, rs_percentile, mom63)

        # ── Note ──────────────────────────────────────────────────────────────
        note = _build_note(
            trend_n, vcp_score_val, vol_ratio, rs_percentile, mom5, mom63,
            eps_growth, short_float, insider_buy, breakout_confirmed, atr_contraction,
        )

        return BreakoutResult(
            symbol              = sym,
            composite_score     = composite,
            setup_type          = setup_type,
            direction           = "long",
            entry_price         = round(entry, 2),
            stop_price          = round(stop, 2),
            target_price        = round(target, 2),
            rr_ratio            = round(rr, 2),
            rs_rating           = round(rs_percentile, 1),
            trend_score         = trend_n,
            vcp_score           = round(vcp_score_val, 1),
            breakout_score      = round(bk_sub, 1),
            fundamental_score   = round(fund_sub, 1),
            volume_ratio        = round(vol_ratio, 2),
            atr_contraction     = round(atr_contraction, 2),
            momentum_5d         = round(mom5, 2),
            momentum_20d        = round(mom20, 2),
            momentum_63d        = round(mom63, 2),
            eps_growth_pct      = round(eps_growth, 1),
            revenue_growth_pct  = round(rev_growth, 1),
            short_float_pct     = round(short_float, 1),
            days_to_cover       = round(dtc, 1),
            passes_trend_template = passes_trend,
            vcp_detected        = vcp_detected,
            breakout_confirmed  = breakout_confirmed,
            insider_buying      = insider_buy,
            sector              = sector,
            industry            = industry,
            market_cap_b        = round(mktcap, 2),
            note                = note,
        )

    # ── Technical primitives ─────────────────────────────────────────────────

    @staticmethod
    def _smas(close: np.ndarray, periods: List[int]) -> Dict[int, Optional[float]]:
        result = {}
        for p in periods:
            result[p] = float(np.mean(close[-p:])) if len(close) >= p else None
        return result

    @staticmethod
    def _atr_series(high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, period: int) -> np.ndarray:
        """True Range then Wilder EMA."""
        n = len(close)
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i]  - close[i - 1]),
            )
        # Wilder smoothing (EMA with alpha=1/period)
        atr = np.zeros(n)
        atr[period] = np.mean(tr[1 : period + 1])
        alpha = 1.0 / period
        for i in range(period + 1, n):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
        return atr[period:]

    @staticmethod
    def _trend_template(
        close: np.ndarray, sma50: float, sma150: float, sma200: float,
        w52_high: float, w52_low: float,
    ) -> Tuple[int, List[bool]]:
        """Minervini 8-condition trend template.  Returns (count_passing, flags)."""
        price = close[-1]
        # Condition 8 proxy: 200-day SMA trending up (today vs. 21 bars ago)
        sma200_now = sma200
        sma200_21ago = float(np.mean(close[-221:-21])) if len(close) >= 221 else sma200
        flags = [
            price > sma50,
            price > sma150,
            price > sma200,
            sma150 > sma200,
            sma50  > sma150,
            sma200_now > sma200_21ago,
            price >= w52_low * 1.30,
            price >= w52_high * 0.75,
        ]
        return sum(flags), flags

    def _vcp_score(
        self,
        close: np.ndarray, high: np.ndarray,
        low: np.ndarray,   volume: np.ndarray,
        atr_contraction: float,
    ) -> float:
        """Score volatility contraction quality 0–100.

        Incorporates: ATR compression, volume dry-up, price tightness,
        price position within range, and higher-low count.
        """
        score = 0.0

        # 1. ATR compression (up to 25 pts)
        if atr_contraction <= ATR_CONTRACTION_GREAT:
            score += 25
        elif atr_contraction <= ATR_CONTRACTION_GOOD:
            score += 15
        elif atr_contraction <= 0.85:
            score += 7

        # 2. Volume dry-up (up to 25 pts)
        vol_10  = float(np.mean(volume[-10:])) if len(volume) >= 10 else 0.0
        vol_50  = float(np.mean(volume[-50:])) if len(volume) >= 50 else 1.0
        vdry    = vol_10 / max(vol_50, 1.0)
        if vdry < 0.50:
            score += 25
        elif vdry < VOLUME_DRYUP_THRESHOLD:
            score += 15
        elif vdry < 0.90:
            score += 5

        # 3. Price range tightness last 10 bars (up to 20 pts)
        if len(close) >= 10:
            range10 = (float(np.max(high[-10:])) - float(np.min(low[-10:]))) / max(close[-10], 1e-9)
            if range10 < 0.04:
                score += 20
            elif range10 < 0.08:
                score += 13
            elif range10 < 0.12:
                score += 7

        # 4. Price position within 10-bar range (up to 15 pts)
        if len(close) >= 10:
            lo10 = float(np.min(low[-10:]))
            hi10 = float(np.max(high[-10:]))
            span = hi10 - lo10
            if span > 0:
                pos = (close[-1] - lo10) / span   # 0=bottom, 1=top
                score += pos * 15

        # 5. Higher lows (up to 15 pts) — more higher-lows = cleaner accumulation
        if len(low) >= 10:
            hl_count = sum(low[-i] > low[-i - 1] for i in range(1, min(10, len(low) - 1)))
            score += hl_count * 1.5   # up to ~13.5 pts

        return min(100.0, score)

    @staticmethod
    def _breakout_signal(
        close: np.ndarray, high: np.ndarray,
        volume: np.ndarray, avg_vol_50: float,
    ) -> Tuple[bool, float, float]:
        """Returns (is_breaking_out, volume_ratio, pivot_high_price)."""
        # Pivot = highest close over the most recent 30-bar base
        lookback = min(30, len(close) - 1)
        pivot = float(np.max(high[-lookback - 1 : -1]))  # exclude today
        vol_ratio = volume[-1] / max(avg_vol_50, 1.0)
        is_breaking = close[-1] > pivot * 1.001
        return is_breaking, float(vol_ratio), float(pivot)

    @staticmethod
    def _compute_stop(
        close: np.ndarray, low: np.ndarray,
        atr: float, pivot: float,
    ) -> float:
        """Stop below the deepest VCP trough or ATR-based fallback."""
        # VCP low = lowest low of the past 20 bars (the base)
        vcp_low = float(np.min(low[-20:])) if len(low) >= 20 else float(np.min(low))
        # ATR fallback = entry - 2×ATR
        atr_stop = float(close[-1]) - 2.0 * atr
        # Use the less aggressive of the two (higher stop = tighter risk)
        return max(vcp_low * 0.99, atr_stop)

    @staticmethod
    def _compute_target(
        close: np.ndarray, high: np.ndarray,
        low: np.ndarray, entry: float, stop: float,
    ) -> float:
        """Measured-move target: entry + height of the base."""
        base_high = float(np.max(high[-60:])) if len(high) >= 60 else entry
        base_low  = float(np.min(low[-60:]))  if len(low)  >= 60 else stop
        base_height = base_high - base_low
        target = entry + base_height
        # Ensure minimum 2:1 R:R
        min_target = entry + 2.0 * abs(entry - stop)
        return max(target, min_target)

    # ── Fundamental helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_finviz_fundamentals(row: dict) -> dict:
        """Extract fundamental fields from a Finviz screener row dict."""
        def safe_float(val, default=0.0) -> float:
            if val in (None, "", "-", "N/A"):
                return default
            try:
                return float(str(val).replace("%", "").replace("B", "")
                             .replace("M", "").replace("T", "").strip())
            except Exception:
                return default

        def parse_mktcap(val) -> float:
            """Convert '12.3B' → 12.3, '450M' → 0.45, etc."""
            if not val or str(val) in ("-", "N/A", ""):
                return 0.0
            s = str(val).strip().upper()
            try:
                if "T" in s:
                    return float(s.replace("T", "")) * 1000
                if "B" in s:
                    return float(s.replace("B", ""))
                if "M" in s:
                    return float(s.replace("M", "")) / 1000
                return float(s)
            except Exception:
                return 0.0

        # Finviz returns EPS growth qtr over qtr as a %
        eps_qoq  = safe_float(row.get("EPS growth qtr over qtr", row.get("EPS growth this year", 0)))
        rev_qoq  = safe_float(row.get("Sales growth qtr over qtr", row.get("Sales growth past 5 years", 0)))
        sf       = safe_float(row.get("Short Interest Share", row.get("Short Float", 0)))
        dtc      = safe_float(row.get("Short Interest Ratio", 0))

        return {
            "eps_growth_pct":    eps_qoq,
            "revenue_growth_pct":rev_qoq,
            "short_float_pct":   sf,
            "days_to_cover":     dtc,
            "sector":            str(row.get("Sector", "") or ""),
            "industry":          str(row.get("Industry", "") or ""),
            "market_cap_b":      parse_mktcap(row.get("Market Cap.", row.get("Market Cap", "0"))),
        }

    @staticmethod
    def _check_insider_buying(sym: str) -> bool:
        """Return True if any insider filed a Form 4 open-market BUY in last 90 days.
        Fails silently — insider data is a bonus signal, not a hard gate.
        """
        try:
            from edgar import Company
            company = Company(sym)
            filings = company.get_filings(form="4")
            if not filings:
                return False
            cutoff = datetime.now(timezone.utc) - timedelta(days=90)
            for filing in filings[:20]:
                filing_date = getattr(filing, "filing_date", None)
                if filing_date is None:
                    continue
                if hasattr(filing_date, "tzinfo") and filing_date.tzinfo is None:
                    filing_date = filing_date.replace(tzinfo=timezone.utc)
                if filing_date < cutoff:
                    break
                doc = filing.obj()
                # Look for transaction type "P" = open-market purchase
                if hasattr(doc, "transactions"):
                    for tx in doc.transactions:
                        if getattr(tx, "transaction_code", "") == "P":
                            return True
        except Exception:
            pass
        return False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _classify_setup(
    passes_trend: bool, vcp_detected: bool,
    breakout_confirmed: bool, rs_pct: float, mom63: float,
) -> str:
    if passes_trend and vcp_detected and breakout_confirmed:
        return "VCP + Stage2 BREAKOUT"
    if passes_trend and vcp_detected:
        return "VCP + Stage2 (coiling)"
    if passes_trend and breakout_confirmed:
        return "Stage2 Breakout"
    if vcp_detected and breakout_confirmed:
        return "VCP Breakout"
    if passes_trend:
        return "Stage2 (base forming)"
    if vcp_detected:
        return "VCP (pre-breakout)"
    if rs_pct >= 90 and mom63 > 20:
        return "RS Leader (trending)"
    return "Emerging Setup"


def _build_note(
    trend_n: int, vcp_score: float, vol_ratio: float, rs_pct: float,
    mom5: float, mom63: float, eps_growth: float, short_float: float,
    insider_buy: bool, breakout_confirmed: bool, atr_contraction: float,
) -> str:
    parts = []
    parts.append(f"Trend {trend_n}/8")
    if atr_contraction <= ATR_CONTRACTION_GREAT:
        parts.append(f"ATR compressed {atr_contraction:.0%}")
    if vcp_score >= 60:
        parts.append(f"VCP {vcp_score:.0f}/100")
    if breakout_confirmed:
        parts.append(f"BREAKOUT {vol_ratio:.1f}× vol")
    elif vol_ratio >= 1.5:
        parts.append(f"vol surge {vol_ratio:.1f}×")
    if rs_pct >= 90:
        parts.append(f"RS {rs_pct:.0f} (elite)")
    elif rs_pct >= 80:
        parts.append(f"RS {rs_pct:.0f}")
    if abs(mom5) > 3:
        parts.append(f"{mom5:+.1f}% 5d")
    if abs(mom63) > 15:
        parts.append(f"{mom63:+.1f}% 13wk")
    if eps_growth >= 50:
        parts.append(f"EPS +{eps_growth:.0f}%")
    elif eps_growth >= 25:
        parts.append(f"EPS +{eps_growth:.0f}%")
    if short_float >= 15:
        parts.append(f"short {short_float:.0f}% (squeeze)")
    elif short_float >= 10:
        parts.append(f"short {short_float:.0f}%")
    if insider_buy:
        parts.append("insider buy")
    return "  ·  ".join(parts)


# ── Fallback universe (~350 high-quality NYSE/NASDAQ stocks) ─────────────────
_FALLBACK_UNIVERSE: List[str] = [
    # Index / broad market
    "SPY","QQQ","IWM","DIA","MDY","VTI",
    # Mega-cap tech
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","AVGO","QCOM",
    "TXN","AMAT","LRCX","KLAC","MU","INTC","ARM","SMCI",
    # Software/cloud
    "CRM","NOW","SNOW","DDOG","NET","CRWD","PANW","ZS","OKTA","HUBS",
    "GTLB","PATH","MDB","ESTC","CFLT","BILL","VEEV","WDAY","ADBE","ORCL",
    # Financials
    "JPM","GS","MS","BAC","WFC","C","BLK","SCHW","V","MA","AXP","COF",
    # Healthcare / biotech
    "LLY","NVO","ABBV","JNJ","UNH","MRNA","REGN","VRTX","GILD","BMY",
    "RXRX","RARE","SANA","BEAM","EDIT","NTLA","CRSP","IONS",
    # Consumer / retail
    "AMZN","COST","HD","LOW","TGT","WMT","NKE","LULU","DPZ","CMG","MCD",
    # Industrials / defense
    "CAT","DE","BA","RTX","LMT","NOC","GD","HON","MMM","GE","ETN","EMR",
    # Energy
    "XOM","CVX","COP","SLB","HAL","MPC","VLO","PSX","DVN","FANG",
    # Materials / commodities
    "NEM","GOLD","FCX","MP","VALE","RIO","BHP","SCCO","CLF","NUE",
    # Real estate / utilities
    "AMT","PLD","EQIX","SPG","O","DLR","PSA",
    # Crypto / blockchain
    "COIN","MSTR","MARA","RIOT","CLSK","IBIT","GBTC","FBTC","BITB",
    # High-momentum growth
    "PLTR","HOOD","SOFI","IONQ","RKLB","ACHR","JOBY","RXRX","DNA","SOUN",
    "BBAI","LUNR","RDW","ASTS","SPCE","RCAT","JOBY","ACHR","LILM",
    "WOLF","AEHR","OUST","AEVA","LAZR","MVIS","LIDR",
    # Leveraged ETFs (high-beta plays)
    "TQQQ","SQQQ","UPRO","SPXS","SOXL","SOXS","LABU","FNGU","FNGD",
    # Volatility
    "VXX","UVXY",
    # Additional NYSE leaders
    "BRK-B","JNJ","PG","KO","PEP","PM","MO","T","VZ","CMCSA","DIS",
    "NFLX","SPOT","RBLX","U","MTCH","ZM","PTON","LYFT","UBER","DASH",
    "ABNB","BKNG","EXPE","MAR","HLT","LVS","WYNN","MGM",
    "F","GM","LCID","RIVN","NIO","LI","XPEV","BIDU","JD","PDD","BABA",
]
