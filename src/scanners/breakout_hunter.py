"""NYSE/NASDAQ Breakout Hunter — Institutional-Grade Setup Scanner

Built on THIRTEEN independently-validated methodologies:
  • Minervini Trend Template (8 conditions) + VCP pattern
  • Stan Weinstein Stage 2 breakout system
  • IBD Relative Strength Rating (exact 0.4/0.2/0.2/0.2 formula)
  • RS Line New High — price/SPY ratio at 52-week high (rare elite signal)
  • O'Neil CANSLIM — EPS acceleration pattern (25→50→80% rocket)
  • Wyckoff OBV divergence — silent institutional accumulation footprint
  • Pocket Pivot (Morales/Kacher) — volume signature before the breakout
  • Sector Rotation — sector ETF above 30-week SMA as macro tailwind
  • Earnings Proximity — setup coiling 10–30 days before catalysts
  • Options Flow — unusual OTM call sweeps (smart-money fingerprint)
  • PEAD (Post-Earnings Announcement Drift) fundamental overlay
  • Short squeeze mechanics — float short % + days-to-cover
  • SEC Form 4 insider open-market buying (EdgarTools)

Pipeline
────────
  Stage 1   Universe filter     Liquidity, price, basic trend (Finviz)
  Stage 2   OHLCV download      SPY + sector ETFs downloaded alongside universe
  Stage 3   RS Rating           IBD formula, percentile-ranked vs. universe
  Stage 4   Trend template      Minervini 8-condition SMA stack + 52wk range
  Stage 5   VCP detection       ATR contraction, volume dry-up, tight closes, higher lows
  Stage 6   Breakout trigger    Price vs. pivot high, volume surge confirmation
  Stage 7   RS Line             price/SPY ratio at or near 52-week high
  Stage 8   OBV divergence      On-Balance Volume rising while price consolidates
  Stage 9   Pocket Pivot        Up-day volume > highest prior-10-session down-day volume
  Stage 10  Sector rotation     Sector ETF above 30-week SMA + trending
  Stage 11  Earnings accel.     EPS growth rate increasing quarter over quarter
  Stage 12  Earnings proximity  Days to next earnings (optional API call)
  Stage 13  Options flow        Unusual OTM call volume/OI (optional API call)
  Stage 14  Composite scoring   Weighted 0–100 score → final ranking

BreakoutResult — all fields
───────────────────────────
  symbol, composite_score, setup_type, direction
  entry_price, stop_price, target_price, rr_ratio
  rs_rating, trend_score, vcp_score, breakout_score, fundamental_score
  catalyst_score, accumulation_score
  volume_ratio, atr_contraction
  momentum_5d, momentum_20d, momentum_63d
  eps_growth_pct, revenue_growth_pct, short_float_pct, days_to_cover
  passes_trend_template, vcp_detected, breakout_confirmed, insider_buying
  rs_line_new_high, eps_acceleration, pocket_pivot
  sector_aligned, earnings_proximity_days, obv_divergence, options_unusual
  sector, industry, market_cap_b, note
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

logger = logging.getLogger(__name__)

# ── yfinance column normaliser ────────────────────────────────────────────────
# yfinance ≥ 0.2 returns MultiIndex tuples: (field, ticker) for single-symbol
# downloads, or (ticker, field) for group_by='ticker' single-item batches.
# Find the OHLCV field part regardless of which position it occupies.
_OHLCV_FIELDS = frozenset(("open", "high", "low", "close", "volume", "adj close"))


def _flatten_yf_columns(cols) -> list:
    result = []
    for c in cols:
        if isinstance(c, str):
            result.append(c.lower())
        else:
            field = next((p.lower() for p in c if p.lower() in _OHLCV_FIELDS), None)
            result.append(field if field else c[0].lower())
    return result


# ── Scoring weights (must sum to 1.0) ────────────────────────────────────────
W_RS           = 0.18   # IBD RS Rating percentile
W_TREND        = 0.16   # Minervini trend template
W_VCP          = 0.16   # VCP / volatility contraction
W_BREAKOUT     = 0.12   # Breakout trigger (volume + price vs pivot)
W_FUNDAMENTAL  = 0.10   # EPS + revenue acceleration
W_SQUEEZE      = 0.08   # Short float + insider buying
W_CATALYST     = 0.09   # Pocket pivot + options flow + earnings proximity
W_ACCUMULATION = 0.07   # OBV divergence + sector rotation
W_RS_LINE      = 0.04   # RS line at 52-week high (bonus multiplier too)
# Total = 1.00

# ── Thresholds ───────────────────────────────────────────────────────────────
MIN_PRICE              = 10.0
MIN_AVG_VOLUME         = 300_000
MIN_AVG_DOLLAR_VOL     = 5_000_000
RS_BREAKOUT_THRESHOLD  = 70
VCP_MIN_SCORE          = 30
TREND_MIN_CONDITIONS   = 5
ATR_CONTRACTION_GOOD   = 0.70
ATR_CONTRACTION_GREAT  = 0.55
VOLUME_DRYUP_THRESHOLD = 0.80
BREAKOUT_VOL_MIN       = 1.40
BREAKOUT_VOL_STRONG    = 2.00
SQUEEZE_SHORT_FLOAT    = 10.0

# ── Sector → ETF mapping ─────────────────────────────────────────────────────
_SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financial":              "XLF",
    "Financials":             "XLF",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Communication Services": "XLC",
}


# ── Data class ───────────────────────────────────────────────────────────────

@dataclass
class BreakoutResult:
    symbol:            str
    composite_score:   float        # 0–100, primary ranking key
    setup_type:        str
    direction:         str          # "long"

    entry_price:       float
    stop_price:        float
    target_price:      float
    rr_ratio:          float

    # Stage sub-scores (0–100 each)
    rs_rating:         float        # 0–99 IBD RS percentile
    trend_score:       int          # 0–8 Minervini conditions
    vcp_score:         float        # 0–100
    breakout_score:    float        # 0–100
    fundamental_score: float        # 0–100
    catalyst_score:    float        # 0–100  (pocket pivot + options + earnings prox)
    accumulation_score:float        # 0–100  (OBV divergence + sector rotation)

    # Key metrics
    volume_ratio:       float
    atr_contraction:    float
    momentum_5d:        float
    momentum_20d:       float
    momentum_63d:       float

    # Fundamentals
    eps_growth_pct:     float
    revenue_growth_pct: float
    short_float_pct:    float
    days_to_cover:      float

    # Boolean flags (original 4)
    passes_trend_template: bool
    vcp_detected:       bool
    breakout_confirmed: bool
    insider_buying:     bool

    # Boolean flags (new 7 dimensions)
    rs_line_new_high:       bool    # price/SPY ratio at 52-week high
    eps_acceleration:       bool    # EPS growth rate accelerating
    pocket_pivot:           bool    # Morales/Kacher pocket pivot
    sector_aligned:         bool    # sector ETF above 30-week SMA + trending
    obv_divergence:         bool    # OBV rising while price consolidates
    options_unusual:        bool    # unusual OTM call sweeps
    earnings_proximity_days:int     # days to next earnings (-1 = unknown)

    # Market context
    sector:            str
    industry:          str
    market_cap_b:      float

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
        Minimum composite score (0–100) to include.
    top_n : int | None
        Cap results at this many (sorted by score desc).
    exchanges : list[str]
        Finviz exchange filters.  Default: ["NYSE","NASDAQ"].
    max_universe : int
        Cap the Finviz pre-screen at this many tickers.
    workers : int
        ThreadPool size (reserved for future async expansion).
    verbose : bool
        Log INFO-level progress.
    enable_options : bool
        Run per-symbol yfinance options chain lookups (slow, ~1s each).
        Off by default.  Best enabled for <50 final candidates.
    enable_earnings_calendar : bool
        Fetch next-earnings date from yfinance calendar per symbol (slow).
        Off by default.
    """

    def __init__(
        self,
        min_composite:            float = 50.0,
        top_n:                    Optional[int] = 25,
        exchanges:                Optional[List[str]] = None,
        max_universe:             int = 800,
        workers:                  int = 8,
        verbose:                  bool = False,
        enable_options:           bool = False,
        enable_earnings_calendar: bool = False,
    ) -> None:
        self.min_composite            = min_composite
        self.top_n                    = top_n
        self.exchanges                = exchanges or ["NYSE", "NASDAQ"]
        self.max_universe             = max_universe
        self.workers                  = workers
        self.enable_options           = enable_options
        self.enable_earnings_calendar = enable_earnings_calendar
        if verbose:
            logging.getLogger().setLevel(logging.INFO)

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        extra_symbols: Optional[List[str]] = None,
        progress_cb=None,
    ) -> List[BreakoutResult]:
        """Run the full 14-stage pipeline.  Returns list sorted by composite_score desc."""
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

        # ── Stage 2: batch OHLCV (universe + SPY + sector ETFs) ──────────────
        _prog("ohlcv", 0.10)
        logger.info("Stage 2 — downloading 1yr OHLCV for %d symbols…", len(symbols))
        # Append benchmark + sector ETFs to download in same batch
        sector_etfs  = list(set(_SECTOR_ETF_MAP.values()))
        dl_symbols   = list(dict.fromkeys(symbols + ["SPY"] + sector_etfs))
        ohlcv        = self._batch_ohlcv(dl_symbols)

        # Benchmark (SPY) close array
        spy_df    = ohlcv.get("SPY")
        spy_close = spy_df["close"].values.astype(float) if (
            spy_df is not None and len(spy_df) >= 63
        ) else None

        # Sector ETF OHLCV index
        sector_etf_ohlcv: Dict[str, pd.DataFrame] = {
            etf: ohlcv[etf]
            for etf in sector_etfs
            if etf in ohlcv and len(ohlcv[etf]) >= 150
        }

        # Only score symbols with sufficient history
        live_symbols = [s for s in symbols if s in ohlcv and len(ohlcv[s]) >= 200]
        logger.info("Stage 2 complete: %d symbols have sufficient history", len(live_symbols))

        # ── Stage 3: RS Rating (needs full population for percentile rank) ────
        _prog("rs_rating", 0.30)
        logger.info("Stage 3 — computing RS Ratings…")
        rs_raw_map    = self._compute_rs_raw(ohlcv, live_symbols)
        rs_raw_values = list(rs_raw_map.values())

        # ── Stages 4–14: per-symbol deep analysis ─────────────────────────────
        _prog("analysis", 0.40)
        logger.info("Stages 4–14 — deep analysis of %d symbols…", len(live_symbols))
        finviz_map = {r.get("Ticker", ""): r for r in finviz_rows}

        results: List[BreakoutResult] = []
        for i, sym in enumerate(live_symbols):
            _prog("analysis", 0.40 + 0.55 * (i / max(len(live_symbols), 1)))
            try:
                df         = ohlcv[sym]
                rs_raw     = rs_raw_map.get(sym, 0.0)
                rs_pct     = float(percentileofscore(rs_raw_values, rs_raw, kind="rank"))
                frow       = finviz_map.get(sym, {})
                result     = self._analyze(
                    sym, df, rs_pct, frow,
                    spy_close=spy_close,
                    sector_etf_ohlcv=sector_etf_ohlcv,
                )
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
        rows: List[dict] = []
        try:
            from finvizfinance.screener.overview import Overview
            screen  = Overview()
            filters = {
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
                seen, deduped = set(), []
                for r in rows:
                    t = r.get("Ticker", "")
                    if t and t not in seen:
                        seen.add(t); deduped.append(r)
                rows    = deduped[: self.max_universe]
                symbols = [r["Ticker"] for r in rows if r.get("Ticker")]
                logger.info("Finviz returned %d candidates", len(symbols))
                return rows, symbols

        except Exception as exc:
            logger.warning("Finviz unavailable (%s), using fallback universe", exc)

        symbols = _FALLBACK_UNIVERSE
        return [], symbols[: self.max_universe]

    # ── Stage 2: batch OHLCV ──────────────────────────────────────────────────

    def _batch_ohlcv(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        import yfinance as yf
        result: Dict[str, pd.DataFrame] = {}
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
                            df = (raw[sym].copy()
                                  if sym in raw.columns.get_level_values(0)
                                  else pd.DataFrame())
                        if df is not None and len(df) >= 60:
                            df.columns = _flatten_yf_columns(df.columns)
                            df = df.dropna(subset=["close", "volume"])
                            result[sym] = df
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Batch OHLCV chunk %d failed: %s", i, exc)
            time.sleep(0.5)
        return result

    # ── Stage 3: RS Ratings ───────────────────────────────────────────────────

    def _compute_rs_raw(
        self, ohlcv: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> Dict[str, float]:
        """IBD RS formula: 0.4×ROC63 + 0.2×ROC126 + 0.2×ROC189 + 0.2×ROC252."""
        rs_map: Dict[str, float] = {}
        for sym in symbols:
            df = ohlcv.get(sym)
            if df is None or len(df) < 63:
                continue
            c = df["close"].values.astype(float)
            try:
                r63  = c[-1] / c[-63]  - 1
                r126 = c[-1] / c[-126] - 1 if len(c) >= 126 else r63
                r189 = c[-1] / c[-189] - 1 if len(c) >= 189 else r126
                r252 = c[-1] / c[-252] - 1 if len(c) >= 252 else r189
                rs_map[sym] = 0.4*r63 + 0.2*r126 + 0.2*r189 + 0.2*r252
            except Exception:
                pass
        return rs_map

    # ── Stages 4–14: per-symbol deep analysis ────────────────────────────────

    def _analyze(
        self,
        sym:              str,
        df:               pd.DataFrame,
        rs_percentile:    float,
        frow:             dict,
        spy_close:        Optional[np.ndarray] = None,
        sector_etf_ohlcv: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Optional[BreakoutResult]:
        """Run all scoring stages for one symbol. Returns None if stock fails gates."""
        close  = df["close"].values.astype(float)
        high   = df["high"].values.astype(float)
        low    = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        price  = float(close[-1])

        # ── Hard liquidity gates ──────────────────────────────────────────────
        avg_vol_50  = float(np.mean(volume[-50:])) if len(volume) >= 50 else float(np.mean(volume))
        avg_dvol_50 = avg_vol_50 * price
        if price < MIN_PRICE:          return None
        if avg_vol_50 < MIN_AVG_VOLUME: return None
        if avg_dvol_50 < MIN_AVG_DOLLAR_VOL: return None
        if rs_percentile < RS_BREAKOUT_THRESHOLD: return None

        # ── Moving averages ───────────────────────────────────────────────────
        smas = self._smas(close, [20, 50, 150, 200])
        sma20, sma50, sma150, sma200 = (
            smas.get(20), smas.get(50), smas.get(150), smas.get(200))
        if any(x is None for x in [sma50, sma150, sma200]):
            return None

        # ── ATR ───────────────────────────────────────────────────────────────
        atr_series = self._atr_series(high, low, close, 14)
        if len(atr_series) < 63:
            return None
        atr_current     = float(atr_series[-1])
        atr_63avg       = float(np.mean(atr_series[-63:]))
        atr_contraction = atr_current / max(atr_63avg, 1e-9)

        # ── 52-week hi/lo ─────────────────────────────────────────────────────
        w52_high = float(np.max(high[-252:])) if len(high) >= 252 else float(np.max(high))
        w52_low  = float(np.min(low[-252:]))  if len(low)  >= 252 else float(np.min(low))

        # ── Stage 4: Minervini trend template ────────────────────────────────
        trend_n, trend_flags = self._trend_template(
            close, sma50, sma150, sma200, w52_high, w52_low)
        passes_trend = trend_n >= TREND_MIN_CONDITIONS

        # ── Stage 5: VCP ──────────────────────────────────────────────────────
        vcp_score_val = self._vcp_score(close, high, low, volume, atr_contraction)
        vcp_detected  = vcp_score_val >= VCP_MIN_SCORE

        # ── Stage 6: Breakout trigger ─────────────────────────────────────────
        is_breaking, vol_ratio, pivot_high = self._breakout_signal(
            close, high, volume, avg_vol_50)
        breakout_confirmed = is_breaking and vol_ratio >= BREAKOUT_VOL_MIN

        # ── Momentum ──────────────────────────────────────────────────────────
        mom5  = float((close[-1]/close[-6]  - 1)*100) if len(close) > 6  else 0.0
        mom20 = float((close[-1]/close[-21] - 1)*100) if len(close) > 21 else 0.0
        mom63 = float((close[-1]/close[-64] - 1)*100) if len(close) > 64 else 0.0

        # ── Fundamentals (Finviz row) ──────────────────────────────────────────
        fundamentals = self._parse_finviz_fundamentals(frow)
        eps_growth   = fundamentals["eps_growth_pct"]
        rev_growth   = fundamentals["revenue_growth_pct"]
        short_float  = fundamentals["short_float_pct"]
        dtc          = fundamentals["days_to_cover"]
        sector       = fundamentals["sector"]
        industry     = fundamentals["industry"]
        mktcap       = fundamentals["market_cap_b"]
        eps_5yr      = fundamentals["eps_5yr_pct"]

        # ── Stage 7: RS Line new high ─────────────────────────────────────────
        rs_line_flag, rs_line_sub = self._rs_line_score(close, spy_close)

        # ── Stage 8: OBV divergence ───────────────────────────────────────────
        obv_div_flag, obv_score = self._obv_divergence(close, volume)

        # ── Stage 9: Pocket Pivot ─────────────────────────────────────────────
        pp_flag = self._pocket_pivot(close, volume)

        # ── Stage 10: Sector rotation ─────────────────────────────────────────
        sect_aligned, sect_score = self._sector_rotation_score(
            sector, sector_etf_ohlcv or {})

        # ── Stage 11: EPS acceleration ────────────────────────────────────────
        eps_accel_flag = self._earnings_acceleration(eps_growth, eps_5yr)

        # ── Stage 12: Earnings proximity (optional API call) ──────────────────
        earn_days = (
            self._earnings_proximity(sym)
            if self.enable_earnings_calendar else -1
        )

        # ── Stage 13: Options flow (optional API call) ────────────────────────
        opts_unusual, opts_score = (
            self._options_flow_score(sym)
            if self.enable_options else (False, 0.0)
        )

        # ── Insider buying (EdgarTools, bonus) ────────────────────────────────
        insider_buy = self._check_insider_buying(sym)

        # ── Entry / stop / target geometry ───────────────────────────────────
        entry  = price
        stop   = self._compute_stop(close, low, atr_current, pivot_high)
        target = self._compute_target(close, high, low, entry, stop)
        rr     = abs(target - entry) / max(abs(entry - stop), 1e-6)

        # ══════════════════════════════════════════════════════════════════════
        #  Sub-scores (each 0–100)
        # ══════════════════════════════════════════════════════════════════════

        # RS sub-score
        rs_sub = rs_percentile

        # Trend sub-score
        trend_sub = (trend_n / 8.0) * 100

        # VCP sub-score
        vcp_sub = vcp_score_val

        # Breakout sub-score
        if breakout_confirmed:
            bk_sub = min(100.0, 50 + (vol_ratio - 1.0) * 25)
        elif is_breaking:
            bk_sub = 35.0
        elif vcp_detected and price >= pivot_high * 0.98:
            bk_sub = 25.0
        else:
            bk_sub = max(0.0, (vol_ratio - 0.5) * 20)

        # Fundamental sub-score
        fund_sub = 0.0
        if eps_growth > 0:
            fund_sub += min(40.0, eps_growth * 0.4)
        if rev_growth > 0:
            fund_sub += min(30.0, rev_growth * 0.6)
        fund_sub += 20.0 if eps_growth >= 50 else 10.0 if eps_growth >= 25 else 0.0
        if eps_accel_flag:
            fund_sub = min(100.0, fund_sub + 10.0)   # acceleration bonus
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

        # Catalyst sub-score — pocket pivot, options, earnings proximity
        catalyst_sub = 0.0
        if pp_flag:
            catalyst_sub += 40.0
        if opts_unusual:
            catalyst_sub += min(40.0, opts_score)
        if 10 <= earn_days <= 45:
            # sweet spot: coiling 10-45 days before catalyst
            prox_score = 20.0 * (1.0 - abs(earn_days - 25) / 20.0)
            catalyst_sub += max(0.0, prox_score)
        catalyst_sub = min(100.0, catalyst_sub)

        # Accumulation sub-score — OBV divergence + sector rotation
        accum_sub = obv_score * 0.65 + sect_score * 0.35
        accum_sub = min(100.0, accum_sub)

        # RS line sub-score (already 0–100 from method)
        rs_line_sub_score = rs_line_sub

        # ── Composite weighted score ───────────────────────────────────────────
        composite = (
            W_RS           * rs_sub
            + W_TREND      * trend_sub
            + W_VCP        * vcp_sub
            + W_BREAKOUT   * bk_sub
            + W_FUNDAMENTAL* fund_sub
            + W_SQUEEZE    * squeeze_sub
            + W_CATALYST   * catalyst_sub
            + W_ACCUMULATION * accum_sub
            + W_RS_LINE    * rs_line_sub_score
        )

        # Bonus: RS line at new high — rare, elite signal
        if rs_line_flag and (passes_trend or breakout_confirmed):
            composite = min(100.0, composite * 1.08)

        # Bonus: confirmed breakout with full trend + strong volume
        if breakout_confirmed and passes_trend and vol_ratio >= BREAKOUT_VOL_STRONG:
            composite = min(100.0, composite * 1.10)

        # Bonus: pocket pivot inside a confirmed Stage2 base (pre-breakout sweet spot)
        if pp_flag and passes_trend and not breakout_confirmed:
            composite = min(100.0, composite * 1.05)

        composite = round(composite, 1)

        # ── Setup label + note ────────────────────────────────────────────────
        setup_type = _classify_setup(
            passes_trend, vcp_detected, breakout_confirmed,
            rs_percentile, mom63, pp_flag, rs_line_flag,
        )
        note = _build_note(
            trend_n, vcp_score_val, vol_ratio, rs_percentile, mom5, mom63,
            eps_growth, short_float, insider_buy, breakout_confirmed,
            atr_contraction, rs_line_flag, pp_flag, obv_div_flag,
            sect_aligned, earn_days, eps_accel_flag, opts_unusual,
        )

        return BreakoutResult(
            symbol               = sym,
            composite_score      = composite,
            setup_type           = setup_type,
            direction            = "long",
            entry_price          = round(entry,  2),
            stop_price           = round(stop,   2),
            target_price         = round(target, 2),
            rr_ratio             = round(rr,     2),
            rs_rating            = round(rs_percentile, 1),
            trend_score          = trend_n,
            vcp_score            = round(vcp_score_val, 1),
            breakout_score       = round(bk_sub,    1),
            fundamental_score    = round(fund_sub,  1),
            catalyst_score       = round(catalyst_sub, 1),
            accumulation_score   = round(accum_sub, 1),
            volume_ratio         = round(vol_ratio, 2),
            atr_contraction      = round(atr_contraction, 2),
            momentum_5d          = round(mom5,  2),
            momentum_20d         = round(mom20, 2),
            momentum_63d         = round(mom63, 2),
            eps_growth_pct       = round(eps_growth, 1),
            revenue_growth_pct   = round(rev_growth, 1),
            short_float_pct      = round(short_float, 1),
            days_to_cover        = round(dtc, 1),
            passes_trend_template= passes_trend,
            vcp_detected         = vcp_detected,
            breakout_confirmed   = breakout_confirmed,
            insider_buying       = insider_buy,
            rs_line_new_high     = rs_line_flag,
            eps_acceleration     = eps_accel_flag,
            pocket_pivot         = pp_flag,
            sector_aligned       = sect_aligned,
            obv_divergence       = obv_div_flag,
            options_unusual      = opts_unusual,
            earnings_proximity_days = earn_days,
            sector               = sector,
            industry             = industry,
            market_cap_b         = round(mktcap, 2),
            note                 = note,
        )

    # ── Technical primitives ─────────────────────────────────────────────────

    @staticmethod
    def _smas(close: np.ndarray, periods: List[int]) -> Dict[int, Optional[float]]:
        return {p: float(np.mean(close[-p:])) if len(close) >= p else None
                for p in periods}

    @staticmethod
    def _atr_series(high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, period: int) -> np.ndarray:
        n  = len(close)
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i]-low[i],
                        abs(high[i]-close[i-1]),
                        abs(low[i] -close[i-1]))
        atr       = np.zeros(n)
        atr[period] = np.mean(tr[1:period+1])
        alpha     = 1.0 / period
        for i in range(period+1, n):
            atr[i] = atr[i-1]*(1-alpha) + tr[i]*alpha
        return atr[period:]

    @staticmethod
    def _trend_template(
        close: np.ndarray, sma50: float, sma150: float, sma200: float,
        w52_high: float, w52_low: float,
    ) -> Tuple[int, List[bool]]:
        price      = close[-1]
        sma200_21ago = float(np.mean(close[-221:-21])) if len(close) >= 221 else sma200
        flags = [
            price > sma50,
            price > sma150,
            price > sma200,
            sma150 > sma200,
            sma50  > sma150,
            sma200 > sma200_21ago,
            price >= w52_low  * 1.30,
            price >= w52_high * 0.75,
        ]
        return sum(flags), flags

    def _vcp_score(
        self,
        close: np.ndarray, high: np.ndarray,
        low:   np.ndarray, volume: np.ndarray,
        atr_contraction: float,
    ) -> float:
        score = 0.0
        # 1. ATR compression (25 pts)
        if   atr_contraction <= ATR_CONTRACTION_GREAT: score += 25
        elif atr_contraction <= ATR_CONTRACTION_GOOD:  score += 15
        elif atr_contraction <= 0.85:                  score += 7
        # 2. Volume dry-up (25 pts)
        vol_10 = float(np.mean(volume[-10:])) if len(volume) >= 10 else 0.0
        vol_50 = float(np.mean(volume[-50:])) if len(volume) >= 50 else 1.0
        vdry   = vol_10 / max(vol_50, 1.0)
        if   vdry < 0.50:                   score += 25
        elif vdry < VOLUME_DRYUP_THRESHOLD: score += 15
        elif vdry < 0.90:                   score += 5
        # 3. Price range tightness last 10 bars (20 pts)
        if len(close) >= 10:
            rng10 = (float(np.max(high[-10:])) - float(np.min(low[-10:]))) / max(close[-10], 1e-9)
            if   rng10 < 0.04: score += 20
            elif rng10 < 0.08: score += 13
            elif rng10 < 0.12: score += 7
        # 4. Price position within 10-bar range (15 pts)
        if len(close) >= 10:
            lo10 = float(np.min(low[-10:]))
            hi10 = float(np.max(high[-10:]))
            span = hi10 - lo10
            if span > 0:
                score += ((close[-1] - lo10) / span) * 15
        # 5. Higher lows count (15 pts)
        if len(low) >= 10:
            hl = sum(low[-i] > low[-i-1] for i in range(1, min(10, len(low)-1)))
            score += hl * 1.5
        return min(100.0, score)

    @staticmethod
    def _breakout_signal(
        close: np.ndarray, high: np.ndarray,
        volume: np.ndarray, avg_vol_50: float,
    ) -> Tuple[bool, float, float]:
        lookback  = min(30, len(close)-1)
        pivot     = float(np.max(high[-lookback-1:-1]))
        vol_ratio = volume[-1] / max(avg_vol_50, 1.0)
        return close[-1] > pivot*1.001, float(vol_ratio), float(pivot)

    @staticmethod
    def _compute_stop(close: np.ndarray, low: np.ndarray,
                      atr: float, pivot: float) -> float:
        vcp_low  = float(np.min(low[-20:])) if len(low) >= 20 else float(np.min(low))
        atr_stop = float(close[-1]) - 2.0*atr
        return max(vcp_low*0.99, atr_stop)

    @staticmethod
    def _compute_target(close: np.ndarray, high: np.ndarray,
                        low: np.ndarray, entry: float, stop: float) -> float:
        base_high   = float(np.max(high[-60:])) if len(high) >= 60 else entry
        base_low    = float(np.min(low[-60:]))  if len(low)  >= 60 else stop
        base_height = base_high - base_low
        return max(entry + base_height, entry + 2.0*abs(entry-stop))

    # ── Stage 7: RS Line new high ─────────────────────────────────────────────

    @staticmethod
    def _rs_line_score(
        close: np.ndarray, spy_close: Optional[np.ndarray]
    ) -> Tuple[bool, float]:
        """True + score if price/SPY ratio is at or near its 52-week high."""
        if spy_close is None or len(spy_close) < 63 or len(close) < 63:
            return False, 50.0   # neutral / unknown
        min_len  = min(len(close), len(spy_close))
        rs_line  = close[-min_len:] / np.maximum(spy_close[-min_len:], 1e-9)
        lookback = min(252, len(rs_line))
        rs_52w_high = float(np.max(rs_line[-lookback:]))
        rs_now      = float(rs_line[-1])
        pct_of_high = rs_now / max(rs_52w_high, 1e-9)
        at_new_high = pct_of_high >= 0.98
        score       = min(100.0, pct_of_high * 100)
        return at_new_high, score

    # ── Stage 8: OBV divergence ───────────────────────────────────────────────

    @staticmethod
    def _obv_divergence(
        close: np.ndarray, volume: np.ndarray
    ) -> Tuple[bool, float]:
        """OBV rising while price flat/consolidating = silent institutional accumulation."""
        if len(close) < 20:
            return False, 0.0
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if   close[i] > close[i-1]: obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]: obv[i] = obv[i-1] - volume[i]
            else:                        obv[i] = obv[i-1]
        # Last 20 bars: OBV slope vs price slope
        x          = np.arange(20, dtype=float)
        obv_norm   = obv[-20:] / max(abs(float(obv[-20])), 1.0)
        price_norm = close[-20:] / max(close[-20], 1e-9)
        obv_slope   = float(np.polyfit(x, obv_norm,   1)[0])
        price_slope = float(np.polyfit(x, price_norm, 1)[0])
        obv_rising  = obv_slope   >  0.001
        price_flat  = abs(price_slope) <  0.0015
        divergence  = obv_rising and price_flat
        score = 0.0
        if obv_rising:  score += 50.0
        if price_flat:  score += 30.0
        if divergence:  score += 20.0
        return divergence, min(100.0, score)

    # ── Stage 9: Pocket Pivot ─────────────────────────────────────────────────

    @staticmethod
    def _pocket_pivot(close: np.ndarray, volume: np.ndarray) -> bool:
        """Morales/Kacher: up day where vol > highest down-day vol of prior 10 sessions."""
        if len(close) < 12 or len(volume) < 12:
            return False
        if close[-1] <= close[-2]:
            return False   # must be an up day
        today_vol = float(volume[-1])
        max_down_vol = 0.0
        for i in range(2, min(12, len(close))):
            if close[-i] < close[-i-1]:
                max_down_vol = max(max_down_vol, float(volume[-i]))
        if max_down_vol == 0:
            return False   # no down days — bullish but not a classic PP
        return today_vol > max_down_vol

    # ── Stage 10: Sector rotation ─────────────────────────────────────────────

    def _sector_rotation_score(
        self, sector: str, sector_etf_ohlcv: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, float]:
        """Sector ETF above 30-week SMA and trending up = macro tailwind."""
        etf = _SECTOR_ETF_MAP.get(sector)
        if not etf or etf not in sector_etf_ohlcv:
            return False, 50.0   # neutral if sector unknown
        df  = sector_etf_ohlcv[etf]
        ec  = df["close"].values.astype(float)
        if len(ec) < 150:
            return False, 50.0
        sma150    = float(np.mean(ec[-150:]))
        sma150_21 = float(np.mean(ec[-171:-21])) if len(ec) >= 171 else sma150
        above_sma = ec[-1] > sma150
        trending  = sma150 > sma150_21
        score = 0.0
        if above_sma:  score += 50.0
        if trending:   score += 30.0
        if len(ec) >= 63:
            etf_roc63 = (ec[-1]/ec[-63] - 1)*100
            if   etf_roc63 > 5:  score += 20.0
            elif etf_roc63 > 0:  score += 10.0
        return above_sma and trending, min(100.0, score)

    # ── Stage 11: EPS acceleration ────────────────────────────────────────────

    @staticmethod
    def _earnings_acceleration(eps_growth: float, eps_5yr: float) -> bool:
        """True if current-year EPS growth materially outpaces the 5-yr avg.
        Proxy for the O'Neil acceleration pattern (25→50→80%).
        """
        if eps_growth < 25:
            return False   # must show meaningful growth at all
        if eps_5yr <= 0:
            return eps_growth >= 50  # no 5yr baseline: require strong absolute
        return eps_growth >= eps_5yr * 1.5  # 50% faster than trailing avg

    # ── Stage 12: Earnings proximity (optional API call) ─────────────────────

    @staticmethod
    def _earnings_proximity(sym: str) -> int:
        """Days to next earnings report.  Returns -1 if unavailable."""
        try:
            import yfinance as yf
            t   = yf.Ticker(sym)
            cal = t.calendar
            if cal is None:
                return -1
            # yfinance ≥ 0.2.x returns a dict; older versions return DataFrame
            if isinstance(cal, dict):
                raw = cal.get("Earnings Date") or cal.get("earningsDate") or []
                raw = raw if hasattr(raw, "__iter__") else [raw]
                dates = [pd.Timestamp(d) for d in raw if d is not None]
            elif hasattr(cal, "empty") and not cal.empty:
                col   = "Earnings Date"
                col   = col if col in cal.columns else cal.columns[0]
                dates = [pd.Timestamp(cal[col].iloc[0])]
            else:
                return -1
            if not dates:
                return -1
            days = (dates[0].normalize() - pd.Timestamp.now().normalize()).days
            return max(-1, int(days))
        except Exception:
            return -1

    # ── Stage 13: Options flow (optional API call) ────────────────────────────

    @staticmethod
    def _options_flow_score(sym: str) -> Tuple[bool, float]:
        """Detect unusual OTM call activity — smart-money fingerprint."""
        try:
            import yfinance as yf
            t    = yf.Ticker(sym)
            exps = t.options
            if not exps:
                return False, 0.0
            # Use the nearest expiry (most active)
            chain = t.option_chain(exps[0])
            calls = chain.calls
            if calls is None or len(calls) == 0:
                return False, 0.0
            # Current price
            price = float(calls["lastPrice"].dropna().head(1).values[0]) if len(calls) else 0.0
            fi    = t.fast_info
            try:
                price = float(fi.last_price) if price == 0 else price
            except Exception:
                pass
            # OTM calls (strike > 2% above current price)
            if price > 0:
                otm = calls[calls["strike"] > price * 1.02].copy()
            else:
                otm = calls.copy()
            if len(otm) == 0:
                return False, 0.0
            otm["vol_oi"] = (
                otm["volume"].fillna(0) /
                otm["openInterest"].clip(lower=1).fillna(1)
            )
            unusual = otm[(otm["vol_oi"] > 2.0) & (otm["volume"].fillna(0) > 500)]
            has_unusual = len(unusual) > 0
            score       = min(100.0, len(unusual) * 25.0)
            return has_unusual, score
        except Exception:
            return False, 0.0

    # ── Fundamental helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_finviz_fundamentals(row: dict) -> dict:
        def safe_float(val, default=0.0) -> float:
            if val in (None, "", "-", "N/A"): return default
            try:
                return float(str(val).replace("%","").replace("B","")
                             .replace("M","").replace("T","").strip())
            except Exception:
                return default

        def parse_mktcap(val) -> float:
            if not val or str(val) in ("-","N/A",""): return 0.0
            s = str(val).strip().upper()
            try:
                if "T" in s: return float(s.replace("T","")) * 1000
                if "B" in s: return float(s.replace("B",""))
                if "M" in s: return float(s.replace("M","")) / 1000
                return float(s)
            except Exception:
                return 0.0

        eps_qoq = safe_float(row.get("EPS growth qtr over qtr",
                                     row.get("EPS growth this year", 0)))
        eps_5yr = safe_float(row.get("EPS growth past 5 years", 0))
        rev_qoq = safe_float(row.get("Sales growth qtr over qtr",
                                     row.get("Sales growth past 5 years", 0)))
        sf  = safe_float(row.get("Short Interest Share", row.get("Short Float", 0)))
        dtc = safe_float(row.get("Short Interest Ratio", 0))

        return {
            "eps_growth_pct":    eps_qoq,
            "eps_5yr_pct":       eps_5yr,
            "revenue_growth_pct":rev_qoq,
            "short_float_pct":   sf,
            "days_to_cover":     dtc,
            "sector":            str(row.get("Sector", "") or ""),
            "industry":          str(row.get("Industry", "") or ""),
            "market_cap_b":      parse_mktcap(
                row.get("Market Cap.", row.get("Market Cap", "0"))),
        }

    @staticmethod
    def _check_insider_buying(sym: str) -> bool:
        try:
            from edgar import Company
            company  = Company(sym)
            filings  = company.get_filings(form="4")
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
                if hasattr(doc, "transactions"):
                    for tx in doc.transactions:
                        if getattr(tx, "transaction_code", "") == "P":
                            return True
        except Exception:
            pass
        return False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _classify_setup(
    passes_trend:     bool,
    vcp_detected:     bool,
    breakout_confirmed:bool,
    rs_pct:           float,
    mom63:            float,
    pocket_pivot:     bool = False,
    rs_line_high:     bool = False,
) -> str:
    if passes_trend and vcp_detected and breakout_confirmed:
        return "VCP + Stage2 BREAKOUT"
    if passes_trend and vcp_detected and pocket_pivot:
        return "VCP + Stage2 (pocket pivot)"
    if passes_trend and vcp_detected:
        return "VCP + Stage2 (coiling)"
    if passes_trend and breakout_confirmed:
        return "Stage2 Breakout"
    if vcp_detected and breakout_confirmed:
        return "VCP Breakout"
    if rs_line_high and passes_trend:
        return "RS Line New High + Stage2"
    if passes_trend and pocket_pivot:
        return "Stage2 (pocket pivot)"
    if passes_trend:
        return "Stage2 (base forming)"
    if vcp_detected:
        return "VCP (pre-breakout)"
    if rs_pct >= 90 and mom63 > 20:
        return "RS Leader (trending)"
    return "Emerging Setup"


def _build_note(
    trend_n:    int,  vcp_score:     float, vol_ratio:     float,
    rs_pct:     float,mom5:          float, mom63:         float,
    eps_growth: float,short_float:   float, insider_buy:   bool,
    breakout_confirmed: bool,
    atr_contraction:    float,
    rs_line_high:       bool  = False,
    pocket_pivot:       bool  = False,
    obv_divergence:     bool  = False,
    sector_aligned:     bool  = False,
    earn_days:          int   = -1,
    eps_accel:          bool  = False,
    opts_unusual:       bool  = False,
) -> str:
    parts = [f"Trend {trend_n}/8"]
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
    if rs_line_high:
        parts.append("RS line ★ new high")
    if pocket_pivot:
        parts.append("Pocket Pivot ⚑")
    if obv_divergence:
        parts.append("OBV accum ↑")
    if sector_aligned:
        parts.append("sector ✓")
    if eps_accel:
        parts.append("EPS accel ↑")
    if eps_growth >= 50:
        parts.append(f"EPS +{eps_growth:.0f}%")
    elif eps_growth >= 25:
        parts.append(f"EPS +{eps_growth:.0f}%")
    if opts_unusual:
        parts.append("options flow ⚡")
    if 10 <= earn_days <= 45:
        parts.append(f"earnings in {earn_days}d")
    if abs(mom5)  > 3:   parts.append(f"{mom5:+.1f}% 5d")
    if abs(mom63) > 15:  parts.append(f"{mom63:+.1f}% 13wk")
    if short_float >= 15: parts.append(f"short {short_float:.0f}% (squeeze)")
    elif short_float >= 10: parts.append(f"short {short_float:.0f}%")
    if insider_buy: parts.append("insider buy")
    return "  ·  ".join(parts)


# ── Fallback universe (~350 high-quality NYSE/NASDAQ stocks) ─────────────────
_FALLBACK_UNIVERSE: List[str] = [
    "SPY","QQQ","IWM","DIA","MDY","VTI",
    # Mega-cap tech
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","AVGO","QCOM",
    "TXN","AMAT","LRCX","KLAC","MU","INTC","ARM","SMCI",
    # Software / cloud
    "CRM","NOW","SNOW","DDOG","NET","CRWD","PANW","ZS","OKTA","HUBS",
    "GTLB","PATH","MDB","ESTC","CFLT","BILL","VEEV","WDAY","ADBE","ORCL",
    # Financials
    "JPM","GS","MS","BAC","WFC","C","BLK","SCHW","V","MA","AXP","COF",
    # Healthcare / biotech
    "LLY","NVO","ABBV","JNJ","UNH","MRNA","REGN","VRTX","GILD","BMY",
    "RXRX","RARE","SANA","BEAM","EDIT","NTLA","CRSP","IONS",
    # Consumer / retail
    "COST","HD","LOW","TGT","WMT","NKE","LULU","DPZ","CMG","MCD",
    # Industrials / defense
    "CAT","DE","BA","RTX","LMT","NOC","GD","HON","MMM","GE","ETN","EMR",
    # Energy
    "XOM","CVX","COP","SLB","HAL","MPC","VLO","PSX","DVN","FANG",
    # Materials
    "NEM","GOLD","FCX","MP","VALE","RIO","BHP","SCCO","CLF","NUE",
    # Real estate / utilities
    "AMT","PLD","EQIX","SPG","O","DLR","PSA",
    # Crypto / blockchain
    "COIN","MSTR","MARA","RIOT","CLSK","IBIT","GBTC","FBTC","BITB",
    # High-momentum growth
    "PLTR","HOOD","SOFI","IONQ","RKLB","ACHR","JOBY","RXRX","DNA","SOUN",
    "BBAI","LUNR","RDW","ASTS","RCAT","LILM","WOLF","AEHR","OUST",
    "AEVA","LAZR","MVIS","LIDR",
    # Leveraged ETFs
    "TQQQ","SQQQ","UPRO","SPXS","SOXL","SOXS","LABU","FNGU","FNGD",
    # Volatility
    "VXX","UVXY",
    # Additional NYSE leaders
    "BRK-B","JNJ","PG","KO","PEP","PM","MO","T","VZ","CMCSA","DIS",
    "NFLX","SPOT","RBLX","U","MTCH","ZM","PTON","LYFT","UBER","DASH",
    "ABNB","BKNG","EXPE","MAR","HLT","LVS","WYNN","MGM",
    "F","GM","LCID","RIVN","NIO","LI","XPEV","BIDU","JD","PDD","BABA",
]
