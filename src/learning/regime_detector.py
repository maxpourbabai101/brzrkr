"""RegimeDetector — classifies the current market regime and recommends
parameter adjustments for the trading agent.

Regime labels
─────────────
  trending_up    — strong directional uptrend (momentum + low vol)
  trending_down  — strong directional downtrend
  ranging        — oscillating / low-momentum / mean-reverting
  volatile       — high realised vol, wide intraday ranges, spiky VIX

Parameter recommendations
─────────────────────────
Each regime returns a dict that the agent can overlay on its config:
  confidence_threshold  — raise in volatile/bear, lower in trending_up
  stop_pct_multiplier   — widen stops in volatile, tighten in ranging
  tp_pct_multiplier     — extend targets in trends, compress in ranging
  max_positions_factor  — 0.5 in volatile (half the usual cap)
  side_bias             — "long_only" | "short_only" | "both"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_CACHE_PATH = Path("data/regime_cache.json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    label:                   str    = "ranging"
    confidence:              float  = 0.5
    momentum_20d:            float  = 0.0    # price % change over 20 days
    realized_vol_20d:        float  = 0.0    # annualised HV (%)
    atr_pct:                 float  = 0.0    # ATR / price
    bb_width_pct:            float  = 0.0    # Bollinger bandwidth / mid
    vix_level:               float  = 20.0
    vix_percentile_1y:       float  = 0.5
    # Recommended agent parameter overrides
    confidence_threshold:    float  = 0.29
    stop_pct_multiplier:     float  = 1.0
    tp_pct_multiplier:       float  = 1.0
    max_positions_factor:    float  = 1.0
    side_bias:               str    = "both"
    computed_at:             str    = field(default_factory=lambda:
                                        datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Classifies the current market regime using SPY OHLCV + VIX proxy.

    Parameters
    ----------
    lookback_days : int
        How many trading days of history to analyse (default 252 = 1 year).
    """

    def __init__(self, lookback_days: int = 252) -> None:
        self.lookback = lookback_days

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, spy_df: Optional[pd.DataFrame] = None) -> RegimeResult:
        """Detect the current regime.

        Parameters
        ----------
        spy_df : pd.DataFrame, optional
            Pre-fetched SPY OHLCV DataFrame with columns
            [open, high, low, close, volume].  If None the detector will
            try to fetch it live via yfinance.
        """
        df = self._ensure_data(spy_df)
        if df is None or len(df) < 30:
            logger.warning("RegimeDetector: insufficient data — defaulting to 'ranging'")
            return RegimeResult()

        result = self._classify(df)
        self._cache(result)
        return result

    @classmethod
    def load_cached(cls) -> Optional[RegimeResult]:
        """Load the last cached regime (avoids re-fetching on every tick)."""
        if not REGIME_CACHE_PATH.exists():
            return None
        try:
            import json
            d = json.loads(REGIME_CACHE_PATH.read_text())
            return RegimeResult(**{k: v for k, v in d.items()
                                   if k in RegimeResult.__dataclass_fields__})
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(self, df: pd.DataFrame) -> RegimeResult:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        # ── Momentum (20-day return)
        n20 = min(20, len(close) - 1)
        momentum_20d = float((close.iloc[-1] / close.iloc[-n20 - 1] - 1) * 100)

        # ── Momentum (50-day return)
        n50 = min(50, len(close) - 1)
        momentum_50d = float((close.iloc[-1] / close.iloc[-n50 - 1] - 1) * 100)

        # ── Realised volatility (20-day annualised)
        log_rets = np.log(close / close.shift(1)).dropna()
        hv20 = float(log_rets.iloc[-20:].std() * np.sqrt(252) * 100)

        # ── ATR% — average daily range / close
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        atr_pct = float(atr14 / close.iloc[-1] * 100)

        # ── Bollinger bandwidth (20d, 2σ)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_width = float(((sma20 + 2*std20) - (sma20 - 2*std20)).iloc[-1]
                         / sma20.iloc[-1] * 100)

        # ── VIX proxy (use SPY HV21 as a rough VIX substitute)
        vix_proxy = hv20

        # ── VIX 1-year percentile  (using our own HV series)
        hv_series = log_rets.rolling(21).std() * np.sqrt(252) * 100
        vix_pct_1y = float(
            (hv_series.dropna().iloc[-1] >
             hv_series.dropna()).mean()
        ) if len(hv_series.dropna()) > 1 else 0.5

        # ── Regime classification
        label, conf = self._label(
            momentum_20d, momentum_50d, hv20, atr_pct, bb_width, vix_proxy,
        )

        result = RegimeResult(
            label               = label,
            confidence          = conf,
            momentum_20d        = round(momentum_20d, 3),
            realized_vol_20d    = round(hv20, 2),
            atr_pct             = round(atr_pct, 3),
            bb_width_pct        = round(bb_width, 3),
            vix_level           = round(vix_proxy, 2),
            vix_percentile_1y   = round(vix_pct_1y, 3),
        )

        # ── Apply parameter recommendations
        self._apply_recommendations(result)
        return result

    @staticmethod
    def _label(
        mom20: float, mom50: float,
        hv20: float, atr_pct: float,
        bb_width: float, vix: float,
    ) -> tuple[str, float]:
        """Simple rule-based regime classifier."""
        is_high_vol  = hv20 > 30 or vix > 25 or atr_pct > 1.5
        is_trending  = abs(mom20) > 3.0 or abs(mom50) > 6.0
        is_up        = mom20 > 0 and mom50 > 0
        is_ranging   = bb_width < 4.0 and abs(mom20) < 2.0

        if is_high_vol:
            return "volatile", min(0.5 + (hv20 - 30) / 40, 0.95)
        if is_trending and is_up:
            conf = min(0.5 + abs(mom20) / 20, 0.95)
            return "trending_up", conf
        if is_trending and not is_up:
            conf = min(0.5 + abs(mom20) / 20, 0.95)
            return "trending_down", conf
        if is_ranging:
            return "ranging", 0.70
        # Default: mild trending / ambiguous
        if mom20 > 0:
            return "trending_up", 0.55
        return "trending_down", 0.55

    @staticmethod
    def _apply_recommendations(r: RegimeResult) -> None:
        """Mutate *r* to fill in the parameter recommendation fields."""
        if r.label == "trending_up":
            r.confidence_threshold  = 0.28   # strong trend → trade when any agreement
            r.stop_pct_multiplier   = 1.0
            r.tp_pct_multiplier     = 1.4    # let winners run
            r.max_positions_factor  = 1.0
            r.side_bias             = "long_only"  # never short into a bull market

        elif r.label == "trending_down":
            r.confidence_threshold  = 0.30   # selective on longs in down market
            r.stop_pct_multiplier   = 0.8    # tighter stops
            r.tp_pct_multiplier     = 0.9
            r.max_positions_factor  = 0.8
            r.side_bias             = "both"

        elif r.label == "ranging":
            r.confidence_threshold  = 0.29   # need clear directional agreement
            r.stop_pct_multiplier   = 0.85   # tighter stops — chop will kill
            r.tp_pct_multiplier     = 0.75   # take profits quicker
            r.max_positions_factor  = 0.75
            r.side_bias             = "both"

        elif r.label == "volatile":
            r.confidence_threshold  = 0.34   # elevated bar in high-vol regime
            r.stop_pct_multiplier   = 1.5    # wider stops — gaps happen
            r.tp_pct_multiplier     = 1.2
            r.max_positions_factor  = 0.5    # half the usual positions
            r.side_bias             = "both"

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _ensure_data(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is not None and not df.empty:
            return df.tail(self.lookback)
        try:
            import yfinance as yf
            raw = yf.download("SPY", period="1y", interval="1d",
                              progress=False, auto_adjust=True)
            if raw.empty:
                return None
            # yfinance ≥ 0.2 returns MultiIndex columns: ('Close', 'SPY') etc.
            # Flatten to simple lowercase strings either way.
            if isinstance(raw.columns[0], tuple):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            return raw
        except Exception as exc:
            logger.warning("RegimeDetector: yfinance fetch failed — %s", exc)
            return None

    def _cache(self, result: RegimeResult) -> None:
        try:
            import json
            REGIME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            REGIME_CACHE_PATH.write_text(json.dumps(result.to_dict(), indent=2))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

_detector: Optional[RegimeDetector] = None


def get_detector() -> RegimeDetector:
    global _detector
    if _detector is None:
        _detector = RegimeDetector()
    return _detector


def current_regime() -> RegimeResult:
    """Quick helper: load cache if fresh (< 1 h old), else re-detect."""
    cached = RegimeDetector.load_cached()
    if cached:
        try:
            from datetime import timedelta
            age = datetime.now(timezone.utc) - datetime.fromisoformat(cached.computed_at)
            if age.total_seconds() < 3600:
                return cached
        except Exception:
            pass
    return get_detector().detect()
