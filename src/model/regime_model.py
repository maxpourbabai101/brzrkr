"""
RegimeAwareModel — statistically grounded replacement for _TransformerVol.

Problems with the old stub:
  - composite = sma_spread + tiny_rsi_tilt
    sma_spread is usually ±0.002, so confidence = 0.50 + 5×0.002 = 0.51
    — nearly always returns 0.51, contributing almost nothing to the ensemble
  - Doesn't actually use regime information

New model:
  - Detects market regime: trending / ranging / volatile
  - Applies regime-appropriate strategy:
      trending  → follow the trend (momentum)
      ranging   → mean-reversion at band extremes
      volatile  → reduce conviction, wait for clarity
  - Uses: ADX, Bollinger bandwidth, realised vol percentile, EMA slope
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeAwareModel:
    """Regime-classified directional model (replaces _TransformerVol)."""

    label = "transformer"

    def predict(self, features: Any) -> Dict[str, Any]:
        try:
            close = features["close"].astype(float)
            high  = features["high"].astype(float)  if "high"  in features.columns else close * 1.01
            low   = features["low"].astype(float)   if "low"   in features.columns else close * 0.99

            if len(close) < 30:
                return {"direction": "long", "expected_return_pct": 0.0,
                        "iv_change_pct": 0.0, "confidence": 0.50}

            price = float(close.iloc[-1])

            # ── Realised vol percentile ──────────────────────────────────
            rets    = close.pct_change().dropna()
            rv20    = float(rets.tail(20).std() * np.sqrt(252))
            rv_hist = rets.rolling(20).std() * np.sqrt(252)
            rv_pct  = float((rv20 > rv_hist.dropna()).mean()) if len(rv_hist.dropna()) > 1 else 0.50

            # ── ADX + DI (trend strength and direction) ──────────────────
            tr_df  = pd.concat([high - low,
                                 (high - close.shift(1)).abs(),
                                 (low  - close.shift(1)).abs()], axis=1)
            tr     = tr_df.max(axis=1)
            atr14  = tr.rolling(14).mean().replace(0, np.nan)
            dm_p   = high.diff().clip(lower=0)
            dm_m   = (-low.diff()).clip(lower=0)
            di_p   = dm_p.rolling(14).mean() / atr14 * 100
            di_m   = dm_m.rolling(14).mean() / atr14 * 100
            dx     = (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan) * 100
            adx    = float(dx.rolling(14).mean().iloc[-1] or 20.0)
            adx_up = float(di_p.iloc[-1] or 0) > float(di_m.iloc[-1] or 0)

            # ── Bollinger bandwidth (ranging detector) ───────────────────
            sma20    = close.rolling(20).mean()
            std20    = close.rolling(20).std()
            bb_width = float(((4 * std20) / sma20.replace(0, np.nan)).iloc[-1] or 0.05)

            # ── EMA slope ────────────────────────────────────────────────
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean() if len(close) >= 50 else ema20
            above_ema20 = price > float(ema20.iloc[-1])
            ema_bullish  = float(ema20.iloc[-1]) > float(ema50.iloc[-1])

            # ── Regime classification + signal ───────────────────────────
            is_volatile = rv_pct > 0.80 or rv20 > 0.35
            is_trending = adx > 25
            is_ranging  = bb_width < 0.04 and adx < 20

            if is_volatile:
                direction  = "long" if above_ema20 and ema_bullish else "short"
                confidence = float(min(0.50 + (adx - 15) / 120, 0.62))

            elif is_trending:
                direction  = "long" if adx_up else "short"
                # Strong trend (ADX > 35) gets more conviction
                edge       = min((adx - 25) / 40.0, 0.38)
                alignment  = 1.20 if (adx_up == above_ema20 == ema_bullish) else (
                             1.05 if (adx_up == above_ema20) else 0.80)
                confidence = float(min(0.55 + edge * alignment, 0.93))

            elif is_ranging:
                zscore = float((price - float(sma20.iloc[-1])) /
                               (float(std20.iloc[-1]) + 1e-9))
                direction  = "long" if zscore < -0.6 else "short"
                # Deep z-score reversals get more confidence
                confidence = float(min(0.52 + abs(zscore) * 0.10, 0.82))

            else:   # ambiguous
                direction  = "long" if above_ema20 and ema_bullish else "short"
                confidence = 0.53

            mom5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0

            return {
                "direction": direction,
                "expected_return_pct": mom5,
                "iv_change_pct": rv20 * 0.1,
                "confidence": float(confidence),
            }

        except Exception as exc:
            logger.debug("RegimeAwareModel.predict error: %s", exc)
            return {"direction": "long", "expected_return_pct": 0.0,
                    "iv_change_pct": 0.0, "confidence": 0.50}
