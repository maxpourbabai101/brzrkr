"""
MultiFactorMomentum — statistically grounded replacement for the
_PriceMomentum LSTM stub.

Problems with the old stub:
  - confidence = 0.5 + 4 × |20d_return|  maps a 5% move to 90% confidence
    but SIZE of move ≠ PROBABILITY of being right
  - single timeframe (20d only) — misses short-term vs long-term divergence
  - no volume confirmation, no trend quality score

New model uses:
  - Multi-lag momentum (5d 30%, 10d 25%, 20d 25%, 50d 20%)
  - Volume confirmation multiplier (price up on volume = more conviction)
  - Trend quality R² (linear regression of price over 20 bars)
  - Volatility-adjusted edge (risk-adjusted momentum = Sharpe-like score)
  - Confidence mapped from risk-adjusted edge, capped and floor-bound
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MultiFactorMomentum:
    """Statistically grounded multi-factor momentum model (replaces _PriceMomentum)."""

    label = "lstm"

    def predict(self, features: Any) -> Dict[str, Any]:
        try:
            close  = features["close"].astype(float)
            volume = features["volume"].astype(float) if "volume" in features.columns else None

            if len(close) < 21:
                return {"direction": "long", "expected_return_pct": 0.0,
                        "iv_change_pct": 0.0, "confidence": 0.50}

            # ── Multi-lag momentum ───────────────────────────────────────
            def safe_ret(n: int) -> float:
                if len(close) <= n:
                    return 0.0
                return float((close.iloc[-1] / close.iloc[-n - 1] - 1))

            ret5  = safe_ret(5)
            ret10 = safe_ret(10)
            ret20 = safe_ret(20)
            ret50 = safe_ret(50) if len(close) > 50 else ret20

            # Weighted composite: shorter = timing, longer = trend filter
            mom_composite = (ret5 * 0.30 + ret10 * 0.25
                             + ret20 * 0.25 + ret50 * 0.20)

            # ── Volume confirmation ──────────────────────────────────────
            vol_mult = 1.0
            if volume is not None and len(volume) >= 21:
                avg_vol  = float(volume.tail(21).iloc[:-1].mean())
                curr_vol = float(volume.iloc[-1])
                vol_ratio = curr_vol / max(avg_vol, 1.0)
                price_up  = float(close.iloc[-1]) > float(close.iloc[-2])
                if vol_ratio > 1.5 and price_up:
                    vol_mult = 1.20    # accumulation signal
                elif vol_ratio > 1.5 and not price_up:
                    vol_mult = 0.82    # distribution signal

            # ── Trend quality (R² of 20-bar linear regression) ──────────
            trend_r2 = 0.5
            if len(close) >= 20:
                x   = np.arange(20, dtype=float)
                y   = close.tail(20).values.astype(float)
                if y.std() > 1e-9:
                    corr = float(np.corrcoef(x, y)[0, 1])
                    trend_r2 = corr ** 2   # 0=random walk, 1=perfect trend

            # ── Volatility-adjusted edge (risk-adjusted momentum) ────────
            vol_20 = float(close.pct_change().tail(20).std() or 0.01)
            risk_adj = mom_composite / max(vol_20, 0.005)

            # ── Confidence mapping ───────────────────────────────────────
            # edge = |risk_adj_mom| scaled to [0, 0.35], modified by vol+trend
            raw_edge  = min(abs(risk_adj) * 0.40, 0.35)
            quality   = vol_mult * (0.60 + trend_r2 * 0.40)   # 0.60–1.00
            confidence = float(min(0.50 + raw_edge * quality, 0.88))

            direction = "long" if mom_composite >= 0 else "short"

            return {
                "direction": direction,
                "expected_return_pct": float(mom_composite * 100),
                "iv_change_pct": 0.0,
                "confidence": confidence,
            }

        except Exception as exc:
            logger.debug("MultiFactorMomentum.predict error: %s", exc)
            return {"direction": "long", "expected_return_pct": 0.0,
                    "iv_change_pct": 0.0, "confidence": 0.50}
