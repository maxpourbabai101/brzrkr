"""
MomentumAgeScanner — detects when a momentum trend has run too long.

Academic basis: Jegadeesh & Titman (1993) show 12-month momentum
strategies mean-revert in months 13–18. Barroso & Santa-Clara (2015)
show momentum crashes happen when momentum has been working longest.

Signal:
  - Count consecutive trading days where 20d return has been positive
  - If > 180 days (≈9 months): momentum is "aged" → reduce conviction
  - If > 252 days (≈12 months): high crash risk → strong short signal
  - If trend just started (< 40 days): fresh momentum → boost long

This is the protection against NVDA-style parabolic collapses.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)

FRESH_MOMENTUM_DAYS    = 40    # < 40d: momentum is fresh, boost
AGED_MOMENTUM_DAYS     = 180   # 180–252d: caution, reduce conviction
CRASH_MOMENTUM_DAYS    = 252   # > 252d: high crash risk


class MomentumAgeScanner:
    """Detects and penalises stale momentum trends to avoid crashes."""

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            close = prices["close"].astype(float)
            if len(close) < 60:
                return None

            # 20-day rolling return — positive = momentum up
            ret20 = close.pct_change(20)

            # Count consecutive days with positive 20d return (momentum streak)
            streak = 0
            for val in reversed(ret20.dropna().values):
                if val > 0:
                    streak += 1
                else:
                    break

            # Also compute negative streak (downtrend age)
            neg_streak = 0
            for val in reversed(ret20.dropna().values):
                if val < 0:
                    neg_streak += 1
                else:
                    break

            current_mom = float(ret20.iloc[-1])

            if streak > CRASH_MOMENTUM_DAYS:
                # Parabolic momentum: very high crash risk
                boost = round(max(-(0.08 + (streak - CRASH_MOMENTUM_DAYS) / 500), -0.15), 3)
                return ScanResult(
                    scanner="momentum_age", symbol=symbol,
                    direction="short", signal_boost=boost,
                    reason=f"Momentum age {streak}d > {CRASH_MOMENTUM_DAYS}d — CRASH RISK (Jegadeesh reversal zone)",
                    metadata={"streak_days": streak, "current_mom_20d": round(current_mom * 100, 2)},
                )

            if AGED_MOMENTUM_DAYS < streak <= CRASH_MOMENTUM_DAYS:
                # Aging momentum: reduce conviction
                boost = round(-(0.03 + (streak - AGED_MOMENTUM_DAYS) / 2000), 3)
                return ScanResult(
                    scanner="momentum_age", symbol=symbol,
                    direction="short", signal_boost=round(max(boost, -0.07), 3),
                    reason=f"Momentum age {streak}d: entering reversal risk zone ({AGED_MOMENTUM_DAYS}–{CRASH_MOMENTUM_DAYS}d)",
                    metadata={"streak_days": streak, "current_mom_20d": round(current_mom * 100, 2)},
                )

            if streak < FRESH_MOMENTUM_DAYS and current_mom > 0 and streak > 5:
                # Fresh momentum: boost conviction
                boost = round(min(0.03 + streak * 0.001, 0.06), 3)
                return ScanResult(
                    scanner="momentum_age", symbol=symbol,
                    direction="long", signal_boost=boost,
                    reason=f"Fresh momentum {streak}d — early in trend, high continuation probability",
                    metadata={"streak_days": streak, "current_mom_20d": round(current_mom * 100, 2)},
                )

            if neg_streak > CRASH_MOMENTUM_DAYS:
                # Long downtrend: potential exhaustion + reversal
                return ScanResult(
                    scanner="momentum_age", symbol=symbol,
                    direction="long", signal_boost=0.05,
                    reason=f"Downtrend age {neg_streak}d — exhaustion bounce likely",
                    metadata={"neg_streak_days": neg_streak},
                )

        except Exception as exc:
            logger.debug("MomentumAgeScanner failed for %s: %s", symbol, exc)

        return None
