"""Portfolio-level risk countermeasures.

These extend the per-trade risk_manager with **session and portfolio
state**: things you can only reason about by looking across multiple
trades. The agent calls into a :class:`CountermeasureSet` on every
tick to decide whether new entries are allowed.

Implemented countermeasures
---------------------------

1. **Circuit breaker** — halt new entries after N consecutive losses.
2. **Post-loss cooldown** — block new entries for N minutes after any loss.
3. **Sector concentration cap** — no more than M positions in one sector.
4. **Volatility-scaled sizing** — shrink notional in high-VIX regimes.
5. **Time-of-day filter** — block N minutes around economic releases.
6. **Spread filter** — refuse to chase wide bid/ask spreads.
7. **Liquidity filter** — minimum 20-day average volume.
8. **News blackout** — block trading during a configurable list of UTC
   datetime windows (earnings, FOMC, etc.).
9. **Daily turnover cap** — max N trades per session.
10. **Slippage anomaly killer** — halt if recent fills slip by > X bps.

All are opt-in via constructor flags. The agent's defaults are
intentionally conservative.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — match conservative paper-trading practice
# ---------------------------------------------------------------------------
@dataclass
class CountermeasureConfig:
    # Circuit breaker
    consecutive_loss_limit: int = 3
    # Post-loss cooldown
    cooldown_minutes: int = 30
    # Sector cap
    sector_position_limit: int = 2
    # Volatility regimes
    vix_high_threshold: float = 22.0
    vix_extreme_threshold: float = 30.0
    high_vix_size_multiplier: float = 0.5
    extreme_vix_size_multiplier: float = 0.0    # = no new trades
    # Spread / liquidity
    max_spread_bps: float = 25.0
    min_avg_volume: float = 250_000
    # Daily turnover
    max_trades_per_session: int = 12
    # Slippage killer
    max_slippage_bps: float = 30.0
    # Blackout windows (UTC, repeating daily)
    daily_blackout_windows: List[Tuple[time, time]] = field(default_factory=list)
    # One-off blackout windows (specific UTC datetimes)
    event_blackouts: List[Tuple[datetime, datetime]] = field(default_factory=list)


@dataclass
class CountermeasureSet:
    """Stateful evaluator. One instance per agent session."""

    cfg: CountermeasureConfig = field(default_factory=CountermeasureConfig)

    # Mutable session state.
    _recent_pnl: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _consecutive_losses: int = 0
    _last_loss_ts: Optional[datetime] = None
    _trades_this_session: int = 0
    _slippages_bps: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    # ------------------------------------------------------------------
    # Recording — call after every closed trade / fill
    # ------------------------------------------------------------------
    def record_trade_outcome(self, pnl: float, *,
                              when: Optional[datetime] = None) -> None:
        when = when or datetime.now(timezone.utc)
        self._recent_pnl.append(pnl)
        self._trades_this_session += 1
        if pnl < 0:
            self._consecutive_losses += 1
            self._last_loss_ts = when
        else:
            self._consecutive_losses = 0

    def record_fill_slippage(self, slippage_bps: float) -> None:
        self._slippages_bps.append(abs(slippage_bps))

    def reset_session(self) -> None:
        """Call at the start of a new trading day."""
        self._trades_this_session = 0
        self._consecutive_losses = 0
        self._last_loss_ts = None
        self._recent_pnl.clear()
        self._slippages_bps.clear()

    # ------------------------------------------------------------------
    # Gate — call before every new entry. Returns (allowed, reason).
    # ------------------------------------------------------------------
    def allow_new_entry(
        self,
        *,
        symbol: str,
        sector: Optional[str] = None,
        existing_positions: Iterable[Dict] = (),
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        avg_volume: Optional[float] = None,
        vix: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        now = now or datetime.now(timezone.utc)

        # 1. Circuit breaker.
        if self._consecutive_losses >= self.cfg.consecutive_loss_limit:
            return False, (f"circuit breaker: {self._consecutive_losses} "
                            f"consecutive losses ≥ {self.cfg.consecutive_loss_limit}")

        # 2. Post-loss cooldown.
        if self._last_loss_ts is not None:
            elapsed = (now - self._last_loss_ts).total_seconds() / 60.0
            if elapsed < self.cfg.cooldown_minutes:
                return False, (f"post-loss cooldown active "
                                f"({self.cfg.cooldown_minutes - elapsed:.0f} min left)")

        # 3. Sector concentration.
        if sector and self.cfg.sector_position_limit > 0:
            same_sector = sum(1 for p in existing_positions
                              if (p.get("sector") or "").lower() == sector.lower())
            if same_sector >= self.cfg.sector_position_limit:
                return False, (f"sector cap: already {same_sector} positions "
                                f"in {sector}")

        # 4. Volatility regime.
        if vix is not None:
            if vix >= self.cfg.vix_extreme_threshold:
                return False, f"VIX={vix:.1f} ≥ extreme threshold {self.cfg.vix_extreme_threshold}"

        # 5. Spread filter.
        if bid is not None and ask is not None and bid > 0:
            mid = (bid + ask) / 2.0
            spread_bps = (ask - bid) / mid * 10_000
            if spread_bps > self.cfg.max_spread_bps:
                return False, (f"spread {spread_bps:.1f} bps > max "
                                f"{self.cfg.max_spread_bps}")

        # 6. Liquidity filter.
        if avg_volume is not None and avg_volume < self.cfg.min_avg_volume:
            return False, (f"avg volume {avg_volume:,.0f} < min "
                            f"{self.cfg.min_avg_volume:,.0f}")

        # 7. Daily turnover.
        if self._trades_this_session >= self.cfg.max_trades_per_session:
            return False, (f"session turnover cap reached "
                            f"({self._trades_this_session} trades)")

        # 8. Slippage killer.
        if len(self._slippages_bps) >= 3:
            recent = list(self._slippages_bps)[-3:]
            if all(s > self.cfg.max_slippage_bps for s in recent):
                return False, (f"3 consecutive fills slipped > "
                                f"{self.cfg.max_slippage_bps} bps")

        # 9. Daily blackout windows.
        now_t = now.time()
        for start, end in self.cfg.daily_blackout_windows:
            if start <= now_t <= end:
                return False, f"daily blackout window {start}-{end} active"

        # 10. One-off event blackouts.
        for start, end in self.cfg.event_blackouts:
            if start <= now <= end:
                return False, f"event blackout {start}-{end} active"

        return True, "ok"

    # ------------------------------------------------------------------
    # Size adjustment — call AFTER allow_new_entry returns True
    # ------------------------------------------------------------------
    def adjust_notional(
        self,
        base_notional: float,
        *,
        vix: Optional[float] = None,
    ) -> float:
        """Scale down the notional in high-vol regimes. Multiplier of 0
        effectively kills the trade (the executor will refuse sub-share
        sizes)."""
        if vix is None:
            return base_notional
        if vix >= self.cfg.vix_extreme_threshold:
            return base_notional * self.cfg.extreme_vix_size_multiplier
        if vix >= self.cfg.vix_high_threshold:
            return base_notional * self.cfg.high_vix_size_multiplier
        return base_notional


# ---------------------------------------------------------------------------
# Trailing stop — call from a separate position-management loop
# ---------------------------------------------------------------------------
def update_trailing_stop(
    *,
    direction: str,
    entry_price: float,
    current_price: float,
    current_stop: float,
    trail_pct: float = 0.01,
) -> float:
    """Ratchet the stop in the favorable direction only.

    Long: stop moves UP as price rises, never down.
    Short: stop moves DOWN as price falls, never up.
    """
    if direction == "long":
        new_stop = current_price * (1.0 - trail_pct)
        return max(current_stop, new_stop)
    if direction == "short":
        new_stop = current_price * (1.0 + trail_pct)
        return min(current_stop, new_stop)
    raise ValueError(f"unknown direction: {direction!r}")


def time_stop_breached(
    *,
    opened_at: datetime,
    max_holding_minutes: int = 240,
    now: Optional[datetime] = None,
) -> bool:
    """True if the position has been open longer than max_holding_minutes."""
    now = now or datetime.now(timezone.utc)
    return (now - opened_at).total_seconds() / 60.0 > max_holding_minutes
