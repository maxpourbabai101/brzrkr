"""Risk management primitives shared by live trading and backtests.

The risk manager is intentionally stateless — each public function
takes everything it needs as arguments and returns a decision. Callers
(signal generator, backtester) wire them together.

Sizing uses a **Kelly half‑criterion** (f = 0.5 * edge) capped at 5 %
of account equity. Edge is approximated from the model's confidence
and expected return.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Iterable, List, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — adjust in config.yaml, not here.
# ---------------------------------------------------------------------------
MAX_POSITION_PCT   = 0.06          # 6 % per trade — 8 slots × 6 % = 48 % max exposure
BASE_RISK_PCT      = 0.01          # risk 1 % of equity per trade
KELLY_FRACTION     = 0.5           # half-Kelly multiplier when Kelly sizing is active
DEFAULT_ATR_MULT_STOP = 2.0
DEFAULT_TP_RR      = 1.5           # 1.5 : 1 reward/risk
CORRELATION_LIMIT  = 0.70
VIX_CRISIS_LEVEL   = 35.0
REALIZED_VOL_CRISIS = 0.04         # 4 % daily = ~63 % ann. — matches config (was 0.12)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def calculate_position_size(
    account_equity: float,
    confidence: float,
    *,
    expected_return_pct: float = 0.01,
    max_loss_pct: float = 0.01,
    historical_win_rate: float = 0.0,   # pass closed-trade win rate for Kelly sizing
    historical_avg_win: float = 0.0,    # avg win as fraction of notional
    historical_avg_loss: float = 0.0,   # avg loss as fraction of notional
) -> float:
    """Return notional USD to allocate to a new trade.

    Sizing strategy (in priority order):

    1. **Kelly half-criterion** — if ≥ 30 closed trades with win/loss stats,
       uses f* = (p × b − q) / b × KELLY_FRACTION, where b = avg_win/avg_loss.
       This ties size directly to measured edge.

    2. **Fixed-fraction fallback** — (equity × BASE_RISK_PCT) / stop_distance_pct.
       Scaled by confidence [0.5×–1.0×]. Used when trade history is thin.

    In both cases, the result is hard-capped at MAX_POSITION_PCT of equity
    (6 % = 48 % max book exposure at 8 positions).

    Examples on a $100 k account, 6 % cap:
      AMD  stop=9.7 %  conf=0.60  → ~$6 000
      DIA  stop=1.7 %  conf=0.65  → $6 000   (capped)
      SPY  stop=3.3 %  conf=0.72  → $6 000   (capped)
    """
    if account_equity <= 0 or max_loss_pct <= 0:
        return 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if confidence <= 0.0:
        return 0.0

    # Floor the stop so a vanishingly tight stop doesn't balloon the size.
    stop_pct = max(float(max_loss_pct), 0.005)   # never tighter than 0.5 %

    # ── Kelly sizing (requires reliable historical stats) ─────────────
    kelly_notional = 0.0
    if (historical_win_rate > 0
            and historical_avg_win > 0
            and historical_avg_loss > 0):
        p = float(historical_win_rate)
        q = 1.0 - p
        b = historical_avg_win / max(historical_avg_loss, 1e-9)
        f_star = (p * b - q) / max(b, 1e-9)       # Kelly fraction of equity
        f_half = max(0.0, f_star) * KELLY_FRACTION  # half-Kelly for safety
        kelly_notional = account_equity * f_half
        logger.debug(
            "Kelly sizing: p=%.2f b=%.2f f*=%.3f half=%.3f notional=%.0f",
            p, b, f_star, f_half, kelly_notional,
        )

    # ── Fixed-fraction fallback ────────────────────────────────────────
    risk_dollars = account_equity * BASE_RISK_PCT
    ff_notional  = risk_dollars / stop_pct

    # Confidence scalar: 0.50 → 0.50×,  1.00 → 1.00×
    # Gives +0 to +50 % adjustment so high-conviction trades are slightly larger.
    conf_scale   = 0.50 + confidence * 0.50
    ff_notional *= conf_scale

    # Blend: use Kelly if available, else fixed-fraction
    notional = kelly_notional if kelly_notional > 0 else ff_notional

    # Hard cap: never exceed MAX_POSITION_PCT of equity.
    notional = min(notional, account_equity * MAX_POSITION_PCT)

    logger.debug(
        "Position sizing: conf=%.3f stop_pct=%.3f kelly=%.0f ff=%.0f → notional=%.2f",
        confidence, stop_pct, kelly_notional, ff_notional, notional,
    )
    return float(notional)


# ---------------------------------------------------------------------------
# Stops / take profits
# ---------------------------------------------------------------------------
def apply_stop_loss(
    entry_price: float,
    direction: str,
    atr: float,
    *,
    atr_mult: float = DEFAULT_ATR_MULT_STOP,
) -> float:
    """Stop‑loss price `atr_mult * ATR` away from entry."""
    if atr <= 0:
        raise ValueError("ATR must be positive.")
    if direction == "long":
        return float(entry_price - atr_mult * atr)
    if direction == "short":
        return float(entry_price + atr_mult * atr)
    raise ValueError(f"Unknown direction: {direction!r}")


def apply_take_profit(
    entry_price: float,
    stop_price: float,
    direction: str,
    *,
    rr: float = DEFAULT_TP_RR,
) -> float:
    """Take‑profit derived from a risk:reward multiple over the stop distance."""
    risk = abs(entry_price - stop_price)
    if direction == "long":
        return float(entry_price + rr * risk)
    if direction == "short":
        return float(entry_price - rr * risk)
    raise ValueError(f"Unknown direction: {direction!r}")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def check_correlation(
    candidate_symbol: str,
    existing_positions: Iterable[Mapping[str, float]],
    correlation_matrix: Mapping[str, Mapping[str, float]],
    *,
    limit: float = CORRELATION_LIMIT,
) -> bool:
    """Return True if candidate is sufficiently uncorrelated with the book."""
    for pos in existing_positions:
        sym = pos.get("symbol")
        if not sym or sym == candidate_symbol:
            continue
        rho = correlation_matrix.get(candidate_symbol, {}).get(sym)
        if rho is None:
            continue
        if abs(rho) >= limit:
            logger.info(
                "Correlation filter: %s vs %s rho=%.2f exceeds %.2f — skip",
                candidate_symbol, sym, rho, limit,
            )
            return False
    return True


def apply_volatility_filter(
    vix: float,
    realized_vol: float,
    *,
    vix_limit: float = VIX_CRISIS_LEVEL,
    rv_limit: float = REALIZED_VOL_CRISIS,
) -> bool:
    """Return True if volatility regime is acceptable for new trades."""
    if vix >= vix_limit:
        logger.warning("Volatility filter blocked: VIX=%.2f >= %.2f", vix, vix_limit)
        return False
    if realized_vol >= rv_limit:
        logger.warning(
            "Volatility filter blocked: realized=%.4f >= %.4f", realized_vol, rv_limit
        )
        return False
    return True


@dataclass
class BlackoutWindow:
    start: time
    end: time
    description: str = ""


def apply_blackout_time(
    current_time: datetime,
    windows: Optional[List[BlackoutWindow]] = None,
) -> bool:
    """Return True if trading is *allowed* (i.e., not in a blackout)."""
    # Default: avoid the first/last 5 minutes of US RTH (UTC, ignoring DST nuance).
    default_windows = [
        BlackoutWindow(time(13, 30), time(13, 35), "open"),
        BlackoutWindow(time(19, 55), time(20, 0), "close"),
    ]
    windows = windows or default_windows
    now_utc = current_time.astimezone(timezone.utc).time()
    for w in windows:
        if w.start <= now_utc <= w.end:
            logger.info("Blackout active: %s (%s–%s)", w.description, w.start, w.end)
            return False
    return True


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------
def monitor_drawdown(equity_curve: Iterable[float], threshold: float = 0.15) -> bool:
    """Return True if current drawdown exceeds `threshold` (e.g., 15%)."""
    arr = np.asarray(list(equity_curve), dtype=float)
    if arr.size == 0:
        return False
    running_max = np.maximum.accumulate(arr)
    dd = (arr - running_max) / running_max
    current = float(dd[-1])
    if current <= -threshold:
        logger.warning("Drawdown breach: %.2f%% <= -%.2f%%", current * 100, threshold * 100)
        return True
    return False
