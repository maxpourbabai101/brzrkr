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
MAX_POSITION_PCT = 0.05           # 5% account cap per trade
KELLY_FRACTION = 0.5              # Half‑Kelly
DEFAULT_ATR_MULT_STOP = 2.0
DEFAULT_TP_RR = 2.0               # 2:1 reward / risk
CORRELATION_LIMIT = 0.70
VIX_CRISIS_LEVEL = 35.0
REALIZED_VOL_CRISIS = 0.12        # ~190% annualised — only blocks genuine crash conditions
# (was 0.04 / ~63% ann. which incorrectly blocked TQQQ, TSLA, AMD in normal markets)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def calculate_position_size(
    account_equity: float,
    confidence: float,
    *,
    expected_return_pct: float = 0.01,
    max_loss_pct: float = 0.01,
) -> float:
    """Return notional dollars to allocate to a new trade.

    Implements Kelly half‑criterion sized by model edge, capped at
    ``MAX_POSITION_PCT`` of the account.
    """
    if account_equity <= 0:
        return 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if confidence <= 0.0:
        return 0.0

    if confidence >= 0.5:
        # Full Kelly regime: confidence is interpreted as win probability.
        # Edge ≈ 2p - 1, where p = win probability.
        edge = max(2.0 * confidence - 1.0, 0.0)
        payoff = max(expected_return_pct, 1e-4) / max(max_loss_pct, 1e-4)
        kelly_f = max(confidence - (1.0 - confidence) / payoff, 0.0)
        f = KELLY_FRACTION * kelly_f
    else:
        # Sub-0.5 regime: ensemble confidence represents directional *conviction*
        # not a calibrated win probability.  Use a conservative linear allocation
        # (0% → 1% of equity) so that ANY directional agreement results in a real
        # (small) position rather than zero.
        #
        # BUG FIX — old code returned 0.0 for confidence < 0.5, silently
        # blocking every trade when the ensemble uses directional-agreement
        # confidence scores (typical range 0.25-0.45, not 0.5+).
        edge = 0.0
        kelly_f = 0.0
        f = confidence * 0.06   # 0% at conf=0 → 3% at conf=0.5 (≥$1800 on $100K)

    f = min(f, MAX_POSITION_PCT)
    notional = account_equity * f
    logger.debug(
        "Position sizing: conf=%.3f edge=%.3f kelly_f=%.3f scaled_f=%.4f notional=%.2f",
        confidence, edge, kelly_f, f, notional,
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
