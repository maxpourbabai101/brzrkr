"""Generate trade signals from ensemble output + risk management.

A signal is only emitted when the ensemble confidence is at least
``CONFIDENCE_THRESHOLD`` (default 0.75). Below threshold the helper
returns ``None`` so callers can no‑op.

The emitted JSON object follows this contract::

    {
      "asset":       "ES",
      "timestamp":   "2026-05-21T13:35:12Z",
      "direction":   "long" | "short",
      "entry_price": 5234.25,
      "stop_loss":   5215.50,
      "take_profit": 5271.75,
      "position_size_usd": 12500.0,
      "expected_return_pct": 0.0072,
      "iv_change_pct":       0.011,
      "confidence":          0.81,
      "risk_flags": {
          "volatility_ok":  true,
          "correlation_ok": true,
          "blackout_ok":    true
      }
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.risk.risk_manager import (
    apply_blackout_time,
    apply_stop_loss,
    apply_take_profit,
    apply_volatility_filter,
    calculate_position_size,
    check_correlation,
)

logger = logging.getLogger(__name__)

_REGIME_CACHE = Path("data/regime_cache.json")

def _load_regime_bias() -> str:
    """Read side_bias from regime_cache.json. Returns 'both' if unavailable."""
    try:
        return json.loads(_REGIME_CACHE.read_text()).get("side_bias", "both")
    except Exception:
        return "both"

CONFIDENCE_THRESHOLD = 0.75   # data-driven: long WR=66.7%@0.75; matches config.yaml


def generate_signal(
    asset: str,
    model_output: Mapping[str, Any],
    risk_params: Mapping[str, Any],
    *,
    as_json: bool = False,
) -> Optional[Dict[str, Any] | str]:
    """Build (and optionally serialize) a trade signal.

    ``model_output`` must contain at least the keys emitted by
    :class:`EnsemblePredictor` (direction, expected_return_pct,
    iv_change_pct, confidence).

    ``risk_params`` must contain: account_equity, entry_price, atr,
    vix, realized_vol, current_time (datetime), existing_positions
    (iterable), correlation_matrix (dict-of-dict).

    Returns ``None`` if confidence is below threshold or a hard risk
    filter is triggered.
    """
    raw_confidence = float(model_output.get("confidence", 0.0))

    # Apply learned confidence calibration if enough trade history exists.
    try:
        from src.learning.signal_calibrator import get_calibrator
        confidence = get_calibrator().calibrate(raw_confidence)
        if abs(confidence - raw_confidence) > 0.001:
            logger.debug(
                "Confidence calibrated: %.4f → %.4f",
                raw_confidence, confidence,
            )
    except Exception:
        confidence = raw_confidence

    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "Signal suppressed: confidence %.2f below threshold %.2f",
            confidence, CONFIDENCE_THRESHOLD,
        )
        return None

    direction = str(model_output["direction"])

    # ── Fix 1: Regime side-bias filter ────────────────────────────────────
    # Block direction that fights the macro regime. Data: short WR=21% in
    # trending_up. This check is also in TradingAgent._evaluate_symbol but
    # we enforce it here so every call path (scanners, tests, CLIs) respects it.
    side_bias = risk_params.get("side_bias") or _load_regime_bias()
    if side_bias == "long_only" and direction == "short":
        logger.info(
            "Signal suppressed: regime side_bias=long_only blocks %s short (asset=%s)",
            asset, asset,
        )
        return None
    if side_bias == "short_only" and direction == "long":
        logger.info(
            "Signal suppressed: regime side_bias=short_only blocks %s long (asset=%s)",
            asset, asset,
        )
        return None

    # ── Fix 8: Semantic inverse-ETF collision check ────────────────────────
    # Blocks new signal if a conceptually-opposite ETF is already open.
    _INVERSE_PAIRS: Dict[str, str] = {
        "TQQQ": "SQQQ", "SQQQ": "TQQQ",
        "SPXL": "SPXU", "SPXU": "SPXL",
        "TNA":  "TZA",  "TZA":  "TNA",
        "UPRO": "SPXU",
        "LABU": "LABD", "LABD": "LABU",
        "NUGT": "DUST", "DUST": "NUGT",
        "JNUG": "JDST", "JDST": "JNUG",
    }
    existing_positions = risk_params.get("existing_positions", [])
    inverse = _INVERSE_PAIRS.get(asset.upper())
    if inverse:
        existing_symbols = {
            (p if isinstance(p, str) else p.get("symbol", "")).upper()
            for p in existing_positions
        }
        if inverse in existing_symbols:
            logger.info(
                "Signal suppressed: %s inverse pair %s is already open — no redundant exposure",
                asset, inverse,
            )
            return None

    entry = float(risk_params["entry_price"])
    atr = float(risk_params["atr"])
    current_time = risk_params.get("current_time", datetime.now(timezone.utc))

    # ── Fix 7: Earnings / macro event blackout ─────────────────────────────
    try:
        from src.filters.event_blackout import is_blocked as _event_blocked
        event_ok_flag, event_reason = _event_blocked(asset, now=current_time)
        if event_ok_flag:   # is_blocked returns True when BLOCKED
            logger.info("Signal suppressed: event blackout for %s — %s", asset, event_reason)
            return None
    except Exception as _exc:
        logger.debug("Event blackout check error (%s): %s", asset, _exc)

    # Apply risk filters first; bail early if any block.
    vol_ok = apply_volatility_filter(
        vix=float(risk_params.get("vix", 0.0)),
        realized_vol=float(risk_params.get("realized_vol", 0.0)),
    )
    corr_ok = check_correlation(
        candidate_symbol=asset,
        existing_positions=risk_params.get("existing_positions", []),
        correlation_matrix=risk_params.get("correlation_matrix", {}),
    )
    blackout_ok = apply_blackout_time(current_time)

    if not (vol_ok and corr_ok and blackout_ok):
        logger.info(
            "Signal suppressed by risk filters: vol_ok=%s corr_ok=%s blackout_ok=%s",
            vol_ok, corr_ok, blackout_ok,
        )
        return None

    stop = apply_stop_loss(entry, direction, atr)
    take_profit = apply_take_profit(entry, stop, direction)
    position_size = calculate_position_size(
        account_equity=float(risk_params["account_equity"]),
        confidence=confidence,
        expected_return_pct=abs(float(model_output.get("expected_return_pct", 0.01))),
        max_loss_pct=abs((entry - stop) / max(entry, 1e-6)),
    )

    signal: Dict[str, Any] = {
        "asset": asset,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "direction": direction,
        "entry_price": round(entry, 4),
        "stop_loss": round(stop, 4),
        "take_profit": round(take_profit, 4),
        "position_size_usd": round(position_size, 2),
        "expected_return_pct": round(float(model_output.get("expected_return_pct", 0.0)), 6),
        "iv_change_pct": round(float(model_output.get("iv_change_pct", 0.0)), 6),
        "confidence": round(confidence, 4),
        "raw_confidence": round(raw_confidence, 4),
        "risk_flags": {
            "volatility_ok": vol_ok,
            "correlation_ok": corr_ok,
            "blackout_ok": blackout_ok,
        },
    }
    logger.info("Generated signal for %s: %s @ %.2f", asset, direction, entry)
    return json.dumps(signal) if as_json else signal
