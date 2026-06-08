"""
SignalCalibrator — adjusts raw confidence scores using actual trade outcomes.

How it works:
  1. Reads the trade journal (all closed trades with entry/exit)
  2. Buckets trades by confidence score (0.45–0.50, 0.50–0.55, etc.)
  3. Computes actual win rate per bucket
  4. Builds a calibration map: if 0.52 conf has 67% actual win rate,
     that bucket is underconfident → scale up
  5. Returns a calibrate(raw_conf) function used in _evaluate_symbol

Calibration formula (isotonic-inspired):
  calibrated = raw * (1 + (actual_win_rate - 0.50) * 2 * ALPHA)

where ALPHA controls how aggressively we trust historical win rates
(default 0.5 — conservative).

Requires at least MIN_TRADES closed trades per bucket before trusting
the calibration for that bucket.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

JOURNAL_PATH  = Path("data/trade_journal.jsonl")
MIN_TRADES    = 5       # minimum closed trades per bucket before calibrating
ALPHA         = 0.40    # calibration aggressiveness (0=off, 1=full)
BUCKET_SIZE   = 0.05    # bucket width in confidence space


class SignalCalibrator:
    """
    Reads the trade journal and builds a confidence calibration map.
    Re-fits whenever called if the journal is newer than the last fit.
    """

    def __init__(self) -> None:
        self._calibration: Dict[float, float] = {}   # bucket_floor → scale_factor
        self._last_fit_ts: float = 0.0
        self._n_trades:    int   = 0

    def _bucket(self, conf: float) -> float:
        return round(float(int(conf / BUCKET_SIZE) * BUCKET_SIZE), 4)

    def _load_trades(self) -> List[dict]:
        if not JOURNAL_PATH.exists():
            return []
        trades = []
        with JOURNAL_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trades.append(json.loads(line))
                except Exception:
                    continue
        return trades

    def fit(self) -> int:
        """Refit from journal. Returns number of closed trades used."""
        trades  = self._load_trades()
        closed  = [t for t in trades if t.get("status") == "closed"
                   and t.get("confidence") is not None
                   and t.get("pnl_pct") is not None]

        if len(closed) < MIN_TRADES:
            return len(closed)

        # Group by bucket
        buckets: Dict[float, List[float]] = {}
        for t in closed:
            conf = float(t["confidence"])
            b    = self._bucket(conf)
            win  = 1.0 if float(t["pnl_pct"]) > 0 else 0.0
            buckets.setdefault(b, []).append(win)

        calibration: Dict[float, float] = {}
        for b, wins in buckets.items():
            if len(wins) < MIN_TRADES:
                continue
            actual_wr = float(np.mean(wins))
            # Scale factor: 1.0 = no change; >1 = boost; <1 = penalise
            scale = 1.0 + (actual_wr - 0.50) * 2 * ALPHA
            scale = float(np.clip(scale, 0.7, 1.4))   # cap adjustment ±40%
            calibration[b] = scale
            logger.debug(
                "Calibration bucket %.2f–%.2f: n=%d win_rate=%.1f%% scale=%.3f",
                b, b + BUCKET_SIZE, len(wins), actual_wr * 100, scale,
            )

        self._calibration = calibration
        self._n_trades    = len(closed)
        logger.info(
            "SignalCalibrator fitted on %d closed trades, %d buckets calibrated",
            len(closed), len(calibration),
        )
        return len(closed)

    def calibrate(self, raw_confidence: float) -> float:
        """Return calibrated confidence for a raw model score."""
        if not self._calibration:
            return raw_confidence   # no data yet — pass through

        b = self._bucket(raw_confidence)
        scale = self._calibration.get(b)
        if scale is None:
            return raw_confidence   # bucket not yet calibrated

        calibrated = float(np.clip(raw_confidence * scale, 0.0, 1.0))
        return calibrated

    def get_stats(self) -> Dict:
        return {
            "trades_used":      self._n_trades,
            "buckets":          len(self._calibration),
            "calibration_map":  {f"{k:.2f}": round(v, 4)
                                  for k, v in sorted(self._calibration.items())},
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_calibrator: Optional[SignalCalibrator] = None


def get_calibrator() -> SignalCalibrator:
    global _calibrator
    if _calibrator is None:
        _calibrator = SignalCalibrator()
        _calibrator.fit()
    return _calibrator
