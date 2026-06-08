"""
ScannerAttributor — tracks which scanners fired on each trade and
computes their marginal contribution (Shapley-ready) to win/loss outcomes.

On every closed trade:
  1. Look up which scanners fired (stored in signal JSON)
  2. Record outcome (win=1, loss=0)
  3. After 30+ trades, compute per-scanner win rate and Shapley value

Output: scanner_weights that CompositeScanner uses instead of equal voting.

Storage: data/scanner_attribution.json
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ATTRIBUTION_PATH = Path("data/scanner_attribution.json")
MIN_TRADES       = 20    # minimum before computing attribution weights
MAX_WEIGHT_RATIO = 3.0   # max weight relative to equal weight (prevent over-concentration)


class ScannerAttributor:
    """Tracks scanner firing patterns and correlates with trade outcomes."""

    def __init__(self) -> None:
        # scanner_name → list of (fired: bool, won: bool) per trade
        self._records: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        if not ATTRIBUTION_PATH.exists():
            return
        try:
            data = json.loads(ATTRIBUTION_PATH.read_text())
            for scanner, records in data.items():
                self._records[scanner] = [(r[0], r[1]) for r in records]
        except Exception:
            pass

    def _save(self) -> None:
        try:
            ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {k: list(v) for k, v in self._records.items()}
            ATTRIBUTION_PATH.write_text(json.dumps(data))
        except Exception:
            pass

    def record_trade(
        self,
        scanners_fired: List[str],
        all_scanners: List[str],
        won: bool,
    ) -> None:
        """Record a completed trade's outcome against which scanners fired."""
        fired_set = set(scanners_fired)
        for scanner in all_scanners:
            fired = scanner in fired_set
            self._records[scanner].append((fired, won))
        self._save()

    def get_scanner_weights(
        self,
        scanner_names: List[str],
    ) -> Dict[str, float]:
        """Return normalised weights for each scanner based on attribution data.
        Falls back to equal weights until MIN_TRADES are available.
        """
        n_scanners  = len(scanner_names)
        equal_weight = 1.0 / n_scanners if n_scanners > 0 else 1.0

        # Check if we have enough data for any scanner
        total_trades = max(
            (len(recs) for recs in self._records.values()), default=0
        )
        if total_trades < MIN_TRADES:
            logger.debug("ScannerAttributor: only %d trades — using equal weights", total_trades)
            return {s: equal_weight for s in scanner_names}

        weights: Dict[str, float] = {}
        for scanner in scanner_names:
            records = self._records.get(scanner, [])
            if not records:
                weights[scanner] = equal_weight
                continue

            # When scanner fired: what was the win rate?
            fired_and_won   = sum(1 for f, w in records if f and w)
            fired_total     = sum(1 for f, w in records if f)
            # When scanner did NOT fire: win rate (baseline)
            no_fire_won     = sum(1 for f, w in records if not f and w)
            no_fire_total   = sum(1 for f, w in records if not f)

            if fired_total < 5:
                weights[scanner] = equal_weight
                continue

            fire_wr    = fired_and_won / fired_total
            nofire_wr  = (no_fire_won / no_fire_total) if no_fire_total > 0 else 0.50

            # Marginal contribution: how much better (or worse) does the trade
            # do when this scanner fires vs when it doesn't?
            marginal = fire_wr - nofire_wr    # positive = scanner adds alpha

            # Convert marginal to weight: centre at equal_weight, scale by signal
            weight = equal_weight * (1.0 + marginal * 4.0)
            weight = float(np.clip(weight, equal_weight / MAX_WEIGHT_RATIO,
                                   equal_weight * MAX_WEIGHT_RATIO))
            weights[scanner] = weight

        # Normalise
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        logger.debug("ScannerAttributor weights: %s",
                     {k: f"{v:.3f}" for k, v in sorted(weights.items())})
        return weights

    def get_stats(self) -> Dict:
        stats = {}
        for scanner, records in self._records.items():
            fired_records = [(f, w) for f, w in records if f]
            if fired_records:
                wr = sum(w for f, w in fired_records) / len(fired_records)
                stats[scanner] = {
                    "times_fired": len(fired_records),
                    "win_rate_when_fired": round(wr, 3),
                }
        return stats


_attributor: Optional[ScannerAttributor] = None


def get_attributor() -> ScannerAttributor:
    global _attributor
    if _attributor is None:
        _attributor = ScannerAttributor()
    return _attributor
